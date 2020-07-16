#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <unordered_set>
#include <stack>

namespace TVM {
namespace ir {

class InfoCollector final : public IRVisitor {
  public:
    explicit InfoCollector(std::vector<bool>& opv) {
      std::stack<std::string> pc;
      std::stack<std::string> store;
      std::stack<std::string> load;
      this->node_stack["ProducerConsumer"] = pc;
      this->node_stack["Store"] = store;
      this->node_stack["Load"] = load;
      this->opv = opv;
    }

    void Visit(const NodeRef& node) final {
      // if (visited_.count(node.get()) != 0) return;
      // visited_.insert(node.get());
      IRVisitor::Visit(node);
    }

    void Visit_(const KernelDef* op) final {
      LOG(INFO) << "Kernel Def " << op->name;
      kernel_flag = true;
      this->Visit(op->body);
      kernel_flag = false;
    }

    void Visit_(const ProducerConsumer* op) final {
      if (kernel_flag){
        LOG(INFO) << "producer " << op->is_producer;
        this->node_stack["ProducerConsumer"].push("PC");
        this->Visit(op->body);
        this->node_stack["ProducerConsumer"].pop();
        if (this->node_stack["ProducerConsumer"].empty()){
          this->op_cnt += this->temp_op_cnt;
          this->temp_op_cnt = 0;
        }
      }
    }

    void Visit_(const StreamStmt *op) final { // write
      LOG(INFO) << "streamstmt " << op->buffer_var << " " << op->value;
      return;
    }

    void Visit_(const StreamExpr *op) final { // read
      LOG(INFO) << "streamexpr " << op->buffer_var;
    }

    void Visit_(const Variable *op) final {
      LOG(INFO) << "var: " << op->name_hint;
      if (!this->node_stack["Load"].empty())
        load_var.push_back(op->name_hint);
      else if (!this->node_stack["Store"].empty())
        store_var.push_back(op->name_hint);
    }

    void Visit_(const Load *op) final {
      LOG(INFO) << "Load " << op->buffer_var << " " << op->type.bytes() << "B " << op->index;
      // get index var
      this->node_stack["Load"].push("load");
      load_var.clear();
      this->Visit(op->index);
      int bytes = 1;
      for (auto var : load_var) {
        LOG(INFO) << "load " << var << " " << this->loop_var_map[var];
        bytes *= this->loop_var_map[var];
      }
      this->load_cnt += bytes * op->type.bytes();
      load_var.clear();
      this->node_stack["Load"].pop();
    }

    void Visit_(const Store *op) final {
      LOG(INFO) << "Store " << op->buffer_var << "[" << op->index << "] " << op->value.type().bytes() << "B " << op->value;
      if (op->value.as<StreamExpr>()) // stream
        return;
      // get index var
      this->node_stack["Store"].push("store");
      store_var.clear();
      this->Visit(op->index);
      int bytes = 1;
      for (auto var : store_var) {
        LOG(INFO) << "store " << var << " " << this->loop_var_map[var];
        bytes *= this->loop_var_map[var];
      }
      this->store_cnt += bytes * op->value.type().bytes();
      store_var.clear();
      this->node_stack["Store"].pop();
      // visit value
      this->Visit(op->value);
    }

    void Visit_(const For *op) final {
      if (!kernel_flag)
        return;
      int min_val = op->min.as<IntImm>()->value;
      int extent = op->extent.as<IntImm>()->value;
      this->loop_var_map[op->loop_var->name_hint] = extent;
      LOG(INFO) << "For " << op->loop_var << " " << min_val << " " << (min_val + extent);
      loop_trip_count.push_back(extent);
      this->Visit(op->body);
      this->temp_op_cnt *= extent;
      // calculate latency
      // TODO: Add validity test (e.g. nested pipelined loops)
      bool pipeline_flag = false;
      if (op->for_type == ForType::Pipelined) {
        LOG(INFO) << "Pipelined loop";
        int II = 0, i = 0;
        for (auto key : op->annotate_keys) {
          if (auto str = key.as<StringImm>()) {
            auto initiation_interval = op->annotate_values[i].as<IntImm>();
            if (str->value == "initiation_interval" &&
                initiation_interval != nullptr) {
              II = initiation_interval->value;
              std::cout << "II: " << II << std::endl;
              int lat = 1;
              for (auto trip_count : loop_trip_count)
                lat *= trip_count;
              // (n - 1) * II + T_{L}
              // assume all the ops have latency 1 here
              lat = (lat - 1) * II + 1;
              latency.push_back(lat);
              pipeline_flag = true;
              break;
            }
          }
          i++;
        }
      }
      // if (!pipeline_flag && !op->body.as<For>()) { // the inner-most loop
      //   int lat = 1;
      //   for (auto trip_count : loop_trip_count)
      //     lat *= trip_count;
      //   latency.push_back(lat);
      // }
      loop_trip_count.pop_back();
    }

    void VisitArith(const OpType op, const std::string str,
                    const NodeRef& node1, const NodeRef& node2,
                    std::string opstr) {
      LOG(INFO) << node1 << " " << opstr << " " << node2;
      if (this->opv[int(op)] &&
          this->node_stack["Store"].empty() &&
          this->node_stack["Load"].empty()) {
        LOG(INFO) << "count " << str;
        this->temp_op_cnt++;
      }
      this->Visit(node1);
      this->Visit(node2);
    }

    void Visit_(const Add *op) final {
      VisitArith(OpType::Add,"Add",op->a,op->b,"+");
    }

    void Visit_(const Sub *op) final {
      VisitArith(OpType::Sub,"Sub",op->a,op->b,"-");
    }

    void Visit_(const Mul *op) final {
      VisitArith(OpType::Mul,"Mul",op->a,op->b,"*");
    }

    void Visit_(const Div *op) final {
      VisitArith(OpType::Div,"Div",op->a,op->b,"/");
    }

    void Visit_(const Mod *op) final {
      VisitArith(OpType::Mod,"Mod",op->a,op->b,"%");
    }

    void Visit_(const Call *op) final {
      if (op->is_intrinsic(Call::bitwise_and))
        VisitArith(OpType::And,"bitwise_and",op->args[0],op->args[1],"&");
      else if (op->is_intrinsic(Call::bitwise_or))
        VisitArith(OpType::Or,"bitwise_or",op->args[0],op->args[1],"|");
      else if (op->is_intrinsic(Call::bitwise_not))
        VisitArith(OpType::Not,"bitwise_not",op->args[0],op->args[1],"~");
      else if (op->is_intrinsic(Call::bitwise_xor))
        VisitArith(OpType::Xor,"bitwise_xor",op->args[0],op->args[1],"^");
      else if (op->is_intrinsic(Call::shift_left))
        VisitArith(OpType::LShift,"shift_left",op->args[0],op->args[1],"<<");
      else if (op->is_intrinsic(Call::shift_right))
        VisitArith(OpType::RShift,"shift_right",op->args[0],op->args[1],">>");
    }

    int get_store_cnt() { return store_cnt; }
    int get_load_cnt() { return load_cnt; }
    int get_op_cnt() { return op_cnt; }
    std::vector<int> get_latency() { return latency; }

  private:
    std::unordered_set<const Node*> visited_;
    std::map<std::string, int> loop_var_map;
    std::map<std::string, std::stack<std::string>> node_stack;
    std::vector<int> loop_trip_count;
    std::vector<int> latency;
    std::vector<std::string> store_var;
    std::vector<std::string> load_var;
    std::vector<bool> opv;
    bool kernel_flag = true;
    int store_cnt = 0;
    int load_cnt = 0;
    int temp_op_cnt = 0;
    int op_cnt = 0;
};

void InfoCollect(const NodeRef& node,
                       std::function<void(int,int,int,Array<Expr>)> fcnt,
                       Array<Expr> ops) {
  std::vector<bool> opv(11,false); // # of different ops
  for (auto op : ops)
    opv[op.as<Halide::Internal::IntImm>()->value] = true;
  InfoCollector collector(opv);
  collector.Visit(node);
  std::vector<int> vcnt;
  int c1 = collector.get_store_cnt();
  int c2 = collector.get_load_cnt();
  int c3 = collector.get_op_cnt();
  std::vector<int> latency = collector.get_latency();
  Array<Expr> latency_arr;
  for (auto lat : latency)
    latency_arr.push_back(IntImm::make(Int(32), lat));
  fcnt(c1,c2,c3,latency_arr);
}

}
}