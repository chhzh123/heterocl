#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/buffer.h>
#include <unordered_set>
#include <stack>

namespace TVM {
namespace ir {

class InfoCollector final : public IRVisitor {
  public:
    explicit InfoCollector() {
      std::stack<std::string> pc;
      std::stack<std::string> store;
      std::stack<std::string> load;
      this->node_stack["ProducerConsumer"] = pc;
      this->node_stack["Store"] = store;
      this->node_stack["Load"] = load;
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
      this->Visit(op->body);
      if (op->body.as<For>())
        this->temp_op_cnt *= extent;
    }

    void Visit_(const Add *op) final {
      LOG(INFO) << op->a << " + " << op->b;
      if (this->node_stack["Store"].empty() && this->node_stack["Load"].empty()) {
        LOG(INFO) << "count add";
        this->temp_op_cnt++;
      }
      this->Visit(op->a);
      this->Visit(op->b);
    }

    void Visit_(const Sub *op) final {
      LOG(INFO) << op->a << " - " << op->b;
      if (this->node_stack["Store"].empty() && this->node_stack["Load"].empty())
        this->temp_op_cnt++;
      this->Visit(op->a);
      this->Visit(op->b);
    }

    void Visit_(const Mul *op) final {
      LOG(INFO) << op->a << " * " << op->b;
      if (this->node_stack["Store"].empty() && this->node_stack["Load"].empty()) {
        LOG(INFO) << "count mul";
        this->temp_op_cnt++;
      }
      this->Visit(op->a);
      this->Visit(op->b);
    }

    void Visit_(const Div *op) final {
      LOG(INFO) << op->a << " / " << op->b;
      if (this->node_stack["Store"].empty() && this->node_stack["Load"].empty())
        this->temp_op_cnt++;
      this->Visit(op->a);
      this->Visit(op->b);
    }

    int get_store_cnt() { return store_cnt; }
    int get_load_cnt() { return load_cnt; }
    int get_op_cnt() { return op_cnt; }

  private:
    std::unordered_set<const Node*> visited_;
    std::map<std::string, int> loop_var_map;
    std::map<std::string, std::stack<std::string>> node_stack;
    std::vector<std::string> store_var;
    std::vector<std::string> load_var;
    bool kernel_flag = true;
    int store_cnt = 0;
    int load_cnt = 0;
    int temp_op_cnt = 0;
    int op_cnt = 0;
};

void InfoCollect(const NodeRef& node,
                       std::function<void(int,int,int)> fcnt) {
  InfoCollector collector;
  collector.Visit(node);
  std::vector<int> vcnt;
  int c1 = collector.get_store_cnt();
  int c2 = collector.get_load_cnt();
  int c3 = collector.get_op_cnt();
  fcnt(c1,c2,c3);
}

}
}