#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/buffer.h>
#include <unordered_set>

namespace TVM {
namespace ir {

class InfoCollector final : public IRVisitor {
  public:
    explicit InfoCollector() {}

    void Visit(const NodeRef& node) final {
      // if (visited_.count(node.get()) != 0) return;
      // visited_.insert(node.get());
      IRVisitor::Visit(node);
    }

    void Visit_(const Variable *op) final {
      LOG(INFO) << "var: " << op->name_hint;
      if (load_flag)
        load_var.push_back(op->name_hint);
      else if (store_flag)
        store_var.push_back(op->name_hint);
    }

    void Visit_(const Load *op) final {
      LOG(INFO) << "Load " << op->type.bytes() << "B " << op->index;
      // get index var
      load_flag = true;
      load_var.clear();
      this->Visit(op->index);
      int bytes = 1;
      for (auto var : load_var) {
        LOG(INFO) << "load " << var << " " << this->loop_var_map[var];
        bytes *= this->loop_var_map[var];
      }
      this->load_cnt += bytes * op->type.bytes();
      load_var.clear();
      load_flag = false;
    }

    void Visit_(const Store *op) final {
      LOG(INFO) << "Store [" << op->index << "] " << op->value.type().bytes() << "B " << op->value;
      // get index var
      store_flag = true;
      store_var.clear();
      this->Visit(op->index);
      int bytes = 1;
      for (auto var : store_var) {
        LOG(INFO) << "store " << var << " " << this->loop_var_map[var];
        bytes *= this->loop_var_map[var];
      }
      this->store_cnt += bytes * op->value.type().bytes();
      store_var.clear();
      store_flag = false;
      // visit value
      this->Visit(op->value);
    }

    void Visit_(const For *op) final {
      int min_val = op->min.as<IntImm>()->value;
      int extent = op->extent.as<IntImm>()->value;
      this->loop_var_map[op->loop_var->name_hint] = extent;
      LOG(INFO) << "For " << op->loop_var << " " << min_val << " " << (min_val + extent);
      this->Visit(op->body);
      this->loop_trip_cnt *= extent;
    }

    void Visit_(const Add *op) final {
      LOG(INFO) << op->a << " + " << op->b;
      if (!store_flag && !load_flag) {
        LOG(INFO) << "count add";
        this->op_cnt++;
      }
      this->Visit(op->a);
      this->Visit(op->b);
    }

    void Visit_(const Sub *op) final {
      LOG(INFO) << op->a << " - " << op->b;
      if (!store_flag && !load_flag)
        this->op_cnt++;
      this->Visit(op->a);
      this->Visit(op->b);
    }

    void Visit_(const Mul *op) final {
      LOG(INFO) << op->a << " * " << op->b;
      if (!store_flag && !load_flag)
        this->op_cnt++;
      this->Visit(op->a);
      this->Visit(op->b);
    }

    void Visit_(const Div *op) final {
      LOG(INFO) << op->a << " / " << op->b;
      if (!store_flag && !load_flag)
        this->op_cnt++;
      this->Visit(op->a);
      this->Visit(op->b);
    }

    int get_store_cnt() { return store_cnt; }
    int get_load_cnt() { return load_cnt; }
    int get_loop_trip_cnt() { return loop_trip_cnt; }
    int get_op_cnt() { return op_cnt; }

  private:
    std::unordered_set<const Node*> visited_;
    std::map<std::string,int> loop_var_map;
    std::vector<std::string> store_var;
    std::vector<std::string> load_var;
    bool store_flag = false;
    bool load_flag = false;
    int store_cnt = 0;
    int load_cnt = 0;
    int loop_trip_cnt = 1;
    int op_cnt = 0;
};

void InfoCollect(const NodeRef& node,
                       std::function<void(int,int,int,int)> fcnt) {
  InfoCollector collector;
  collector.Visit(node);
  std::vector<int> vcnt;
  int c1 = collector.get_store_cnt();
  int c2 = collector.get_load_cnt();
  int c4 = collector.get_loop_trip_cnt();
  int c3 = collector.get_op_cnt() * c4;
  fcnt(c1,c2,c3,c4);
}

}
}