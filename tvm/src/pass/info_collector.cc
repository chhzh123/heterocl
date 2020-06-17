#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <unordered_set>

namespace TVM {
namespace ir {

class InfoCollector final : public IRVisitor {
  public:
    explicit InfoCollector() {}

    void Visit(const NodeRef& node) final {
      if (visited_.count(node.get()) != 0) return;
      visited_.insert(node.get());
      IRVisitor::Visit(node);
    }

    void Visit_(const Load *op) final {
      LOG(INFO) << "Load " << op->type.bytes() << "B " << op->index;
      // this->Visit(op->index);
      // this->Visit(op->predicate);
      this->load_cnt += op->type.bytes();
    }

    void Visit_(const Store *op) final {
      LOG(INFO) << "Store " << op->value.type().bytes() << "B " << op->value;
      this->store_cnt += op->value.type().bytes();
      this->Visit(op->value);
      // this->Visit(op->index);
      // this->Visit(op->predicate);
    }

    void Visit_(const For *op) final {
      int min_val = op->min.as<IntImm>()->value;
      int extent = op->extent.as<IntImm>()->value;
      LOG(INFO) << "For " << op->loop_var << " " << min_val << " " << (min_val + extent);
      this->Visit(op->body);
      this->loop_trip_cnt *= extent;
    }

    void Visit_(const Add *op) final {
      LOG(INFO) << op->a << " + " << op->b;
      this->op_cnt++;
      this->Visit(op->a);
      this->Visit(op->b);
    }
    void Visit_(const Sub *op) final {
      LOG(INFO) << op->a << " - " << op->b;
      this->op_cnt++;
      this->Visit(op->a);
      this->Visit(op->b);
    }
    void Visit_(const Mul *op) final {
      LOG(INFO) << op->a << " * " << op->b;
      this->op_cnt++;
      this->Visit(op->a);
      this->Visit(op->b);
    }
    void Visit_(const Div *op) final {
      LOG(INFO) << op->a << " / " << op->b;
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
  int c3 = collector.get_op_cnt();
  int c4 = collector.get_loop_trip_cnt();
  fcnt(c1,c2,c3,c4);
}

}
}