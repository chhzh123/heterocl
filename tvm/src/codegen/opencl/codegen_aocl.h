#ifndef TVM_CODEGEN_CODEGEN_AOCL_H_
#define TVM_CODEGEN_CODEGEN_AOCL_H_

# include <tvm/codegen.h>
# include <tvm/packed_func_ext.h>
# include "./codegen_opencl.h"

namespace TVM {
namespace codegen {

class CodeGenAOCL : public CodeGenOpenCL {
  public:
    CodeGenAOCL(){}
    // void AddFunction(LoweredFunc f);
    void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
    void PrintType(Type t, std::ostream& os) override; //NOLINT(*)

    void VisitStmt_(const For* op) override;
    void VisitStmt_(const StreamStmt* op) override;

    void VisitExpr_(const StreamExpr* op, std::ostream& os) override;
};
} // namespace codegen
} // namespace TVM

#endif // TVM_CODEGEN_CODEGEN_AOCL_H_
