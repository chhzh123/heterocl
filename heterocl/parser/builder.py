# Reference: taichi/python/taichi/lang/ast/transform.py

import ast
from hcl_mlir.ir import (
    Location,
    InsertionPoint,
    FunctionType,
    MemRefType,
    StringAttr,
    AffineExpr,
    AffineMap,
    AffineMapAttr,
)
from hcl_mlir.dialects import (
    hcl as hcl_d,
    func as func_d,
    memref as memref_d,
    affine as affine_d,
    arith as arith_d,
)
from hcl_mlir import get_mlir_type
from ..utils import get_src_loc
from ..context import get_context, get_location
from ..ir.transform import build_for_loops


def get_extra_type_hints_from_str(dtype):
    """
    dtype: HeteroCL type
    """
    if dtype.startswith("int"):
        return "s"
    if dtype.startswith("uint"):
        return "u"
    return "_"


class Builder:
    def __call__(self, ctx, node):
        method = getattr(self, "build_" + node.__class__.__name__, None)
        if method is None:
            error_msg = f'Unsupported node "{node.__class__.__name__}"'
            raise RuntimeError(error_msg)
        with get_context(), get_location():
            return method(ctx, node)


class ASTContext:
    def __init__(self):
        self.ip_stack = []
        self.buffers = {}
        self.induction_vars = {}
        self.top_func = None


class ASTTransformer(Builder):
    @staticmethod
    def build_Name(ctx, node):
        return

    @staticmethod
    def build_AnnAssign(ctx, node):
        ip = ctx.ip_stack[-1]
        filename, lineno = get_src_loc()
        loc = Location.file(filename, lineno, 0)
        type_hint = node.annotation
        type_str = type_hint.value.id
        shape = [x.value for x in type_hint.slice.value.elts]
        ele_type = get_mlir_type(type_str)
        memref_type = MemRefType.get(shape, ele_type)
        alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ip, loc=loc)
        alloc_op.attributes["name"] = StringAttr.get(node.target.id)
        ctx.buffers[node.target.id] = alloc_op.result

    @staticmethod
    def build_Assign(ctx, node):
        build_stmt(ctx, node.value)

    @staticmethod
    def build_Attribute(ctx, node):
        build_stmt(ctx, node.value)

    @staticmethod
    def build_Constant(ctx, node):
        return node

    @staticmethod
    def build_keyword(ctx, node):
        build_stmt(ctx, node.value)
        return node

    @staticmethod
    def build_Tuple(ctx, node):
        build_stmts(ctx, node.elts)
        return node

    @staticmethod
    def build_Subscript(ctx, node):
        build_stmt(ctx, node.value)
        build_stmt(ctx, node.slice)

    @staticmethod
    def build_Index(ctx, node):
        build_stmt(ctx, node.value)

    @staticmethod
    def build_Lambda(ctx, node):
        build_stmt(ctx, node.body)
        return node

    @staticmethod
    def build_For(ctx, node):
        if node.orelse:
            raise RuntimeError("'else' clause for 'for' not supported in HCL kernels")
        ip = ctx.ip_stack[-1]
        grid = [x.value for x in node.iter.args]
        names = [x.id for x in node.target.elts]
        for_loops = build_for_loops(grid, ip, names)
        ivs = [loop.induction_variable for loop in for_loops]
        for name, iv in zip(names, ivs):
            ctx.induction_vars[name] = iv
        ctx.ip_stack.append(InsertionPoint(for_loops[-1].body.operations[0]))
        build_stmts(ctx, node.body)
        ctx.ip_stack.pop()

    @staticmethod
    def build_BinOp(ctx, node):
        lhs = build_stmt(ctx, node.left)
        rhs = build_stmt(ctx, node.right)
        op = {
            ast.Add: arith_d.AddIOp,
            ast.Sub: arith_d.SubIOp,
            ast.Mult: arith_d.MulIOp,
            # FIXME
            ast.Div: lambda l, r: l / r,
            ast.FloorDiv: lambda l, r: l // r,
            ast.Mod: lambda l, r: l % r,
            ast.Pow: lambda l, r: l**r,
            ast.LShift: lambda l, r: l << r,
            ast.RShift: lambda l, r: l >> r,
            ast.BitOr: lambda l, r: l | r,
            ast.BitXor: lambda l, r: l ^ r,
            ast.BitAnd: lambda l, r: l & r,
        }.get(type(node.op))
        # FIXME
        return op(lhs, rhs, ip=ctx.ip_stack[-1])

    @staticmethod
    def build_AugAssign(ctx, node):
        # Compute RHS
        rhs = build_stmt(ctx, node.value)
        # Load LHS
        node.target.ctx = ast.Load()
        lhs = build_stmt(ctx, node.target)
        node.target.ctx = ast.Store()
        # Aug LHS
        op = {
            ast.Add: arith_d.AddIOp,
            ast.Sub: arith_d.SubIOp,
            ast.Mult: arith_d.MulIOp,
            # FIXME
            ast.Div: lambda l, r: l / r,
            ast.FloorDiv: lambda l, r: l // r,
            ast.Mod: lambda l, r: l % r,
            ast.Pow: lambda l, r: l**r,
            ast.LShift: lambda l, r: l << r,
            ast.RShift: lambda l, r: l >> r,
            ast.BitOr: lambda l, r: l | r,
            ast.BitXor: lambda l, r: l ^ r,
            ast.BitAnd: lambda l, r: l & r,
        }.get(type(node.op))
        ip = ctx.ip_stack[-1]
        res = op(lhs, rhs, ip=ip)
        # Store LHS
        build_stmt(ctx, node.target)
        dim_count = len(node.target.slice.value.elts)
        index_exprs = []
        for index in range(dim_count):
            index_exprs.append(AffineExpr.get_dim(index))
        affine_map = AffineMap.get(
            dim_count=dim_count, symbol_count=0, exprs=index_exprs
        )
        affine_attr = AffineMapAttr.get(affine_map)
        ivs = [ctx.induction_vars[x.id] for x in node.target.slice.value.elts]
        store_op = affine_d.AffineStoreOp(
            res.result, ctx.buffers[node.target.value.id], ivs, affine_attr, ip=ip
        )
        return store_op

    @staticmethod
    def build_affine_expr(ctx, node):
        # TODO
        return AffineExpr.get_dim(list(ctx.induction_vars.keys()).index(node.id))

    @staticmethod
    def build_Subscript(ctx, node):
        # Load op
        dim_count = len(node.slice.value.elts)
        index_exprs = []
        for index in range(dim_count):
            index_exprs.append(AffineExpr.get_dim(index))
        ip = ctx.ip_stack[-1]
        if isinstance(node.ctx, ast.Load):
            affine_map = AffineMap.get(
                dim_count=dim_count, symbol_count=0, exprs=index_exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            ivs = [ctx.induction_vars[x.id] for x in node.slice.value.elts]
            load_op = affine_d.AffineLoadOp(
                ctx.buffers[node.value.id], ivs, affine_attr, ip=ip
            )
            return load_op

    @staticmethod
    def build_FunctionDef(ctx, node):
        ip = ctx.ip_stack[-1]
        filename, lineno = get_src_loc()
        loc = Location.file(filename, lineno, 0)
        input_types = []
        input_typehints = []
        arg_names = []
        for arg in node.args.args:
            type_hint = arg.annotation
            type_str = type_hint.value.id
            shape = [x.value for x in type_hint.slice.value.elts]
            ele_type = get_mlir_type(type_str)
            input_typehints.append(get_extra_type_hints_from_str(type_str))
            memref_type = MemRefType.get(shape, ele_type)
            input_types.append(memref_type)
            arg_names.append(arg.arg)
        output_types = []
        output_typehints = []
        type_hint = node.returns
        type_str = type_hint.value.id
        shape = [x.value for x in type_hint.slice.value.elts]
        ele_type = get_mlir_type(type_str)
        output_typehints.append(get_extra_type_hints_from_str(type_str))
        memref_type = MemRefType.get(shape, ele_type)
        output_types.append(memref_type)
        func_type = FunctionType.get(input_types, output_types)
        func_op = func_d.FuncOp(name=node.name, type=func_type, ip=ip, loc=loc)
        # func_op.attributes["sym_visibility"] = StringAttr.get("private")
        func_op.add_entry_block()
        for name, arg in zip(arg_names, func_op.arguments):
            ctx.buffers[name] = arg
        ctx.ip_stack.append(InsertionPoint(func_op.entry_block))
        build_stmts(ctx, node.body)
        ctx.top_func = func_op

    @staticmethod
    def build_Module(ctx, node):
        for stmt in node.body:
            build_stmt(ctx, stmt)
        return None

    @staticmethod
    def build_Call(ctx, node):
        build_stmt(ctx, node.func)
        build_stmts(ctx, node.args)
        build_stmts(ctx, node.keywords)

    @staticmethod
    def build_Return(ctx, node):
        ip = ctx.ip_stack[-1]
        func_d.ReturnOp([ctx.buffers[node.value.id]], ip=ip)
        return


build_stmt = ASTTransformer()


def build_stmts(ctx, stmts):
    for stmt in stmts:
        # if ctx.returned != ReturnStatus.NoReturn or ctx.loop_status() != LoopStatus.Normal:
        #     break
        # else:
        build_stmt(ctx, stmt)
    return stmts
