# Reference: taichi/python/taichi/lang/ast/transform.py

import ast
from hcl_mlir.ir import (
    Location,
    InsertionPoint,
    FunctionType,
    MemRefType,
    IntegerType,
    F32Type,
    UnitAttr,
    IntegerAttr,
    FloatAttr,
    StringAttr,
    AffineExpr,
    AffineConstantExpr,
    AffineMap,
    AffineMapAttr,
    IntegerSet,
)
from hcl_mlir.dialects import (
    hcl as hcl_d,
    func as func_d,
    memref as memref_d,
    affine as affine_d,
    arith as arith_d,
    math as math_d,
    linalg as linalg_d,
)
from hcl_mlir import get_mlir_type
from ..utils import get_src_loc
from ..context import get_context, get_location
from .transform import build_for_loops


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
    def __init__(self, global_vars):
        self.ip_stack = []
        self.buffers = {}
        self.induction_vars = {}
        self.top_func = None
        self.global_vars = global_vars
        # used for AffineExpr dim counting
        self.dim_count = 0
        self.affine_vars = []

    def set_ip(self, ip):
        if not isinstance(ip, InsertionPoint):
            ip = InsertionPoint(ip)
        self.ip_stack.append(ip)

    def get_ip(self):
        return self.ip_stack[-1]

    def pop_ip(self):
        return self.ip_stack.pop()


class MockOp:
    def __init__(self):
        pass


class MockArg(MockOp):
    def __init__(self, val):
        self.val = val

    @property
    def result(self):
        return self.val


class MockConstant(MockOp):
    def __init__(self, val, ctx):
        self.val = val
        self.ctx = ctx

    @property
    def result(self):
        # TODO: Support other types
        if isinstance(self.val, int):
            dtype = IntegerType.get_signless(32)
            value_attr = IntegerAttr.get(dtype, self.val)
        else:
            dtype = F32Type.get()
            value_attr = FloatAttr.get(dtype, self.val)
        const_op = arith_d.ConstantOp(dtype, value_attr, ip=self.ctx.get_ip())
        return const_op.result


class ASTTransformer(Builder):
    @staticmethod
    def build_Name(ctx, node):
        if node.id in ctx.buffers:
            return ctx.buffers[node.id]
        elif node.id in ctx.global_vars:
            return MockConstant(ctx.global_vars[node.id], ctx)
        else:
            raise RuntimeError("Unsupported Name")

    @staticmethod
    def build_Constant(ctx, node):
        return MockConstant(node.value, ctx)

    @staticmethod
    def build_range_for(ctx, node):
        ip = ctx.get_ip()
        grid = [
            x.value if isinstance(x, ast.Constant) else ctx.global_vars[x.id]
            for x in node.iter.args
        ]
        names = [node.target.id]
        for_loops = build_for_loops(grid, ip, names)
        ivs = [loop.induction_variable for loop in for_loops]
        for name, iv in zip(names, ivs):
            ctx.induction_vars[name] = iv
            ctx.buffers[name] = MockArg(iv)
        ctx.set_ip(for_loops[-1].body.operations[0])
        build_stmts(ctx, node.body)
        ctx.pop_ip()

    @staticmethod
    def build_grid_for(ctx, node):
        ip = ctx.get_ip()
        grid = [
            x.value if isinstance(x, ast.Constant) else ctx.global_vars[x.id]
            for x in node.iter.args
        ]
        if isinstance(node.target, ast.Tuple):
            names = [x.id for x in node.target.elts]
        else:
            names = [node.target.id]
        for_loops = build_for_loops(grid, ip, names)
        ivs = [loop.induction_variable for loop in for_loops]
        for name, iv in zip(names, ivs):
            ctx.induction_vars[name] = iv
            ctx.buffers[name] = MockArg(iv)
        ctx.set_ip(for_loops[-1].body.operations[0])
        build_stmts(ctx, node.body)
        if node.iter.func.attr == "reduction":
            for loop in for_loops:
                loop.attributes["reduction"] = UnitAttr.get()
        ctx.pop_ip()

    @staticmethod
    def build_For(ctx, node):
        if node.orelse:
            raise RuntimeError("'else' clause for 'for' not supported in HCL kernels")
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            return ASTTransformer.build_range_for(ctx, node)
        elif (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and (node.iter.func.attr == "grid" or node.iter.func.attr == "reduction")
        ):
            return ASTTransformer.build_grid_for(ctx, node)
        else:
            raise RuntimeError("Unsupported for loop")

    @staticmethod
    def build_general_binop(ctx, node, lhs, rhs):
        opcls = {
            ast.Add: {
                "float": arith_d.AddFOp,
                "int": arith_d.AddIOp,
                "fixed": hcl_d.AddFixedOp,
            },
            ast.Sub: {
                "float": arith_d.SubFOp,
                "int": arith_d.SubIOp,
                "fixed": hcl_d.SubFixedOp,
            },
            ast.Mult: {
                "float": arith_d.MulFOp,
                "int": arith_d.MulIOp,
                "fixed": hcl_d.MulFixedOp,
            },
            ast.Div: {
                "float": arith_d.DivFOp,
                "int": arith_d.DivSIOp,
                "uint": arith_d.DivUIOp,
                "fixed": hcl_d.DivFixedOp,
            },
            ast.FloorDiv: {
                "float": arith_d.DivFOp,
                "int": arith_d.DivSIOp,
                "uint": arith_d.DivUIOp,
            },
            ast.Mod: {
                "float": arith_d.RemFOp,
                "int": arith_d.RemSIOp,
                "uint": arith_d.RemUIOp,
            },
            ast.Pow: {
                "float": math_d.PowFOp,
                "int": RuntimeError,
                "uint": RuntimeError,
            },
            ast.LShift: {
                "float": RuntimeError,
                "int": arith_d.ShLIOp,
                "uint": RuntimeError,
            },
            ast.RShift: {
                "float": RuntimeError,
                "int": arith_d.ShRUIOp,
                "uint": RuntimeError,
            },
            ast.BitOr: {
                "float": RuntimeError,
                "int": arith_d.OrIOp,
                "uint": RuntimeError,
            },
            ast.BitXor: {
                "float": RuntimeError,
                "int": arith_d.XOrIOp,
                "uint": RuntimeError,
            },
            ast.BitAnd: {
                "float": RuntimeError,
                "int": arith_d.AndIOp,
                "uint": RuntimeError,
            },
        }.get(type(node.op))
        dtype = str(lhs.result.type)
        if dtype.startswith("i"):
            op = opcls["int"]
        elif dtype.startswith("fixed"):
            op = opcls["fixed"]
        elif dtype.startswith("f"):
            op = opcls["float"]
        else:
            raise RuntimeError("Unsupported types for binary op: {}".format(dtype))
        return op(lhs.result, rhs.result, ip=ctx.get_ip())

    @staticmethod
    def build_BinOp(ctx, node):
        lhs = build_stmt(ctx, node.left)
        rhs = build_stmt(ctx, node.right)
        return ASTTransformer.build_general_binop(ctx, node, lhs, rhs)

    @staticmethod
    def build_store(ctx, node, val):
        ip = ctx.get_ip()
        if isinstance(node, ast.Subscript):
            # Note: Python 3.10 will generate different AST for Subscript compared to Python 3.8
            #       3.10 directly flattens the Index node and removes all the None attributes
            #       inside the node
            slice = (
                node.slice if isinstance(node.slice, ast.Tuple) else node.slice.value
            )  # ast.Index
            elts = slice.elts if isinstance(slice, ast.Tuple) else [slice]
            dim_count = len(elts)
            index_exprs = []
            for index in range(dim_count):
                index_exprs.append(AffineExpr.get_dim(index))
            affine_map = AffineMap.get(
                dim_count=dim_count, symbol_count=0, exprs=index_exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            ivs = [ctx.induction_vars[x.id] for x in elts]
            store_op = affine_d.AffineStoreOp(
                val.result, ctx.buffers[node.value.id].result, ivs, affine_attr, ip=ip
            )
            store_op.attributes["to"] = StringAttr.get(node.value.id)
            return store_op
        elif isinstance(node, ast.Name):  # scalar
            affine_map = AffineMap.get(
                dim_count=0, symbol_count=0, exprs=[AffineConstantExpr.get(0)]
            )
            affine_attr = AffineMapAttr.get(affine_map)
            store_op = affine_d.AffineStoreOp(
                val.result, ctx.buffers[node.id].result, [], affine_attr, ip=ip
            )
            store_op.attributes["to"] = StringAttr.get(node.id)
            return store_op
        else:
            raise RuntimeError("Unsupported store")

    @staticmethod
    def build_Assign(ctx, node):
        # Compute RHS
        ip = ctx.get_ip()
        if isinstance(node.value, ast.Name):
            affine_map = AffineMap.get(
                dim_count=0, symbol_count=0, exprs=[AffineConstantExpr.get(0)]
            )
            affine_attr = AffineMapAttr.get(affine_map)
            load = affine_d.AffineLoadOp(
                ctx.buffers[node.value.id].result, [], affine_attr, ip=ip
            )
            rhs = load
        else:
            rhs = build_stmt(ctx, node.value)
        if len(node.targets) > 1:
            raise RuntimeError("Cannot assign to multiple targets")
        # Store LHS
        store_op = ASTTransformer.build_store(ctx, node.targets[0], rhs)
        return store_op

    @staticmethod
    def build_AugAssign(ctx, node):
        ip = ctx.get_ip()
        # Compute RHS
        rhs = build_stmt(ctx, node.value)
        # Load LHS
        if isinstance(node.target, ast.Subscript):
            node.target.ctx = ast.Load()
            lhs = build_stmt(ctx, node.target)
            node.target.ctx = ast.Store()
            lhs.attributes["to"] = StringAttr.get(node.target.value.id)
        elif isinstance(node.target, ast.Name):  # scalar
            affine_map = AffineMap.get(
                dim_count=0, symbol_count=0, exprs=[AffineConstantExpr.get(0)]
            )
            affine_attr = AffineMapAttr.get(affine_map)
            lhs = affine_d.AffineLoadOp(
                ctx.buffers[node.target.id].result, [], affine_attr, ip=ip
            )
            lhs.attributes["from"] = StringAttr.get(node.target.id)
        else:
            raise RuntimeError("Unsupported AugAssign")
        # Aug LHS
        res = ASTTransformer.build_general_binop(ctx, node, lhs, rhs)
        # Store LHS
        store_op = ASTTransformer.build_store(ctx, node.target, res)
        return store_op

    @staticmethod
    def build_affine_expr(ctx, node):
        # TODO
        return AffineExpr.get_dim(list(ctx.induction_vars.keys()).index(node.id))

    @staticmethod
    def build_affine_exp(ctx, node):
        if isinstance(node, ast.Name):
            ctx.dim_count += 1
            ctx.affine_vars.append(node.id)
            return AffineExpr.get_dim(ctx.dim_count - 1)
        elif isinstance(node, ast.BinOp):
            lhs = ASTTransformer.build_affine_exp(ctx, node.left)
            rhs = ASTTransformer.build_affine_exp(ctx, node.right)
            op = {
                ast.Add: lambda l, r: l + r,
                ast.Sub: lambda l, r: l - r,
                ast.Mult: lambda l, r: l * r,
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
            return op(lhs, rhs)
        elif isinstance(node, ast.Constant):
            return AffineConstantExpr.get(node.value)
        else:
            raise RuntimeError("Unsupported affine expression")

    @staticmethod
    def build_Subscript(ctx, node):
        # Load op
        ctx.dim_count = 0
        index_exprs = []
        slice = (
            node.slice if isinstance(node.slice, ast.Tuple) else node.slice.value
        )  # ast.Index
        elts = slice.elts if isinstance(slice, ast.Tuple) else [slice]
        for index in elts:
            index_exprs.append(ASTTransformer.build_affine_exp(ctx, index))
        ip = ctx.get_ip()
        if isinstance(node.ctx, ast.Load):
            affine_map = AffineMap.get(
                dim_count=ctx.dim_count, symbol_count=0, exprs=index_exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            ivs = [ctx.buffers[x].result for x in ctx.affine_vars]
            load_op = affine_d.AffineLoadOp(
                ctx.buffers[node.value.id].result, ivs, affine_attr, ip=ip
            )
            ctx.dim_count = 0
            ctx.affine_vars = []
            return load_op
        else:
            raise RuntimeError("Unsupported Subscript")

    @staticmethod
    def build_AnnAssign(ctx, node):
        ip = ctx.get_ip()
        filename, lineno = get_src_loc()
        loc = Location.file(filename, lineno, 0)
        type_hint = node.annotation
        if node.value is None:
            raise RuntimeError(
                "Please explicitly initialize the buffer with an initial value"
            )
        if not isinstance(node.value, ast.Constant):
            raise RuntimeError("Only support constant value for now")
        if node.value.value != 0:
            raise RuntimeError("Only support zero value for now")
        if isinstance(type_hint, ast.Subscript):
            type_str = type_hint.value.id
            slice = (
                type_hint.slice
                if isinstance(type_hint.slice, ast.Tuple)
                else type_hint.slice.value
            )  # ast.Index
            elts = slice.elts if isinstance(slice, ast.Tuple) else [slice]
            shape = [
                x.value if isinstance(x, ast.Constant) else ctx.global_vars[x.id]
                for x in elts
            ]
            ele_type = get_mlir_type(type_str)
            memref_type = MemRefType.get(shape, ele_type)
            alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ip, loc=loc)
            alloc_op.attributes["name"] = StringAttr.get(node.target.id)
            ctx.buffers[node.target.id] = alloc_op
            cst = MockConstant(node.value.value, ctx)
            with ip:
                linalg_d.fill(cst.result, outs=[alloc_op.result])
        elif isinstance(type_hint, ast.Name):
            type_str = type_hint.id
            # TODO: figure out why zero-shape cannot work
            shape = (1,)
            ele_type = get_mlir_type(type_str)
            memref_type = MemRefType.get(shape, ele_type)
            alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ip, loc=loc)
            alloc_op.attributes["name"] = StringAttr.get(node.target.id)
            ctx.buffers[node.target.id] = alloc_op
            ASTTransformer.build_store(
                ctx, node.target, MockConstant(node.value.value, ctx)
            )
        else:
            raise RuntimeError("Unsupported AnnAssign")

    @staticmethod
    def build_FunctionDef(ctx, node):
        ip = ctx.get_ip()
        filename, lineno = get_src_loc()
        loc = Location.file(filename, lineno, 0)
        input_types = []
        input_typehints = []
        arg_names = []

        def build_type(type_hint):
            if isinstance(type_hint, ast.Subscript):
                type_str = type_hint.value.id
                slice = (
                    type_hint.slice
                    if isinstance(type_hint.slice, ast.Tuple)
                    else type_hint.slice.value
                )  # ast.Index
                elts = slice.elts if isinstance(slice, ast.Tuple) else [slice]
                shape = [
                    x.value if isinstance(x, ast.Constant) else ctx.global_vars[x.id]
                    for x in elts
                ]
                ele_type = get_mlir_type(type_str)
                memref_type = MemRefType.get(shape, ele_type)
            elif isinstance(type_hint, ast.Name):
                type_str = type_hint.id
                memref_type = get_mlir_type(type_str)
            else:
                raise RuntimeError("Unsupported function argument type")
            extra_type_hint = get_extra_type_hints_from_str(type_str)
            return memref_type, extra_type_hint

        # Build input types
        for arg in node.args.args:
            arg_type, extra_type_hint = build_type(arg.annotation)
            input_types.append(arg_type)
            input_typehints.append(extra_type_hint)
            arg_names.append(arg.arg)

        # Build return type
        output_types = []
        output_typehints = []
        output_type, extra_type_hint = build_type(node.returns)
        output_types.append(output_type)
        output_typehints.append(extra_type_hint)

        # Build function
        func_type = FunctionType.get(input_types, output_types)
        func_op = func_d.FuncOp(name=node.name, type=func_type, ip=ip, loc=loc)
        func_op.add_entry_block()
        for name, arg in zip(arg_names, func_op.arguments):
            ctx.buffers[name] = MockArg(arg)
        ctx.set_ip(func_op.entry_block)
        build_stmts(ctx, node.body)
        ctx.top_func = func_op

    @staticmethod
    def build_Compare(ctx, node):
        eq_flags = []
        cond_op = node.ops[0]
        if not isinstance(cond_op, ast.Eq):
            raise NotImplementedError("Only support '==' for now")
        exprs = []
        exprs.append(
            AffineExpr.get_dim(0) - AffineConstantExpr.get(node.comparators[0].value)
        )
        eq_flags.append(True)
        if_cond_set = IntegerSet.get(1, 0, exprs, eq_flags)
        attr = hcl_d.IntegerSetAttr.get(if_cond_set)
        return attr, ctx.buffers[node.left.id]

    @staticmethod
    def build_If(ctx, node):
        # Should build the condition on-the-fly
        cond, var = build_stmt(ctx, node.test)
        if_op = affine_d.AffineIfOp(
            cond, [var.result], ip=ctx.get_ip(), hasElse=len(node.orelse), results_=[]
        )
        ctx.set_ip(if_op.then_block)
        build_stmts(ctx, node.body)
        affine_d.AffineYieldOp([], ip=ctx.get_ip())
        ctx.pop_ip()
        if len(node.orelse) > 0:
            ctx.set_ip(if_op.else_block)
            build_stmts(ctx, node.orelse)
            affine_d.AffineYieldOp([], ip=ctx.get_ip())
            ctx.pop_ip()

    @staticmethod
    def build_Module(ctx, node):
        for stmt in node.body:
            build_stmt(ctx, stmt)
        return None

    @staticmethod
    def build_Call(ctx, node):
        if isinstance(node.func, ast.Name):
            if node.func.id == "float":
                if node.args[0].id in ctx.global_vars:
                    return MockConstant(float(ctx.global_vars[node.args[0].id]), ctx)
                else:
                    # TODO: Support other types
                    return arith_d.SIToFPOp(F32Type.get(), ctx.buffers[node.args[0].id].result, ip=ctx.get_ip())
            elif node.func.id == "int":
                return MockConstant(int(ctx.global_vars[node.args[0].id]), ctx)
        if node.func.value.id != "hcl":
            raise RuntimeError("Only support hcl functions for now")
        opcls = {
            "exp": math_d.ExpOp,
            "log": math_d.LogOp,
            "log2": math_d.Log2Op,
            "log10": math_d.Log10Op,
            "sqrt": math_d.SqrtOp,
            "sin": math_d.SinOp,
            "cos": math_d.CosOp,
            "tan": math_d.TanOp,
            "pow": math_d.PowFOp,
        }.get(node.func.attr)
        new_args = [stmt.result for stmt in build_stmts(ctx, node.args)]
        return opcls(*new_args, ip=ctx.get_ip())

    @staticmethod
    def build_Return(ctx, node):
        ip = ctx.pop_ip()
        if MemRefType(ctx.buffers[node.value.id].result.type).shape == [1]:  # scalar
            affine_map = AffineMap.get(
                dim_count=0, symbol_count=0, exprs=[AffineConstantExpr.get(0)]
            )
            affine_attr = AffineMapAttr.get(affine_map)
            load = affine_d.AffineLoadOp(
                ctx.buffers[node.value.id].result, [], affine_attr, ip=ip
            )
            func_d.ReturnOp([load.result], ip=ip)
        else:
            func_d.ReturnOp([ctx.buffers[node.value.id].result], ip=ip)
        return


build_stmt = ASTTransformer()


def build_stmts(ctx, stmts):
    results = []
    for stmt in stmts:
        results.append(build_stmt(ctx, stmt))
    return results
