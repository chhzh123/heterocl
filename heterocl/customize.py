# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import textwrap
import ast

from hcl_mlir.ir import Module, InsertionPoint, StringAttr, IntegerType, IntegerAttr
from hcl_mlir.dialects import hcl as hcl_d

from .parser.builder import ASTTransformer, ASTContext
from .context import get_context, set_context, get_location
from .ir.transform import get_loop_band_names
from .build_module import _mlir_lower_pipeline


def getsourcefile(obj):
    ret = inspect.getsourcefile(obj)
    if ret is None:
        ret = inspect.getfile(obj)
    return ret


def getsourcelines(obj):
    return inspect.getsourcelines(obj)


def _get_global_vars(_func):
    # Discussions: https://github.com/taichi-dev/taichi/issues/282
    global_vars = _func.__globals__.copy()

    freevar_names = _func.__code__.co_freevars
    closure = _func.__closure__
    if closure:
        freevar_values = list(map(lambda x: x.cell_contents, closure))
        for name, value in zip(freevar_names, freevar_values):
            global_vars[name] = value

    return global_vars


def wrapped_apply(fn):
    def wrapper(*args):
        with get_context(), get_location():
            fn(*args)
        _mlir_lower_pipeline(args[0].module)

    return wrapper


class Schedule:
    def __init__(self, module, top_func, ip):
        self.module = module
        self.top_func = top_func
        self.ip = ip

    @wrapped_apply
    def split(self, name, factor):
        band_name = get_loop_band_names(self.top_func)[0]
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdl = hcl_d.CreateLoopHandleOp(
            op_hdl.result, StringAttr.get(name), ip=self.ip
        )
        i32 = IntegerType.get_unsigned(32)
        factor = IntegerAttr.get(i32, factor)
        hcl_d.SplitOp(loop_hdl.result, factor, ip=self.ip)

    @wrapped_apply
    def reorder(self, *args):
        band_name = get_loop_band_names(self.top_func)[0]
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdls = []
        for name in args:
            loop_hdls.append(
                hcl_d.CreateLoopHandleOp(
                    op_hdl.result, StringAttr.get(name), ip=self.ip
                )
            )
        arg_results = [arg.result for arg in loop_hdls]
        hcl_d.ReorderOp(arg_results, ip=self.ip)


def customize(fn):
    # Get Python AST
    file = getsourcefile(fn)
    src, start_lineno = getsourcelines(fn)
    src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
    src = textwrap.dedent("\n".join(src))
    print(src)
    tree = ast.parse(src)
    print(ast.dump(tree))
    # Create MLIR module
    set_context()
    with get_context() as mlir_ctx, get_location():
        hcl_d.register_dialect(mlir_ctx)
        module = Module.create()
    # Start building IR
    global_vars = _get_global_vars(fn)
    print(global_vars)
    ctx = ASTContext(global_vars=global_vars)
    ctx.set_ip(module.body)
    ret = ASTTransformer()(ctx, tree)
    return Schedule(
        module,
        ctx.top_func,
        InsertionPoint.at_block_terminator(ctx.top_func.entry_block),
    )
