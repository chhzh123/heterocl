# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import inspect
import textwrap
import ast
from dataclasses import dataclass

from hcl_mlir.ir import (
    Module,
    InsertionPoint,
    StringAttr,
    IntegerType,
    IntegerAttr,
    F32Type,
    MemRefType,
)
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.exceptions import (
    HCLValueError,
)

from .ir.builder import ASTTransformer, ASTContext
from .context import get_context, set_context, get_location
from .ir.transform import get_loop_band_names
from .build_module import _mlir_lower_pipeline, build_llvm
from .runtime import copy_build_files
from .module import HCLModule


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
    def wrapper(*args, **kwargs):
        with get_context(), get_location():
            fn(*args, **kwargs)
        _mlir_lower_pipeline(args[0].module)

    return wrapper


@dataclass
class Partition:
    Complete = 0
    Block = 1
    Cyclic = 2


class Schedule:
    def __init__(self, module, top_func, ip):
        self.module = module
        self.top_func = top_func
        self.ip = ip

    def __repr__(self):
        return str(self.module)

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

    @wrapped_apply
    def partition(self, target, partition_type=Partition.Complete, dim=0, factor=0):
        if partition_type > 2:
            raise HCLValueError("Invalid partition type")
        if dim < 0:
            raise HCLValueError("Invalid dimension")
        if factor < 0:
            raise HCLValueError("Invalid factor")
        if partition_type == Partition.Complete:
            partition_type = 0
        elif partition_type == Partition.Block:
            partition_type = 1
        elif partition_type == Partition.Cyclic:
            partition_type = 2
        else:
            raise HCLValueError("Not supported partition type")
        i32 = IntegerType.get_signless(32)
        ui32 = IntegerType.get_unsigned(32)
        partition_type = IntegerAttr.get(i32, partition_type)
        dim = IntegerAttr.get(ui32, dim)
        factor = IntegerAttr.get(ui32, factor)
        hcl_d.PartitionOp(
            target.result,
            partition_kind=partition_type,
            dim=dim,
            factor=factor,
            ip=self.ip,
        )

    @wrapped_apply
    def buffer_at(self, target, axis: str):
        band_name = get_loop_band_names(self.top_func)[0]
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdl = hcl_d.CreateLoopHandleOp(
            op_hdl.result, StringAttr.get(axis), ip=self.ip
        )
        memref_type = MemRefType.get((1,), F32Type.get())
        hcl_d.BufferAtOp(memref_type, target.result, loop_hdl.result, ip=self.ip)

    @wrapped_apply
    def pipeline(self, axis, initiation_interval=1):
        i32 = IntegerType.get_unsigned(32)
        ii = IntegerAttr.get(i32, initiation_interval)
        band_name = get_loop_band_names(self.top_func)[0]
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdl = hcl_d.CreateLoopHandleOp(
            op_hdl.result, StringAttr.get(axis), ip=self.ip
        )
        hcl_d.PipelineOp(loop_hdl.result, ii=ii, ip=self.ip)

    @wrapped_apply
    def reuse_at(self, target, axis):
        band_name = get_loop_band_names(self.top_func)[0]
        op_hdl = hcl_d.CreateOpHandleOp(band_name, ip=self.ip)
        loop_hdl = hcl_d.CreateLoopHandleOp(
            op_hdl.result, StringAttr.get(axis), ip=self.ip
        )
        memref_type = MemRefType.get((1,), F32Type.get())
        hcl_d.ReuseAtOp(memref_type, target.result, loop_hdl.result, ip=self.ip)

    def build(self, target=None):
        if target is None:
            target = "llvm"
            return build_llvm(self, top_func_name=self.top_func.name.value)
        elif target == "vhls":
            buf = io.StringIO()
            hcl_d.emit_vhls(self.module, buf)
            buf.seek(0)
            hls_code = buf.read()
            return hls_code
        elif str(target.tool.mode) == "debug":
            target.top = self.top_func.name.value
            copy_build_files(target)
            buf = io.StringIO()
            hcl_d.emit_vhls(self.module, buf)
            buf.seek(0)
            hls_code = buf.read()
            with open(f"{target.project}/kernel.cpp", "w", encoding="utf-8") as outfile:
                outfile.write(hls_code)
            with open(f"{target.project}/host.cpp", "w", encoding="utf-8") as outfile:
                outfile.write("")

            hcl_module = HCLModule(target.top, hls_code, target, host_src=None)
            return hcl_module
        else:
            NotImplementedError("Target {} is not supported".format(target))


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
    ASTTransformer()(ctx, tree)
    # Attach buffers to function
    for name, buffer in ctx.buffers.items():
        setattr(fn, name, buffer)
    return Schedule(
        module,
        ctx.top_func,
        InsertionPoint.at_block_terminator(ctx.top_func.entry_block),
    )
