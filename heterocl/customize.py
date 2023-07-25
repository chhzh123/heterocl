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
    UnitAttr,
    StringAttr,
    IntegerType,
    IntegerAttr,
    F32Type,
    MemRefType,
)
from hcl_mlir.dialects import (
    hcl as hcl_d,
    memref as memref_d,
    func as func_d,
)
from hcl_mlir.exceptions import (
    HCLValueError,
)
import os
import ctypes
from hcl_mlir.passmanager import PassManager
from hcl_mlir.execution_engine import ExecutionEngine
from hcl_mlir.runtime import (
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    ranked_memref_to_numpy,
)
import numpy as np

from .ir.builder import ASTTransformer, ASTContext
from .context import get_context, set_context, get_location
from .ir.transform import get_loop_band_names
from .build_module import _mlir_lower_pipeline
from .runtime import copy_build_files
from .module import HCLModule


def np_type_to_str(dtype):
    if dtype == np.float32:
        return "f32"
    elif dtype == np.float64:
        return "f64"
    elif dtype == np.int32:
        return "i32"
    elif dtype == np.int64:
        return "i64"
    else:
        raise RuntimeError("Unsupported dtype")


def getsourcefile(obj):
    ret = inspect.getsourcefile(obj)
    if ret is None:
        ret = inspect.getfile(obj)
    return ret


def getsourcelines(obj):
    return inspect.getsourcelines(obj)


def _get_global_vars(_func):
    # Discussions: https://github.com/taichi-dev/taichi/issues/282
    # global_vars = _func.__globals__.copy()
    global_vars = {}

    freevar_names = _func.__code__.co_freevars
    closure = _func.__closure__
    # Get back to the outer-most scope (user-defined function)
    for name, var in inspect.stack()[2][0].f_locals.items():
        if isinstance(var, (int, float)):
            global_vars[name] = var
    if closure:
        freevar_values = list(map(lambda x: x.cell_contents, closure))
        for name, value in zip(freevar_names, freevar_values):
            global_vars[name] = value

    return global_vars


def wrapped_apply(fn):
    def wrapper(*args, **kwargs):
        with get_context(), get_location():
            res = fn(*args, **kwargs)
        _mlir_lower_pipeline(args[0].module)
        args[0].primitive_sequences.append((fn.__name__, args[1:], kwargs))
        return res

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
        self.primitive_sequences = []

    def __repr__(self):
        # Used for Module.parse
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

        def find_reuse_buffers(res):
            for op in self.top_func.entry_block.operations:
                if (
                    isinstance(op, memref_d.AllocOp)
                    and "name" in op.attributes
                    and StringAttr(band_name).value + "_reuse"
                    in StringAttr(op.attributes["name"]).value
                ):
                    res.append(op)

        prev_reuse_buffers = []
        find_reuse_buffers(prev_reuse_buffers)
        hcl_d.ReuseAtOp(memref_type, target.result, loop_hdl.result, ip=self.ip)
        _mlir_lower_pipeline(self.module)
        new_reuse_buffers = []
        find_reuse_buffers(new_reuse_buffers)
        new_reuse_buffers = [
            buf for buf in new_reuse_buffers if buf not in prev_reuse_buffers
        ]
        if len(new_reuse_buffers) != 1:
            raise RuntimeError("Reuse buffer not found")
        return new_reuse_buffers[0]

    @wrapped_apply
    def compose(self, *schs):
        for sch in schs:
            if not isinstance(sch, Schedule):
                raise TypeError("The first argument must be a Schedule object")
            func_to_replace = sch.top_func
            for func in self.module.body.operations:
                if func.name.value == func_to_replace.name.value:
                    func.operation.erase()
                    break
            new_mod = Module.parse(str(sch.top_func))
            for func in new_mod.body.operations:
                if func.name.value == func_to_replace.name.value:
                    func.move_before(self.module.body.operations[0])
            # Need to update CallOp arguments since some of them may be partitioned
            # We simply replay all the primitives and find the `partition`
            for primitive in sch.primitive_sequences:
                if primitive[0] == "partition":
                    args, kwargs = primitive[1:]
                    if len(args) != 0:
                        target = args[0]
                    else:
                        target = kwargs["target"]
                    arg_idx = -1
                    for idx, arg in enumerate(sch.top_func.arguments):
                        if arg == target.result:
                            arg_idx = idx
                            break
                    else:
                        raise RuntimeError("Target not found")
                    for op in self.top_func.entry_block.operations:
                        if (
                            isinstance(op, func_d.CallOp)
                            and str(op.attributes["callee"])[1:]
                            == func_to_replace.name.value
                        ):
                            from .ir.builder import MockArg

                            self.partition(
                                MockArg(op.operands[arg_idx]), *args[1:], **kwargs
                            )
                            break

    def build(self, target=None):
        if target is None or target == "llvm":
            target = "llvm"
            _mlir_lower_pipeline(self.module, lower_linalg=True)
            mod = LLVMModule(self.module, top_func_name=self.top_func.name.value)
            return mod
        elif target == "vhls":
            # FIXME: Handle linalg.fill
            _mlir_lower_pipeline(self.module, lower_linalg=True)
            buf = io.StringIO()
            hcl_d.emit_vhls(self.module, buf)
            buf.seek(0)
            hls_code = buf.read()
            return hls_code
        elif str(target.tool.mode) == "debug":
            _mlir_lower_pipeline(self.module, lower_linalg=True)
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


class LLVMModule:
    def __init__(self, mod, top_func_name):
        # Copy the module to avoid modifying the original one
        with get_context() as ctx, get_location():
            self.module = Module.parse(str(mod), ctx)
            # find top func op
            func = None
            for op in self.module.body.operations:
                if isinstance(op, func_d.FuncOp) and op.name.value == top_func_name:
                    func = op
                    break
            if func is None:
                raise RuntimeError(
                    "No top-level function found in the built MLIR module"
                )
            func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            func.attributes["top"] = UnitAttr.get()
            self.top_func_type = func.type
            self.top_func_name = top_func_name
            # Remove .partition() annotation
            hcl_d.remove_stride_map(self.module)
            # Run through lowering passes
            pm = PassManager.parse(
                "func.func(convert-linalg-to-affine-loops),lower-affine,convert-scf-to-cf,convert-arith-to-llvm,convert-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts"
            )
            pm.run(self.module)
            # Add shared library
            if os.getenv("LLVM_BUILD_DIR") is not None:
                shared_libs = [
                    os.path.join(
                        os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_runner_utils.so"
                    ),
                    os.path.join(
                        os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_c_runner_utils.so"
                    ),
                ]
            else:
                shared_libs = None
            self.execution_engine = ExecutionEngine(
                self.module, opt_level=3, shared_libs=shared_libs
            )

    def __call__(self, *args):
        input_types = self.top_func_type.inputs
        new_args = []
        for arg, in_type in zip(args, input_types):
            if not isinstance(arg, np.ndarray):
                if isinstance(arg, int):
                    if str(in_type) != "i32":
                        raise RuntimeError(
                            "Input type mismatch, expected i32, but got {}".format(
                                str(in_type)
                            )
                        )
                    c_int_p = ctypes.c_int * 1
                    new_args.append(c_int_p(arg))
                elif isinstance(arg, float):
                    if str(in_type) != "f32":
                        raise RuntimeError(
                            "Input type mismatch, expected f32, but got {}".format(
                                str(in_type)
                            )
                        )
                    c_float_p = ctypes.c_float * 1
                    new_args.append(c_float_p(arg))
                else:
                    raise RuntimeError("Unsupported input type")
            else:
                np_type = np_type_to_str(arg.dtype)
                target_type = str(MemRefType(in_type).element_type)
                if np_type != target_type:
                    raise RuntimeError(
                        "Input type mismatch: {} vs {}".format(np_type, target_type)
                    )
                new_args.append(
                    ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg)))
                )
        # TODO: only support one return value for now
        result_types = self.top_func_type.results
        if len(result_types) != 1:
            raise RuntimeError("Only support one return value for now")
        if MemRefType.isinstance(result_types[0]):
            result_type = MemRefType(result_types[0])
            shape = result_type.shape
            result_type = result_type.element_type
            if str(result_type) == "f32":
                dtype = ctypes.c_float
            elif str(result_type) == "f64":
                dtype = ctypes.c_double
            elif str(result_type) == "i32":
                dtype = ctypes.c_int32
            elif str(result_type) == "i64":
                dtype = ctypes.c_int64
            else:
                raise RuntimeError("Unsupported return type")
            return_desc = make_nd_memref_descriptor(len(shape), dtype)()
            return_tensor = ctypes.pointer(ctypes.pointer(return_desc))
        elif IntegerType.isinstance(result_types[0]):
            result_type = IntegerType(result_types[0])
            if str(result_type) == "i32":
                dtype = ctypes.c_int32
            elif str(result_type) == "i64":
                dtype = ctypes.c_int64
            else:
                raise RuntimeError("Unsupported return type")
            dtype_p = dtype * 1
            return_tensor = dtype_p(-1)
        elif F32Type.isinstance(result_types[0]):
            result_type = F32Type(result_types[0])
            dtype_p = ctypes.c_float * 1
            return_tensor = dtype_p(-1.0)
        else:
            raise RuntimeError("Unsupported return type")
        if MemRefType.isinstance(result_types[0]):
            self.execution_engine.invoke(self.top_func_name, return_tensor, *new_args)
            ret = ranked_memref_to_numpy(return_tensor[0])
        else:
            self.execution_engine.invoke(self.top_func_name, *new_args, return_tensor)
            ret = return_tensor[0]
        return ret


def customize(fn, verbose=False):
    # Get Python AST
    file = getsourcefile(fn)
    src, start_lineno = getsourcelines(fn)
    src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
    src = textwrap.dedent("\n".join(src))
    if verbose:
        print(src)
    tree = ast.parse(src)
    if verbose:
        try:
            import astpretty

            astpretty.pprint(tree, indent=2, show_offsets=False)
        except:
            print(ast.dump(tree))
    # Create MLIR module
    set_context()
    with get_context() as mlir_ctx, get_location():
        hcl_d.register_dialect(mlir_ctx)
        module = Module.create()
    # Start building IR
    global_vars = _get_global_vars(fn)
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
