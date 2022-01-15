import inspect
from collections import OrderedDict

import hcl_mlir
import hcl_mlir.affine as affine
from hcl_mlir import (ASTBuilder, GlobalInsertionPoint, get_context,
                      get_location)

from mlir.dialects import memref, std
from mlir.ir import *

from .. import config, types
from .schedule import Stage


def init(init_dtype=types.Int(32), raise_assert_exception=True):
    """Initialize a HeteroCL environment with configurations.
    """
    config.init_dtype = init_dtype
    config.raise_assert_exception = raise_assert_exception


def get_dtype_str(dtype=None):
    if not dtype is None and not isinstance(dtype, types.Type):
        raise RuntimeError("Type error")
    dtype = config.init_dtype if dtype is None else dtype
    str_type = types.dtype_to_str(dtype)
    return str_type


def placeholder(shape, name=None, dtype=None):
    """Construct a HeteroCL placeholder for inputs/outputs.
    """
    tensor = hcl_mlir.TensorOp(
        shape, memref.AllocOp, get_dtype_str(dtype), name=name)
    return tensor


def reduce_axis(lower, upper, name=None):
    """Create a reduction axis for reduction operations.
    """
    return hcl_mlir.ReduceVar(None, bound=(lower, upper), name=name)


def sum(data, axis=None, dtype=None, name=""):
    return hcl_mlir.SumOp(data, axis, get_dtype_str(dtype))


def compute(shape, fcompute, name=None, dtype=None, attrs=OrderedDict()):
    """Construct a new tensor based on the shape and the compute function.
    """
    # check API correctness
    if not isinstance(shape, tuple):
        raise RuntimeError("The shape of compute API must be a tuple")

    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    out_ndim = len(shape)

    argspec = inspect.getfullargspec(fcompute)
    if len(argspec.args) == 0 and argspec.varargs is None:
        arg_names = ["i%d" % i for i in range(out_ndim)]
    elif argspec.varargs is not None:
        # if there is a varargs, it takes the remaining dimensions of out_ndim
        arg_names = argspec.args + [
            f"i{i}" for i in range(out_ndim - len(argspec.args))
        ]
    else:
        arg_names = argspec.args
        # if there are fewer args than out dimensions, the remaining dimensions
        # are implicitly broadcast
        out_ndim = len(arg_names)
    assert argspec.varkw is None, "Variable keyword arguments not supported in fcompute"
    assert argspec.defaults is None, "Default arguments not supported in fcompute"
    assert (
        len(argspec.kwonlyargs) == 0
    ), "Keyword arguments are not supported in fcompute"

    with get_context() as ctx, get_location() as loc, Stage(name) as stage:
        func_ip = GlobalInsertionPoint.get()
        # create return tensor
        ret_tensor = placeholder(shape, dtype=dtype, name=name)
        ret_tensor.build()

        with func_ip:
            # create loop handles
            loop_handles = []
            for loop_name in arg_names:
                loop_handles.append(
                    hcl_mlir.CreateLoopHandleOp(
                        hcl_mlir.LoopHandleType.get(
                            ctx), StringAttr.get(loop_name)
                    )
                )

        # create for loops in the stage
        loops = []
        body_ip = func_ip
        for i, (ub, loop_name) in enumerate(zip(shape, arg_names)):
            loop = hcl_mlir.make_constant_for(
                0,
                ub,
                step=1,
                name=loop_name,
                stage=(name if i == 0 else ""),
                ip=body_ip,
            )
            if i != 0:  # manually add terminator!
                affine.AffineYieldOp([], ip=body_ip)
            loops.append(loop)
            body_ip = InsertionPoint(loop.body)

        # transform lambda function to MLIR
        GlobalInsertionPoint.save(body_ip)  # inner-most loop
        # get loop variables (BlockArgument)
        iter_var = [hcl_mlir.IterVar(loop.induction_variable)
                    for loop in loops]

        # calculate the lambda funtion,
        # at the same time build up MLIR nodes;
        # the Python builtin operators are overloaded in our custom class,
        # thus fcompute can be directly called and run
        result_expr = fcompute(*iter_var)
        builder = ASTBuilder()
        true_result = builder.visit(result_expr)
        result_expr.op = true_result

        # store the result back to tensor
        # we have to read the ssa value out first, then store back to tensor
        if isinstance(result_expr, hcl_mlir.SumOp):
            zero_idx = std.ConstantOp(
                IndexType.get(), IntegerAttr.get(IndexType.get(), 0), ip=GlobalInsertionPoint.get())
            value = memref.LoadOp(
                hcl_mlir.get_mlir_type(result_expr.dtype),
                result_expr.op.result,
                [zero_idx.result],
                loc=loc,
                ip=GlobalInsertionPoint.get()
            )
        else:
            value = result_expr.op
        ret_val = memref.StoreOp(
            value.result,
            ret_tensor.op.result,
            [loop.induction_variable for loop in loops],
            ip=GlobalInsertionPoint.get(),
        )

        # remember to add affine.yield after each for loop
        affine.AffineYieldOp([], ip=GlobalInsertionPoint.get())

        # set loop handles
        stage.set_output(ret_tensor)
        stage.op.set_axis(loop_handles)

        # recover insertion point
        GlobalInsertionPoint.restore()

        return ret_tensor