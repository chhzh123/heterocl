# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)
from hcl_mlir.ir import (
    Location,
    InsertionPoint,
    IntegerType,
    IntegerAttr,
)
from hcl_mlir.dialects import hcl as hcl_d
from ..ast import ast
from ..ast.ir_builder import IRBuilder
from ..utils import get_src_loc
from .base import Primitive, register_primitive
from ..context import get_context, get_location


@register_primitive()
class PipelinePrimitive(Primitive):
    name = "pipeline"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, var, initiation_interval=1):
        """Pipeline the iteration."""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".pipeline() must be called before lowering")
        if isinstance(var, int):
            var = stage.tensor.axis[var]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        pipeline_op = ast.PipelineOp(var, initiation_interval, loc)
        sch.ast.top_func.body.append(pipeline_op)
        op = pipeline_op
        with get_context(), get_location():
            loc = Location.file(op.loc.filename, op.loc.lineno, 0)
            ir_builder = IRBuilder(sch._ast)
            ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
            ir_builder.build_visitor(op.target, ip)
            i32 = IntegerType.get_unsigned(32)
            ii = IntegerAttr.get(i32, op.ii)
            pipeline_op = hcl_d.PipelineOp(op.target.result, ii=ii, ip=ip, loc=loc)
            op.ir_op = pipeline_op
