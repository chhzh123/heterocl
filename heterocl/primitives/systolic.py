# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)

from ..ast import ast
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class SystolicPrimitive(Primitive):
    name = "systolic"
    is_stage_primitive = True

    @staticmethod
    def apply(sch, stage):
        """Wrap the current stage as a systolic array"""
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        systolic_op = ast.SystolicOp(stage.tensor, loc)
        schedule = sch._CurrentSchedule
        schedule.ast.top_func.body.append(systolic_op)
