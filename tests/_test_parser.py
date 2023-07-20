import sys
import heterocl as hcl
from heterocl.ir.types import int32


def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
    C: int32[32, 32] = 0
    for i, j, k in hcl.grid(32, 32, 32):
        C[i, j] += A[i, k] * B[k, j]
    return C


s = hcl.customize(gemm)
s.split("i", 8)
s.split("j", 8)
s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
print(s.module)
# transformations are applied immediately

sys.exit()

Module(
    body=[
        FunctionDef(
            name="gemm",
            args=arguments(
                posonlyargs=[],
                args=[
                    arg(
                        arg="A",
                        annotation=Subscript(
                            value=Name(id="int32", ctx=Load()),
                            slice=Index(
                                value=Tuple(
                                    elts=[
                                        Constant(value=32, kind=None),
                                        Constant(value=32, kind=None),
                                    ],
                                    ctx=Load(),
                                )
                            ),
                            ctx=Load(),
                        ),
                        type_comment=None,
                    ),
                    arg(
                        arg="B",
                        annotation=Subscript(
                            value=Name(id="int32", ctx=Load()),
                            slice=Index(
                                value=Tuple(
                                    elts=[
                                        Constant(value=32, kind=None),
                                        Constant(value=32, kind=None),
                                    ],
                                    ctx=Load(),
                                )
                            ),
                            ctx=Load(),
                        ),
                        type_comment=None,
                    ),
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[
                AnnAssign(
                    target=Name(id="C", ctx=Store()),
                    annotation=Subscript(
                        value=Name(id="int32", ctx=Load()),
                        slice=Index(
                            value=Tuple(
                                elts=[
                                    Constant(value=32, kind=None),
                                    Constant(value=32, kind=None),
                                ],
                                ctx=Load(),
                            )
                        ),
                        ctx=Load(),
                    ),
                    value=Constant(value=0, kind=None),
                    simple=1,
                ),
                For(
                    target=Tuple(
                        elts=[
                            Name(id="i", ctx=Store()),
                            Name(id="j", ctx=Store()),
                            Name(id="k", ctx=Store()),
                        ],
                        ctx=Store(),
                    ),
                    iter=Call(
                        func=Attribute(
                            value=Name(id="hcl", ctx=Load()), attr="grid", ctx=Load()
                        ),
                        args=[
                            Constant(value=32, kind=None),
                            Constant(value=32, kind=None),
                            Constant(value=32, kind=None),
                        ],
                        keywords=[],
                    ),
                    body=[
                        AugAssign(
                            target=Subscript(
                                value=Name(id="C", ctx=Load()),
                                slice=Index(
                                    value=Tuple(
                                        elts=[
                                            Name(id="i", ctx=Load()),
                                            Name(id="j", ctx=Load()),
                                        ],
                                        ctx=Load(),
                                    )
                                ),
                                ctx=Store(),
                            ),
                            op=Add(),
                            value=BinOp(
                                left=Subscript(
                                    value=Name(id="A", ctx=Load()),
                                    slice=Index(
                                        value=Tuple(
                                            elts=[
                                                Name(id="i", ctx=Load()),
                                                Name(id="k", ctx=Load()),
                                            ],
                                            ctx=Load(),
                                        )
                                    ),
                                    ctx=Load(),
                                ),
                                op=Mult(),
                                right=Subscript(
                                    value=Name(id="B", ctx=Load()),
                                    slice=Index(
                                        value=Tuple(
                                            elts=[
                                                Name(id="k", ctx=Load()),
                                                Name(id="j", ctx=Load()),
                                            ],
                                            ctx=Load(),
                                        )
                                    ),
                                    ctx=Load(),
                                ),
                            ),
                        )
                    ],
                    orelse=[],
                    type_comment=None,
                ),
                Return(value=Name(id="C", ctx=Load())),
            ],
            decorator_list=[],
            returns=Subscript(
                value=Name(id="int32", ctx=Load()),
                slice=Index(
                    value=Tuple(
                        elts=[
                            Constant(value=32, kind=None),
                            Constant(value=32, kind=None),
                        ],
                        ctx=Load(),
                    )
                ),
                ctx=Load(),
            ),
            type_comment=None,
        )
    ],
    type_ignores=[],
)
