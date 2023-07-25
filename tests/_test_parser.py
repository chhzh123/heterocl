# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import numpy as np
import heterocl as hcl
from heterocl.ir.types import int1, int32, float32


def test_gemm_grid_for():
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in hcl.grid(32, 32, 32):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = hcl.customize(gemm)
    # transformations are applied immediately
    s.split("i", 8)
    s.split("j", 8)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    print(s.module)


def test_gemm_range_for():
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    C[i, j] += A[i, k] * B[k, j]
        return C

    s = hcl.customize(gemm)
    # transformations are applied immediately
    s.split("i", 8)
    s.split("j", 8)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    print(s.module)


def test_gemm_float():
    def gemm(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
        C: float32[32, 32] = 0
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    C[i, j] += A[i, k] * B[k, j]
        return C

    s = hcl.customize(gemm)
    # transformations are applied immediately
    s.split("i", 8)
    s.split("j", 8)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    print(s.module)


def test_gemm_reduction_var():
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j in hcl.grid(32, 32):
            v: int32 = 0
            for k in range(32):
                v += A[i, k] * B[k, j]
            C[i, j] = v
        return C

    s = hcl.customize(gemm)
    # transformations are applied immediately
    s.split("i", 8)
    s.split("j", 8)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    print(s.module)


def test_nested_if():
    def kernel(a: int32, b: int32) -> int32:
        r: int32 = 0
        if a == 0:
            r = 1
        elif a == 1:
            r = 2
            if b == 2:
                r = 3
        else:
            r = 4
        return r

    s = hcl.customize(kernel)
    print(s.module)


def test_interleaving_acc():
    # https://github.com/cornell-zhang/hcl-dialect/blob/v0.1/test/Transforms/memory/buffer_gemm.mlir#L86
    M = 1024
    N = 1024
    K = 1024

    def gemm(A: float32[M, K], B: float32[K, N]) -> float32[M, N]:
        C: float32[M, N] = 0
        for i, j in hcl.grid(M, N):
            for k in hcl.reduction(K):
                C[i, j] += A[i, k] * B[k, j]
        return C

    s = hcl.customize(gemm)
    s.reorder("k", "j")
    s.buffer_at(gemm.C, axis="i")
    s.pipeline("j")
    print(s.module)
    print(s.build(target="vhls"))
    return s


def test_platform():
    s = test_interleaving_acc()
    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="debug", project="gemm-inter-acc.prj")
    mod = s.build(target=target)
    mod()


def test_buffer_at():
    M, N = 1024, 1024

    def gemm(A: float32[M, N]) -> float32[M, N]:
        B: float32[M, N] = 0
        for i, j in hcl.grid(M, N):
            B[i, j] = A[i, j] + 1.0
        return B

    s = hcl.customize(gemm)
    s.buffer_at(gemm.B, axis="i")
    print(s.module)


def test_conv2D():
    def conv2D(A: int32[10, 10]) -> int32[8, 8]:
        B: int32[8, 8] = 0
        for i, j in hcl.grid(8, 8):
            v: int32 = 0
            for rx, ry in hcl.reduction(3, 3):
                v += A[i + rx, j + ry]
            B[i, j] = v
        return B

    s = hcl.customize(conv2D)
    s.split("j", 4)
    s.reorder("j.outer", "i", "j.inner")
    LB = s.reuse_at(conv2D.A, axis="i")
    WB = s.reuse_at(LB, axis="j.inner")
    s.partition(LB, dim=2)
    s.partition(WB)
    s.pipeline("i")
    print(s.module)
    mod = s.build()

    # testing
    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((8, 8), dtype="int")
    np_C = np.zeros((8, 8), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            for r in range(0, 3):
                for c in range(0, 3):
                    np_C[y][x] += np_A[y + r][x + c]

    mod(np_A, np_B)

    assert np.array_equal(np_B, np_C)


def test_bconv2D_nchw():
    bs = 4
    ic, oc = 6, 16
    ih, iw = 8, 8
    kh, kw = 3, 3
    oh, ow = ih - kh + 1, iw - kw + 1

    def bconv(
        A: int1[bs, ic, ih, iw], F: int1[oc, ic, kh, kw]
    ) -> int32[bs, oc, oh, ow]:
        B: int32[bs, oc, oh, ow] = 0
        for n, c, h, w in hcl.grid(bs, oc, oh, ow):
            for rc, rh, rw in hcl.reduction(ic, kh, kw):
                B[n, c, h, w] += A[n, rc, h + rh, w + rw] ^ F[c, rc, rh, rw]
        return B

    s = hcl.customize(bconv)
    print(s.module)


def test_nested_functions():
    M, K, N = 32, 32, 32

    def matrix_add(A: int32[M, N]) -> int32[M, N]:
        B: int32[M, N] = 0
        for i, j in hcl.grid(M, N):
            B[i, j] = A[i, j] + 1
        return B

    def gemm(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        C: int32[M, N] = 0
        for i, j, k in hcl.grid(M, N, K):
            C[i, j] += A[i, k] * B[k, j]
        return C

    def top(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        C = gemm(A, B)
        D = matrix_add(C)
        return D

    # Separate compilation (just for testing)
    s_gemm = hcl.customize(gemm)
    mod_gemm = s_gemm.build()

    # Top-level
    s = hcl.customize(top)
    print(s.module)
    mod = s.build()

    # Testing
    np_A = np.random.randint(0, 10, size=(M, K)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(K, N)).astype(np.int32)
    np_D = np.matmul(np_A, np_B)
    np_C = mod_gemm(np_A, np_B)
    assert np.array_equal(np_C, np_D)

    np_A = np.random.randint(0, 10, size=(M, K)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(K, N)).astype(np.int32)
    np_D = np_A @ np_B + 1
    np_C = mod(np_A, np_B)
    assert np.array_equal(np_C, np_D)


def test_nested_functions_2():
    M, K, N = 32, 32, 32

    def gemm(A: int32[M, K], B: int32[K, N], C: int32[M, N]) -> None:
        for i, j in hcl.grid(M, N):
            for k in hcl.reduction(K):
                C[i, j] += A[i, k] * B[k, j]

    def top(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        C: int32[M, N] = 0
        gemm(A, B, C)
        return C

    s1 = hcl.customize(gemm)
    s1.reorder("k", "j")
    s1.partition(gemm.C, dim=2)
    s1.buffer_at(gemm.C, axis="i")
    s1.pipeline("j")
    # Top-level
    s = hcl.customize(top, verbose=True)
    s.compose(s1)
    print(s.module)
    mod = s.build()

    # Testing
    np_A = np.random.randint(0, 100, size=(M, K)).astype(np.int32)
    np_B = np.random.randint(0, 100, size=(K, N)).astype(np.int32)
    np_C = mod(np_A, np_B)
    np_D = np.matmul(np_A, np_B)
    assert np.array_equal(np_C, np_D)
    print("Success!")


def test_nested_functions_3():
    M = 1024
    N = 1024
    K = 1024

    def gemm(inp: float32[M, K], W: float32[K, N], B: float32[N]) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        # This code can be Executed correctly
        # for i in range(M):
        #     for j in range(N):
        #         v: float32 = 0.0
        #         for k in hcl.reduction(K):
        #             v += inp[i, k] * W[k, j]
        #         outp[i, j] = v + B[j]
        # return outp
        # if I write in this way, it would report "RuntimeError: Failure while executing pass pipeline":
        # for i in range(M):
        #     for j in range(N):
        #         for k in hcl.reduction(K):
        #             outp[i, j] += inp[i, k] * W[k, j]
        #         outp[i, j] += B[j]
        # return outp

        # In addition, if there are other for construction, it would also report the same issue, such as:
        for i in range(M):
            for j in range(N):
                outp[i, j] = B[j]
        for i in range(M):
            for j in range(N):
                for k in hcl.reduction(K):
                    outp[i, j] += inp[i, k] * W[k, j]
        return outp

    s = hcl.customize(gemm)
    print(s.module)
    f = s.build(target="vhls")
    print(f)
    return f


def test_rhs_binaryop():
    def kernel() -> int32[11]:
        v: int32 = 5
        res: int32[11] = 0
        res[0] = 1 + v
        res[1] = 1 - v
        res[2] = v * 3
        res[3] = 52 / v
        res[4] = 6 // v
        res[5] = 6 % v
        res[6] = 1 << v
        res[7] = 64 >> v
        res[8] = 1 & v
        res[9] = 1 | v
        res[10] = res[9]
        return res

    s = hcl.customize(kernel, verbose=True)
    print(s.module)


def test_fcompute_function_wrapper():
    def kernel(A: int32[10]) -> int32[10]:
        def foo(x: int32) -> int32:
            y: int32 = 0
            y = x + 1
            return y

        B: int32[10] = 0
        for i in range(10):
            B[i] = foo(A[i])
        return B

    s = hcl.customize(kernel)
    print(s.module)


def test_fcompute_function_wrapper():
    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0

        def foo(x: int32) -> int32:
            return x + 1

        for i in range(10):
            B[i] = foo(A[i])
        return B

    s = hcl.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.random.randint(0, 10, size=(10,)).astype(np.int32)
    np_C = np_A + 1
    np_B = mod(np_A)
    assert np.array_equal(np_B, np_C)


if __name__ == "__main__":
    # test_gemm_grid_for()
    # test_gemm_range_for()
    # test_gemm_reduction_var()
    # test_gemm_float()
    # test_nested_if()
    # test_buffer_at()
    # test_conv2D()
    # test_interleaving_acc()
    # test_nested_functions()
    test_nested_functions_2()
    # test_nested_functions_3()
    # test_rhs_binaryop()
    # test_fcompute_function_wrapper()

sys.exit()
