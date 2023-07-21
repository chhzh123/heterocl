import sys
import heterocl as hcl
from heterocl.ir.types import int32, float32


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
        for i in range(10):
            if i == 0:
                r = 1
            elif i == 1:
                r = 2
                if i == 2:
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


def test_conv2D_lb():
    def conv2D(A: int32[10, 10]) -> int32[8, 8]:
        B: int32[8, 8] = 0
        for i, j in hcl.grid(8, 8):
            v: int32 = 0
            for rx, ry in hcl.reduction(3, 3):
                v += A[i + rx, j + ry]
            B[i, j] = v
        return B

    s = hcl.customize(conv2D)
    s.reuse_at(conv2D.A, axis="i")
    print(s.module)
    mod = s.build()

    # testing
    import numpy as np

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


if __name__ == "__main__":
    # test_gemm_grid_for()
    # test_gemm_range_for()
    # test_gemm_reduction_var()
    # test_gemm_float()
    # test_nested_if()
    # test_interleaving_acc()
    # test_buffer_at()
    # test_platform()
    test_conv2D_lb()

sys.exit()
