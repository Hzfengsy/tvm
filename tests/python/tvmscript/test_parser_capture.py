from tvm.script.builder import ir as I
from tvm.script.builder import tir as T


def test_capture_func():
    from tvm.script.builder.tir import axis as ax
    from tvm.script.builder.tir import block, match_buffer

    @T.prim_func
    def scalar_func(a: T.handle, b: T.handle, c: T.Buffer((128,))):
        A = match_buffer(a, (128, 128))
        B = match_buffer(b, (128, 128))
        with block():
            for i, j in T.grid(128, 128):
                with block("inner_block"):
                    vi, vj = ax.remap("SR", [i, j])
                    A[i, j] = B[i - 1, j + 1] + A[i - 1, j - 1]

    print(scalar_func.script())


def test_capture_class():
    from tvm.script.builder.tir import axis as ax
    from tvm.script.builder.tir import block, match_buffer

    @I.ir_module
    class Module:
        @T.prim_func
        def scalar_func(a: T.handle, b: T.handle, c: T.Buffer((128,))):
            A = match_buffer(a, (128, 128))
            B = match_buffer(b, (128, 128))
            with block():
                for i, j in T.grid(128, 128):
                    with block("inner_block"):
                        vi, vj = ax.remap("SR", [i, j])
                        A[i, j] = B[i - 1, j + 1] + A[i - 1, j - 1]

    print(Module.script())


if __name__ == "__main__":
    test_capture_func()
    test_capture_class()
