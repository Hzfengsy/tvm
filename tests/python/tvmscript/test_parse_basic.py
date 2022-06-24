import inspect

from tvm.script.builder import ir as I
from tvm.script.builder import tir as T


def test_parse_elementwise():
    # pylint: disable=unused-argument,unused-variable,invalid-name
    @T.prim_func
    def elementwise(
        A: T.Buffer(shape=(128, 128, 128), dtype="float32"),  # type: ignore
        B: T.Buffer(shape=(128, 128, 128), dtype="float32"),  # type: ignore
    ) -> None:
        for i, j, *vvv, k in T.grid(128, 128, 128, 128, 128, 128, 128):
            with T.block("inner_block"):
                # vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                vi = T.axis.S(128, i + 1)
                vj = T.axis.S(128, j + 20)
                vk = T.axis.R(128, k - i)

    # pylint: enable=unused-argument,unused-variable,invalid-name

    result = elementwise
    # print(result.script())


def test_parse_skip():
    class Skip:
        @T.prim_func
        def f():  # type: ignore
            ...

    assert inspect.isfunction(Skip.f)


def test_parse_class():
    # pylint: disable=unused-argument,unused-variable,invalid-name
    @I.ir_module
    class C:
        @T.prim_func
        def elementwise(
            A: T.Buffer(shape=(128, 128, 128), dtype="float32"),  # type: ignore
            B: T.Buffer(shape=(128, 128, 128), dtype="float32"),  # type: ignore
        ) -> None:
            for i, j, *vvv, k in T.grid(128, 128, 128, 128, 128, 128, 128):
                with T.block("inner_block"):
                    # vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    vi = T.axis.S(128, i + 1)
                    vj = T.axis.S(128, j + 20)
                    vk = T.axis.R(128, k - i)

    # pylint: enable=unused-argument,unused-variable,invalid-name

    # print(C.script())


if __name__ == "__main__":
    test_parse_elementwise()
    test_parse_skip()
    test_parse_class()
