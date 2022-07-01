import inspect

import pytest
import tvm
from tvm.ir import structural_equal
from tvm.script.builder import ir as I
from tvm.script.builder import tir as T


def test_parse_elementwise():
    # pylint: disable=unused-argument,unused-variable,invalid-name
    @T.prim_func
    def elementwise(
        A: T.Buffer[(128, 128), "float32"],
        B: T.Buffer(shape=(128, 128, 128), dtype="float32"),  # type: ignore
    ) -> None:
        for i, j, *vvv, k in T.grid(128, 128, 128, 128, 128, 128, 128):
            with T.block("inner_block"):
                # vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                vi = T.axis.S(128, i + 1)
                vj = T.axis.S(128, j + 20)
                vk = T.axis.R(128, k - i)
                A[vi + 1, vj] = A[vi, vk] * B[vvv[0], vvv[1], vvv[2]] + 2
                B[vi, vj, vk] = A[vvv[0], vvv[-1]]

    # pylint: enable=unused-argument,unused-variable,invalid-name

    result = elementwise
    print(result.script())


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

    print(C.script())


def test_parse_atomic():
    @T.prim_func
    def f(A: T.int32, B: T.int64, C: T.handle) -> None:
        pass

    assert f.params[0].name == "A"
    assert f.params[0].dtype == "int32"
    assert f.params[1].name == "B"
    assert f.params[1].dtype == "int64"
    assert f.params[2].name == "C"
    assert f.params[2].dtype == "handle"


def test_parse_report_error():
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def elementwise() -> None:
            for (*vvv,) in T.grid(128, 128, 128, 128, 128, 128, 128):
                with T.block("inner_block"):
                    vj = T.axis.S(128, vvv[10] + 20)


def test_parse_concise_scope():
    # pylint: disable=unused-argument,unused-variable,invalid-name
    @T.prim_func
    def concise_scope(
        A: T.handle,
    ) -> None:
        A_local = T.allocate([64], "float32", "local")
        B_local = T.allocate([64], "float32", "local")
        C_local = T.allocate([64], "float32", "local")
        T.evaluate(1)
        T.evaluate(2)
        T.evaluate(3)

    @T.prim_func
    def normal_scope(
        A: T.handle,
    ) -> None:
        with T.allocate([64], "float32", "local") as A_local:
            with T.allocate([64], "float32", "local") as B_local:
                with T.allocate([64], "float32", "local") as C_local:
                    T.evaluate(1)
                    T.evaluate(2)
                    T.evaluate(3)

    assert structural_equal(normal_scope, concise_scope)


if __name__ == "__main__":
    test_parse_elementwise()
    test_parse_skip()
    test_parse_class()
    test_parse_atomic()
    test_parse_report_error()
    test_parse_concise_scope()
