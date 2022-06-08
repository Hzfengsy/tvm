from tvm.script.builder import tir as T
from tvm.script.parse import parse

elementwise = """
@T.prim_func
def elementwise(
    A: T.Buffer(shape=(128, 128, 128), dtype="float32"),
    B: T.Buffer(shape=(128, 128, 128), dtype="float32"),
) -> None:
    for i, j, *vvv, k in T.grid(128, 128, 128, 128, 128, 128, 128):
        with T.block("inner_block"):
            # vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            vi = T.axis.S(128, i + 1)
            vj = T.axis.S(128, j + 20)
            vk = T.axis.R(128, k - i)
"""


def main():
    result = parse(elementwise, extra_vars={"T": T})
    print(result.script())


if __name__ == "__main__":
    main()
