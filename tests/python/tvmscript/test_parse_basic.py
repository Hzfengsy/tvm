from tvm.script.builder import tir as T


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


def main():
    result = elementwise
    print(result.script())


if __name__ == "__main__":
    main()
