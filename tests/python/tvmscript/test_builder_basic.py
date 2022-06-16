# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
from tvm.script.builder import Builder, def_, def_many
from tvm.script.builder import tir as T
from tvm.tir import BufferLoad, BufferRegion
from tvm.ir import Range


def test_builder_root_block():
    print("test_builder_root_block")
    # impilict root block
    with Builder() as b0:
        with T.prim_func():
            T.func_name("main")
            T.func_attr({"key": "value"})
            with T.block(name="block"):
                pass
    print(b0.get().script())
    with Builder() as b1:
        with T.prim_func():
            T.func_name("main")
            T.func_attr({"key": "value"})
            A = def_("A", T.alloc_buffer((128,)))
            with T.block(name="block"):
                pass
    print(b1.get().script())
    with Builder() as b2:
        with T.prim_func():
            T.func_name("main")
            T.func_attr({"key": "value"})
            A = def_("A", T.alloc_buffer((128,)))
            with T.block(name="block0"):
                pass
            with T.block(name="block1"):
                pass
    print(b2.get().script())
    # expilict root block
    with Builder() as b0_r:
        with T.prim_func():
            T.func_name("main")
            T.func_attr({"key": "value"})
            with T.block(name="root"):
                with T.block(name="block"):
                    pass
    print(b0_r.get().script())
    with Builder() as b1_r:
        with T.prim_func():
            T.func_name("main")
            T.func_attr({"key": "value"})
            with T.block(name="root"):
                A = def_("A", T.alloc_buffer((128,)))
                with T.block(name="block"):
                    pass
    print(b1_r.get().script())
    with Builder() as b2_r:
        with T.prim_func():
            T.func_name("main")
            T.func_attr({"key": "value"})
            with T.block(name="root"):
                A = def_("A", T.alloc_buffer((128,)))
                with T.block(name="block0"):
                    pass
                with T.block(name="block1"):
                    pass
    print(b2_r.get().script())


def test_builder_axis():
    print("test_builder_axis")
    with Builder() as b:
        with T.prim_func():
            T.func_name("main")
            with T.grid(128, 128, 128, 128, 128) as (i, j, k, m, n):
                def_many(["i", "j", "k", "m", "n"], [i, j, k, m, n])
                with T.block(name="block"):
                    vi = def_("vi", T.axis.spatial(128, i))
                    vj = def_("vj", T.axis.spatial(128, j))
                    vk = def_("vk", T.axis.reduce(128, k))
                    vm = def_("vm", T.axis.scan(128, m))
                    vn = def_("vn", T.axis.opaque(128, n))
                    x, y, z = def_many(["x", "y", "z"], T.axis.remap("SSR", [i, j, k]))
    print(b.get().script())


def test_builder_prim_func():
    print("test_builder_prim_func")
    with Builder() as b:
        with T.prim_func():
            T.func_name("main")
            T.func_attr({"global_symbol": "main"})
            arg_a = T.arg("a", T.handle())
            arg_b = T.arg("b", T.handle())
            buffer_c = T.Buffer((128,), "float32")
            buffer_d = T.Buffer((128,), "float32")
            arg_c = T.arg("c", buffer_c)
            arg_d = T.arg("d", buffer_d)
            T.func_ret(tvm.ir.PrimType("int8"))
            A = def_("A", T.match_buffer(arg_a, (128, 128, 128), "int32"))
            B = def_("B", T.match_buffer(arg_b, (128, 128, 128), "int32"))
            T.preflattened_buffer(buffer_c, (128,), data=buffer_c.data)
            T.preflattened_buffer(buffer_d, (128,), data=buffer_d.data)
    print(b.get().script())


def test_builder_block():
    print("test_builder_block")
    with Builder() as b:
        with T.prim_func():
            arg_a = T.arg("a", T.handle())
            arg_b = T.arg("b", T.handle())
            A = def_("A", T.match_buffer(arg_a, (128, 128, 128), "int32"))
            B = def_("B", T.match_buffer(arg_b, (128, 128, 128), "int32"))
            with T.grid(128, 128, 128) as (i, j, k):
                def_many(["i", "j", "k"], [i, j, k])
                with T.block(name="block"):
                    T.block_attr({"axis": 1})
                    T.where(i > 1)
                    with T.init():
                        pass
                    vi, vj, vk = def_many(["vi", "vj", "vk"], T.axis.remap("SSR", [i, j, k]))
                    T.reads(
                        BufferRegion(
                            A,
                            [
                                Range(vi, vi + 1),
                                Range.from_min_extent(vj, 2),
                                Range(vk, vk + BufferLoad(B, [1, 2, BufferLoad(A, [3, 4, 5])])),
                            ],
                        )
                    )
                    T.writes([BufferLoad(A, [100, BufferLoad(A, [50, 51, 52]), 102])])
                    E = def_("E", T.alloc_buffer((128, 128)))
                    F = def_("F", T.alloc_buffer((128, 128)))
    print(b.get().script())


def test_builder_for():
    print("test_builder_for")
    with Builder() as b:
        with T.prim_func():
            with T.grid(128, 128, 128) as (i, j, k):
                def_many(["i", "j", "k"], [i, j, k])
            with T.serial(0, 128) as w:
                w = def_("w", w)
            with T.parallel(0, 128) as x:
                x = def_("x", x)
            with T.vectorized(0, 128) as y:
                y = def_("y", y)
            with T.unroll(0, 128) as z:
                z = def_("z", z)
            with T.thread_binding(0, 32, thread="blockIdx.x") as bx:
                bx = def_("bx", bx)
                with T.thread_binding(0, 2, thread="vthread.y") as vy:
                    vy = def_("vy", vy)
                    with T.thread_binding(0, 8, thread="threadIdx.z") as tz:
                        tz = def_("tz", tz)
    print(b.get().script())


if __name__ == "__main__":
    test_builder_root_block()
    test_builder_axis()
    test_builder_prim_func()
    test_builder_block()
    test_builder_for()
