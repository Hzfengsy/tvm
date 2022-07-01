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
                    T.reads(A[vi, vj, vk : vk + B[1, 2, A[3, 4, 5]]])
                    T.writes(A[100, A[50, 51, 52], 102])
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


def test_builder_stmt():
    print("test_builder_stmt")
    with Builder() as b:
        with T.prim_func():
            thread_x = def_("thread_x", T.env_thread("threadIdx.x"))
            thread_y = def_("thread_y", T.env_thread("threadIdx.y"))
            buffer_x = def_("buffer_x", T.Buffer([128, 128]))
            buffer_y = def_("buffer_y", T.Buffer([128, 128]))
            var_x = def_("var_x", tvm.tir.Var("", dtype="int32"))
            var_y = def_("var_y", tvm.tir.Var("", dtype="int32"))
            with T.Assert(var_x < var_y, ""):
                with T.Assert(1, "true"):
                    pass
            with T.let(var_x, var_y):
                pass
            with T.allocate([128], "uint8", "global") as alloc_x:
                with T.allocate([128], "uint8", "global") as alloc_y:
                    alloc_x, alloc_y = def_many(["alloc_x", "alloc_y"], [alloc_x, alloc_y])
            with T.allocate_const([1, 1, 1, 1, 1], "int32", [5]) as alloc_const_x:
                with T.allocate_const([10, 10, 10], "float32", [3]) as alloc_const_y:
                    alloc_const_x, alloc_const_y = def_many(
                        ["alloc_const_x", "alloc_const_y"], [alloc_const_x, alloc_const_y]
                    )
            with T.realize(buffer_x[0:var_x, 0:var_y], ""):
                with T.realize(buffer_x[var_x:128, var_y:128], ""):
                    pass
            with T.attr(buffer_x, "key_x", "value_x"):
                with T.attr(buffer_y, "key_y", "value_y"):
                    pass
            with T.launch_thread(thread_x, 4):
                with T.launch_thread(thread_y, 4):
                    pass
            with T.while_(var_x < var_y):
                with T.while_(var_x > 0):
                    pass
            with T.if_(var_x < var_y):
                with T.then_():
                    T.evaluate(0)
                    T.evaluate(1)
                with T.else_():
                    T.evaluate(0)
                    T.evaluate(1)
            with T.if_(1):
                with T.then_():
                    T.evaluate(1)
            T.prefetch(buffer_x, [Range(0, 64), Range(64, 128)])
            T.prefetch(buffer_y, [Range(0, var_x), Range(var_y, 128)])
            T.buffer_store(buffer_x, 1, [0, 0])
            T.buffer_store(buffer_x, var_x + var_y, [var_x, var_y])
            T.evaluate(var_x + var_y)
            T.evaluate(1)

    print(b.get().script())


if __name__ == "__main__":
    test_builder_root_block()
    test_builder_axis()
    test_builder_prim_func()
    test_builder_block()
    test_builder_for()
    test_builder_stmt()
