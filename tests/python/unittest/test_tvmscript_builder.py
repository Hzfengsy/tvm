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


def test_builder_basic():
    b = Builder()
    with b:
        with T.prim_func(name="main"):
            A = T.arg("A", T.Buffer((128, 128, 128), "float32"))
            B = T.arg("B", T.Buffer((128, 128, 128), "float32"))
            with T.grid(128, 128, 128) as (i, j, k):
                def_many(["i", "j", "k"], [i, j, k])
                with T.block(name="block"):
                    vi = def_("vi", T.axis.spatial(128, i))
                    vj = def_("vj", T.axis.spatial(128, j))
                    vk = def_("vk", T.axis.reduce(128, k))
    print(b.get().script())
    tvm._ffi.get_global_func("test_poc")()


if __name__ == "__main__":
    test_builder_basic()
