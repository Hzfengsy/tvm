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

import contextlib
from functools import partial
from typing import Any

from ...builder import Frame, def_
from ...builder import tir as T
from .. import dispatch, doc
from ..parser import Parser


@dispatch.register(token="relax", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    with self.var_table.with_frame():
        self.var_table.add("range", T.serial)
        with T.prim_func():
            T.func_name(node.name)
            with self.with_dispatch_token("tir"):
                # TODO: define the GlobalVar, handle the return value
                self.visit(node.args)
                self.visit_body(node.body)
