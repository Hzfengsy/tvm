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
import inspect
from contextlib import contextmanager
from typing import Any, Callable, Dict


def deferred(f: Callable[[], None]):
    @contextmanager
    def context():
        try:
            yield
        finally:
            f()

    return context()


def inspect_function_capture(func: Callable) -> Dict[str, Any]:
    PREFIX = "tvm.script.builder."
    vars = {}
    closure_vars = inspect.getclosurevars(func)
    captured = {**closure_vars.nonlocals, **closure_vars.globals}
    for k, v in captured.items():
        # Case 1: a moduel like `T` or `tvm.script.builder.tir`
        if inspect.ismodule(v) and v.__name__.startswith(PREFIX):
            vars[k] = v
            continue
        # Case 2: a function like `T.match_buffer`
        if hasattr(v, "__module__") and v.__module__.startswith(PREFIX):
            vars[k] = v
            continue
        # Case 3: atomic types
        if v is None or isinstance(v, (int, float, str, bool)):
            vars[k] = v
            continue
    return vars


def inspect_class_capture(cls: type) -> Dict[str, Any]:
    vars: Dict[str, Any] = {}
    for _, v in cls.__dict__.items():
        if inspect.isfunction(v):
            func_vars = inspect_function_capture(v)
            vars.update(**func_vars)
    return vars
