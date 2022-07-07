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
"""TVM Script TIR Axis"""

from tvm.ir import Range
from tvm.tir import IterVar

from . import _ffi_api

from typing import List, Union


def spatial(dom, binding, dtype="int32") -> IterVar:
    if not isinstance(dom, Range):
        dom = Range(0, dom)
    return _ffi_api.AxisSpatial(dom, binding, dtype)  # pylint: disable=no-member # type: ignore


def reduce(dom, binding, dtype="int32") -> IterVar:
    if not isinstance(dom, Range):
        dom = Range(0, dom)
    return _ffi_api.AxisReduce(dom, binding, dtype)  # pylint: disable=no-member # type: ignore


def scan(dom, binding, dtype="int32") -> IterVar:
    if not isinstance(dom, Range):
        dom = Range(0, dom)
    return _ffi_api.AxisScan(dom, binding, dtype)  # pylint: disable=no-member # type: ignore


def opaque(dom, binding, dtype="int32") -> IterVar:
    if not isinstance(dom, Range):
        dom = Range(0, dom)
    return _ffi_api.AxisOpaque(dom, binding, dtype)  # pylint: disable=no-member # type: ignore


def remap(kinds, bindings, dtype="int32") -> Union[List[IterVar], IterVar]:
    iter_vars = _ffi_api.AxisRemap(kinds, bindings, dtype)
    return (
        iter_vars[0] if len(iter_vars) == 1 else iter_vars
    )  # pylint: disable=no-member # type: ignore


S = spatial
R = reduce
