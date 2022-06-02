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
"""TVM Script TIR For Frame"""
from tvm._ffi import register_object as _register_object

from tvm.tir import Var

from . import _ffi_api
from ._ffi_api import _base_ffi_api
from .base import TIRFrame
from typing import List


@_register_object("script.builder.tir.ForFrame")
class ForFrame(TIRFrame):
    def __enter__(self) -> List[Var]:
        _base_ffi_api.FrameEnter(self)
        return self.vars


def serial(min_val, extent, attrs) -> ForFrame:
    return _ffi_api.Serial(min_val, extent, attrs)


def parallel(min_val, extent, attrs) -> ForFrame:
    return _ffi_api.Parallel(min_val, extent, attrs)


def vectorized(min_val, extent, attrs) -> ForFrame:
    return _ffi_api.Vectorized(min_val, extent, attrs)


def unroll(min_val, extent, attrs) -> ForFrame:
    return _ffi_api.Unroll(min_val, extent, attrs)


def thread_binding(min_val, extent, attrs) -> ForFrame:
    return _ffi_api.ThreadBinding(min_val, extent, attrs)


def grid(*extents) -> ForFrame:
    return _ffi_api.Grid(extents)
