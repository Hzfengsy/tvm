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
from typing import List

from tvm._ffi import register_object as _register_object
from tvm.tir import Var

from . import _ffi_api
from .. import _ffi_api as _base_ffi_api
from .base import TIRFrame


@_register_object("script.builder.tir.ForFrame")
class ForFrame(TIRFrame):
    def __enter__(self) -> List[Var]:
        _base_ffi_api.FrameEnter(self)  # pylint: disable=no-member # type: ignore
        return self.vars if len(self.vars) > 1 else self.vars[0]


def serial(start, stop=None, *, annotations=None) -> ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Serial(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def parallel(start, stop=None, *, annotations=None) -> ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Parallel(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def vectorized(start, stop=None, *, annotations=None) -> ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Vectorized(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def unroll(start, stop=None, *, annotations=None) -> ForFrame:
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Unroll(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def thread_binding(start, stop=None, thread=None, *, annotations=None) -> ForFrame:
    if thread is None:
        if not isinstance(stop, str):
            raise ValueError("Thread cannot be None for thread_binding")
        thread = stop
        stop = start
        start = 0
    elif stop is None:
        stop = start
        start = 0
    return _ffi_api.ThreadBinding(  # pylint: disable=no-member # type: ignore
        start, stop, thread, annotations
    )


def grid(*extents) -> ForFrame:
    return _ffi_api.Grid(extents)  # pylint: disable=no-member # type: ignore
