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
"""TVM Script TIR Block Frame"""
from typing import Any, Dict, List, Union

from tvm._ffi import register_object as _register_object
from tvm.tir import Buffer, BufferLoad, BufferRegion

from . import _ffi_api
from .base import TIRFrame


@_register_object("script.builder.tir.BlockFrame")
class BlockFrame(TIRFrame):
    ...


@_register_object("script.builder.tir.BlockInitFrame")
class BlockInitFrame(TIRFrame):
    ...


def block(name: str, no_realize: bool = False) -> BlockFrame:
    return _ffi_api.BlockFrame(name, no_realize)  # pylint: disable=no-member # type: ignore


def init() -> BlockInitFrame:
    return _ffi_api.BlockInitFrame()  # pylint: disable=no-member # type: ignore


def where(predicate) -> None:
    _ffi_api.Where(predicate)  # pylint: disable=no-member # type: ignore


def reads(buffer_slices: List[Union[BufferRegion, BufferLoad]]) -> None:
    if not isinstance(buffer_slices, List):
        buffer_slices = [buffer_slices]
    _ffi_api.Reads(buffer_slices)


def writes(buffer_slices: List[Union[BufferRegion, BufferLoad]]) -> None:
    if not isinstance(buffer_slices, List):
        buffer_slices = [buffer_slices]
    _ffi_api.Writes(buffer_slices)


def block_attr(attrs: Dict[str, Any]) -> None:
    return _ffi_api.BlockAttrs(attrs)  # pylint: disable=no-member # type: ignore


def alloc_buffer(
    shape,
    dtype="float32",
    data=None,
    strides=[],
    elem_offset=None,
    storage_scope="",
    align=-1,
    offset_factor=0,
    buffer_type="default",
    axis_separators=None,
) -> Buffer:
    return _ffi_api.AllocBuffer(
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        storage_scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )
