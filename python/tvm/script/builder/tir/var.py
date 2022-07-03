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
"""TVM Script TIR Buffer"""
from tvm._ffi import register_object as _register_object
from tvm.ir import Array, PrimExpr, Range
from tvm.runtime import DataType, Object
from tvm.tir import BufferLoad, BufferRegion, IntImm, Var

from . import _ffi_api


@_register_object("script.builder.tir.Buffer")
class Buffer_(Object):
    def __init__(
        self,
        shape,
        dtype="float32",
        name="buffer",
        data=None,
        strides=None,
        elem_offset=None,
        scope="",
        data_alignment=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Buffer,
            shape,
            dtype,
            name,
            data,
            strides,
            elem_offset,
            scope,
            data_alignment,
            offset_factor,
            buffer_type,
            axis_separators,
        )

    @property
    def data(self) -> Var:
        return self.buffer.data

    @property
    def dtype(self) -> DataType:
        return self.buffer.dtype

    @property
    def shape(self) -> Array:
        return self.buffer.shape

    @property
    def axis_separators(self) -> Array:
        return self.buffer.axis_separators

    @property
    def strides(self) -> Array:
        return self.buffer.strides

    @property
    def elem_offset(self) -> PrimExpr:
        return self.buffer.elem_offset

    @property
    def name(self) -> str:
        return self.buffer.name

    @property
    def data_alignment(self) -> int:
        return self.buffer.data_alignment

    @property
    def offset_factor(self) -> int:
        return self.buffer.offset_factor

    @property
    def buffer_type(self) -> int:
        return self.buffer.buffer_type

    def __getitem__(self, indices):
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
        if any(isinstance(index, slice) for index in indices):
            region = []
            for index in indices:
                if isinstance(index, slice):
                    region.append(Range(index.start, index.stop))
                else:
                    region.append(Range.from_min_extent(index, 1))
            return BufferRegion(self.buffer, region)
        else:
            return BufferLoad(self.buffer, indices)


class BufferProxy:
    def __call__(
        self,
        shape,
        dtype="float32",
        name="buffer",
        data=None,
        strides=None,
        elem_offset=None,
        scope="",
        data_alignment=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ) -> Buffer_:
        return Buffer_(
            shape,
            dtype,
            name,
            data,
            strides,
            elem_offset,
            scope,
            data_alignment,
            offset_factor,
            buffer_type,
            axis_separators,
        )

    def __getitem__(self, keys) -> Buffer_:
        return self(*keys)  # pylint: disable=no-member # type: ignore


Buffer = BufferProxy()
