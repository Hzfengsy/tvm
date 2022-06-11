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
"""TVM Script TIR Prim Func Frame"""
from typing import Union, Dict, Any

from tvm._ffi import register_object as _register_object
from tvm.tir.buffer import Buffer
from tvm.tir.expr import Var
from tvm.ir import Type

from ..builder import Builder
from . import _ffi_api
from .base import TIRFrame


@_register_object("script.builder.tir.PrimFuncFrame")
class PrimFuncFrame(TIRFrame):
    ...


def prim_func(name) -> PrimFuncFrame:
    return _ffi_api.PrimFuncFrame(name)  # pylint: disable=no-member # type: ignore


def arg(name, obj) -> Union[Var, Buffer]:
    return _ffi_api.Arg(name, obj)  # pylint: disable=no-member # type: ignore


setattr(prim_func, "dispatch_token", "tir")


def func_attr(attrs: Dict[str, Any]) -> None:
    return _ffi_api.FuncAttrs(attrs)  # pylint: disable=no-member # type: ignore


def func_ret(ret_type) -> Type:
    return _ffi_api.FuncRet(ret_type)  # pylint: disable=no-member # type: ignore


def match_buffer(
    param,
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
    span=None,
) -> Buffer:
    return _ffi_api.MatchBuffer(
        param,
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
        span,
    )


def preflattened_buffer(
    postflattened,
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
    span=None,
) -> None:
    _ffi_api.PreflattenedBuffer(
        postflattened,
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
        span,
    )
