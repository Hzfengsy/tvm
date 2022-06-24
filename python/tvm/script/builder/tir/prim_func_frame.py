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
import inspect
from typing import Any, Callable, Dict, Optional, Union

from tvm._ffi import register_object as _register_object
from tvm.ir import Type
from tvm.tir import Buffer, PrimFunc
from tvm.tir.expr import Var

from . import _ffi_api
from .base import TIRFrame


@_register_object("script.builder.tir.PrimFuncFrame")
class PrimFuncFrame(TIRFrame):
    ...


def _is_defined_in_class(frames):
    if len(frames) > 2:
        maybe_class_frame = frames[2]
        statement_list = maybe_class_frame[4]
        first_statement = statement_list[0]
        if first_statement.strip().startswith("class "):
            return True
    return False


def prim_func(f: Optional[Callable] = None) -> Union[PrimFuncFrame, PrimFunc, Callable]:
    if f is not None:
        from tvm.script.parse import parse  # pylint: disable=import-outside-toplevel

        if not inspect.isfunction(f):
            raise TypeError(f"Expect a function, but got: {f}")

        if _is_defined_in_class(inspect.stack()):
            return f
        return parse(f)
    return _ffi_api.PrimFunc()  # pylint: disable=no-member # type: ignore


setattr(prim_func, "dispatch_token", "tir")


def arg(name, obj) -> Union[Var, Buffer]:
    return _ffi_api.Arg(name, obj)  # pylint: disable=no-member # type: ignore


def func_name(name) -> str:
    return _ffi_api.FuncName(name)  # pylint: disable=no-member # type: ignore


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
) -> Buffer:
    return _ffi_api.MatchBuffer(  # pylint: disable=no-member # type: ignore
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
) -> None:
    _ffi_api.PreflattenedBuffer(  # pylint: disable=no-member # type: ignore
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
    )
