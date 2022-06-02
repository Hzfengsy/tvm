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
"""TVM Script IR Builder"""
from typing import List, TypeVar

from tvm._ffi import register_object as _register_object
from tvm.runtime import Object

from . import _ffi_api
from .frame import Frame


@_register_object("script.builder.Builder")
class Builder(Object):
    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Builder  # pylint: disable=no-member # type: ignore
        )

    def __enter__(self) -> "Builder":
        _ffi_api.BuilderEnter(self)  # pylint: disable=no-member # type: ignore
        return self

    def __exit__(self, ptype, value, trace) -> None:  # pylint: disable=unused-argument
        _ffi_api.BuilderExit(self)  # pylint: disable=no-member # type: ignore

    @staticmethod
    def current() -> "Builder":
        return _ffi_api.BuilderCurrent()  # pylint: disable=no-member # type: ignore

    def get(self) -> Frame:
        return _ffi_api.BuilderGet(self)  # pylint: disable=no-member # type: ignore


DefType = TypeVar("DefType", bound=Object)


def def_(name: str, var: DefType) -> DefType:
    return _ffi_api.Def(name, var)  # pylint: disable=no-member # type: ignore


def def_many(
    names: List[str],
    vars: List[DefType],  # pylint: disable=redefine-builtin
) -> List[DefType]:
    assert len(names) == len(vars)
    return [def_(name, var) for name, var in zip(names, vars)]
