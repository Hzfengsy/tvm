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
"""The dispatcher"""

from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

from .doc import AST

if TYPE_CHECKING:
    from .parser import Parser


ParseMethod = Callable[
    ["Parser", AST],
    None,
]


class DispatchTable:
    """Dispatch table for parse methods"""

    _instance: Optional["DispatchTable"] = None
    table: Dict[Tuple[str, str], ParseMethod]

    def __init__(self):
        self.table = {}


DispatchTable._instance = DispatchTable()  # pylint: disable=protected-access


def register(
    token: str,
    type_name: str,
):
    """Register a method for a dispatch token and type name"""

    def f(method: ParseMethod):
        DispatchTable._instance.table[  # pylint: disable=protected-access
            (token, type_name)
        ] = method

    return f


def get(
    token: str,
    type_name: str,
    default: Optional[ParseMethod] = None,
) -> Optional[ParseMethod]:
    return DispatchTable._instance.table.get(  # pylint: disable=protected-access
        (token, type_name),
        default,
    )
