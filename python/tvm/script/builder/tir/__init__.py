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
# pylint: disable=unused-import
"""Namespace for the TVMScript TIR Builder API."""

from . import axis
from .base import TIRFrame
from .block_frame import block, where, reads, writes, alloc_buffer, block_attr, init
from .for_frame import (
    ForFrame,
    grid,
    parallel,
    serial,
    thread_binding,
    unroll,
    vectorized,
)
from .op import *
from .prim_func_frame import (
    arg,
    func_attr,
    func_name,
    func_ret,
    match_buffer,
    preflattened_buffer,
    prim_func,
)
from .var import Buffer, buffer_decl, var
from .stmt import (
    Assert,
    let,
    allocate,
    allocate_const,
    launch_thread,
    realize,
    attr,
    while_,
    if_,
    then_,
    else_,
    env_thread,
    buffer_store,
    prefetch,
    evaluate,
)
