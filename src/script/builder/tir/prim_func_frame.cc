/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "./prim_func_frame.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

void Arg(tvm::tir::Var var) {
  PrimFuncFrame frame = Builder::Current()->FindFrame<PrimFuncFrame>().value();
  frame->args.push_back(var);
}

void Arg(tvm::tir::Buffer buffer) {
  using namespace tvm::tir;
  PrimFuncFrame frame = Builder::Current()->FindFrame<PrimFuncFrame>().value();
  Var handle(buffer->name + "_handle", DataType::Handle());
  frame->args.push_back(handle);
  frame->buffer_map.Set(handle, buffer);
}

TVM_REGISTER_NODE_TYPE(PrimFuncFrameNode);

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm
