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
#include "./op.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

PrimExpr prim_type(String type_name, PrimExpr expr) {
  return cast(DataType(runtime::String2DLDataType(type_name)), expr);
}

TVM_REGISTER_GLOBAL("script.builder.tir.PrimType").set_body_typed(prim_type);
TVM_REGISTER_GLOBAL("script.builder.tir.Handle").set_body_typed(handle);
TVM_REGISTER_GLOBAL("script.builder.tir.min").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return tvm::min(a, b, span);
});
TVM_REGISTER_GLOBAL("script.builder.tir.max").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return tvm::max(a, b, span);
});

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm
