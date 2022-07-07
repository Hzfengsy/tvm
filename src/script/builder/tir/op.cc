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

TVM_REGISTER_GLOBAL("script.builder.tir.Int8").set_body_typed(Int8);
TVM_REGISTER_GLOBAL("script.builder.tir.Int16").set_body_typed(Int16);
TVM_REGISTER_GLOBAL("script.builder.tir.Int32").set_body_typed(Int32);
TVM_REGISTER_GLOBAL("script.builder.tir.Int64").set_body_typed(Int64);
TVM_REGISTER_GLOBAL("script.builder.tir.UInt8").set_body_typed(UInt8);
TVM_REGISTER_GLOBAL("script.builder.tir.UInt16").set_body_typed(UInt16);
TVM_REGISTER_GLOBAL("script.builder.tir.UInt32").set_body_typed(UInt32);
TVM_REGISTER_GLOBAL("script.builder.tir.UInt64").set_body_typed(UInt64);
TVM_REGISTER_GLOBAL("script.builder.tir.Float8").set_body_typed(Float8);
TVM_REGISTER_GLOBAL("script.builder.tir.Float16").set_body_typed(Float16);
TVM_REGISTER_GLOBAL("script.builder.tir.Float32").set_body_typed(Float32);
TVM_REGISTER_GLOBAL("script.builder.tir.Float64").set_body_typed(Float64);
TVM_REGISTER_GLOBAL("script.builder.tir.Boolean").set_body_typed(Boolean);
TVM_REGISTER_GLOBAL("script.builder.tir.Ptr").set_body_typed(Ptr);
TVM_REGISTER_GLOBAL("script.builder.tir.PrimType").set_body_typed(PrimType);
TVM_REGISTER_GLOBAL("script.builder.tir.Handle").set_body_typed(Handle);
TVM_REGISTER_GLOBAL("script.builder.tir.min")
    .set_body_typed([](PrimExpr a, PrimExpr b) -> PrimExpr { return tvm::min(a, b); });
TVM_REGISTER_GLOBAL("script.builder.tir.max")
    .set_body_typed([](PrimExpr a, PrimExpr b) -> PrimExpr { return tvm::max(a, b); });

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm
