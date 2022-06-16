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
#ifndef TVM_SCRIPT_BUILDER_TIR_UTILS_H_
#define TVM_SCRIPT_BUILDER_TIR_UTILS_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace script {
namespace builder {
namespace tir {

tvm::tir::BufferRegion BufferRegionFromLoad(tvm::tir::BufferLoad buffer_load) {
  using namespace tvm::tir;
  Array<Range> ranges;
  for (const PrimExpr& index : buffer_load->indices) {
    ranges.push_back(Range::FromMinExtent(index, 1));
  }
  return BufferRegion(buffer_load->buffer, ranges);
}

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_UTILS_H_
