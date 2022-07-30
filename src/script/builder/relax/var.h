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
#ifndef TVM_SCRIPT_BUILDER_RELAX_VAR_H_
#define TVM_SCRIPT_BUILDER_RELAX_VAR_H_

#include <tvm/relax/expr.h>

#include "./base.h"

namespace tvm {
namespace script {
namespace builder {
namespace relax {

tvm::relax::Var Tensor(Array<PrimExpr> shape, DataType dtype);

// TODO(@Siyuan): add dyn shape support

}  // namespace relax
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_VAR_H_
