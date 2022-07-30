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

#include "./var.h"

#include "../builder.h"

namespace tvm {
namespace script {
namespace builder {
namespace relax {

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::relax::VarNode>([](const ObjectRef& node, String name) -> void {
      using tvm::relax::VarNode;
      VarNode* var = const_cast<VarNode*>(node.as<VarNode>());
      var->vid = tvm::relax::Id(name);
    });

tvm::relax::Var Tensor(Array<PrimExpr> shape, DataType dtype) {
  using namespace tvm::relax;
  int ndim = shape.size();
  Type dyn_tensor_type = DynTensorType(ndim, dtype);
  ShapeExpr shape_expr = ShapeExpr(shape);
  return Var("", shape_expr, dyn_tensor_type);
}

TVM_REGISTER_GLOBAL("script.builder.relax.Tensor").set_body_typed(Tensor);

}  // namespace relax
}  // namespace builder
}  // namespace script
}  // namespace tvm

}  // namespace relax
}  // namespace builder
}  // namespace script
}  // namespace tvm
