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

namespace tvm {
namespace script {
namespace builder {
namespace tir {

Buffer::Buffer(Array<PrimExpr> shape, DataType dtype, String buffer_name,
               Optional<tvm::tir::Var> data, Optional<Array<PrimExpr>> strides,
               Optional<PrimExpr> elem_offset, String storage_scope, int align, int offset_factor,
               String buffer_type_str, Optional<Array<IntImm>> axis_separators) {
  ObjectPtr<BufferNode> n = make_object<BufferNode>();
  tvm::tir::Var buffer_data;
  if (!data.defined()) {
    DataType storage_dtype = dtype;
    if (storage_dtype == DataType::Bool()) {
      storage_dtype = DataType::Int(8);
    }
    buffer_data = tvm::tir::Var(buffer_name, PointerType(PrimType(storage_dtype), storage_scope));
  } else {
    buffer_data = data.value();
  }
  tvm::tir::BufferType buffer_type =
      (buffer_type_str == "auto_broadcast") ? tvm::tir::kAutoBroadcast : tvm::tir::kDefault;
  n->buffer = tvm::tir::Buffer(buffer_data, dtype, shape, strides.value_or(Array<PrimExpr>()),
                               elem_offset.value_or(PrimExpr()), buffer_name, align, offset_factor,
                               buffer_type, axis_separators.value_or(Array<IntImm>()));
  data_ = n;
}

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tir::BufferNode>([](const ObjectRef& node, String name) -> void {
      tvm::tir::BufferNode* buffer =
          const_cast<tvm::tir::BufferNode*>(node.as<tvm::tir::BufferNode>());
      buffer->name = name;
      Namer::Name(buffer->data, name + "_data");
      int n = buffer->strides.size();
      for (int i = 0; i < n; ++i) {
        PrimExpr e = buffer->strides[i];
        if (const tvm::tir::VarNode* v = e.as<tvm::tir::VarNode>()) {
          Namer::Name(GetRef<tvm::tir::Var>(v), name + "_s" + std::to_string(i));
        }
      }
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<BufferNode>([](const ObjectRef& node, String name) -> void {
      Namer::Name(node.as<BufferNode>()->buffer, name);
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tir::SizeVarNode>([](const ObjectRef& node, String name) -> void {
      using namespace tvm::tir;
      SizeVarNode* var = const_cast<SizeVarNode*>(node.as<SizeVarNode>());
      var->name_hint = name;
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tir::VarNode>([](const ObjectRef& node, String name) -> void {
      using namespace tvm::tir;
      VarNode* var = const_cast<VarNode*>(node.as<VarNode>());
      var->name_hint = name;
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tir::IterVarNode>([](const ObjectRef& node, String name) -> void {
      using namespace tvm::tir;
      IterVarNode* var = const_cast<IterVarNode*>(node.as<IterVarNode>());
      Namer::Name(var->var, name);
    });

TVM_REGISTER_NODE_TYPE(BufferNode);
TVM_REGISTER_GLOBAL("script.builder.tir.Buffer")
    .set_body_typed([](Array<PrimExpr> shape, DataType dtype, String buffer_name,
                       Optional<tvm::tir::Var> data, Optional<Array<PrimExpr>> strides,
                       Optional<PrimExpr> elem_offset, String storage_scope, int align,
                       int offset_factor, String buffer_type_str,
                       Optional<Array<IntImm>> axis_separators) {
      return Buffer(shape, dtype, buffer_name, data, strides, elem_offset, storage_scope, align,
                    offset_factor, buffer_type_str, axis_separators);
    });

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm
