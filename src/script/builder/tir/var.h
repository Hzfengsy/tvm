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
#ifndef TVM_SCRIPT_BUILDER_TIR_VAR_H_
#define TVM_SCRIPT_BUILDER_TIR_VAR_H_

#include "./base.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

class BufferNode : public runtime::Object {
 public:
  tvm::tir::Buffer buffer;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("buffer", &buffer); }

  tvm::tir::BufferLoad operator[](Array<PrimExpr> indices) const {
    return tvm::tir::BufferLoad(buffer, indices);
  }

  tvm::tir::BufferRegion operator[](Array<Range> region) const {
    return tvm::tir::BufferRegion(buffer, region);
  }

  static constexpr const char* _type_key = "script.builder.tir.Buffer";
  TVM_DECLARE_BASE_OBJECT_INFO(BufferNode, runtime::Object);
};

class Buffer : public runtime::ObjectRef {
 public:
  TVM_DLL Buffer(Array<PrimExpr> shape, DataType dtype = DataType::Float(32),
                 String buffer_name = "buffer", Optional<tvm::tir::Var> data = NullOpt,
                 Optional<Array<PrimExpr>> strides = NullOpt,
                 Optional<PrimExpr> elem_offset = NullOpt, String storage_scope = "", int align = 0,
                 int offset_factor = 0, String buffer_type_str = "",
                 Optional<Array<IntImm>> axis_separators = NullOpt);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Buffer, ObjectRef, BufferNode);
};

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_VAR_H_
