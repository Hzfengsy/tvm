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
#ifndef TVM_SCRIPT_BUILDER_TIR_PRIM_FUNC_FRAME_H_
#define TVM_SCRIPT_BUILDER_TIR_PRIM_FUNC_FRAME_H_

#include "./base.h"
#include "./block_frame.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

class PrimFuncFrameNode : public TIRFrameNode {
 public:
  Optional<String> name;
  Array<tvm::tir::Var> args;
  Optional<Type> ret_type;
  Map<tvm::tir::Var, tvm::tir::Buffer> buffer_map;
  Map<tvm::tir::Var, tvm::tir::Buffer> preflattened_buffer_map;
  Map<String, ObjectRef> attrs;
  BlockFrame root_block_frame{nullptr};

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("args", &args);
    v->Visit("ret_type", &ret_type);
    v->Visit("buffer_map", &buffer_map);
    v->Visit("preflattened_buffer_map", &preflattened_buffer_map);
    v->Visit("attrs", &attrs);
    v->Visit("root_block_frame", &root_block_frame);
  }

  static constexpr const char* _type_key = "script.builder.tir.PrimFuncFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimFuncFrameNode, TIRFrameNode);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class PrimFuncFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrimFuncFrame, TIRFrame, PrimFuncFrameNode);
};

PrimFuncFrame PrimFunc_();
PrimFuncFrame FindPrimFuncFrame(const String& method);
tvm::tir::Var Arg(String name, tvm::tir::Var var);
tvm::tir::Buffer Arg(String name, tvm::tir::Buffer buffer);
void FuncName(String name);
void FuncAttrs(Map<String, ObjectRef> attrs);
tvm::Type FuncRet(tvm::Type ret_type);

tvm::tir::Buffer MatchBuffer(ObjectRef param, Array<PrimExpr> shape,
                             DataType dtype = DataType::Float(32),
                             Optional<tvm::tir::Var> data = NullOpt, Array<PrimExpr> strides = {},
                             PrimExpr elem_offset = PrimExpr(), String storage_scope = "",
                             int align = -1, int offset_factor = 0,
                             String buffer_type_str = "default", Array<IntImm> axis_separators = {},
                             Span span = Span());

void PreflattenedBuffer(tvm::tir::Buffer postflattened_buffer, Array<PrimExpr> shape,
                        DataType dtype = DataType::Float(32),
                        Optional<tvm::tir::Var> data = NullOpt, Array<PrimExpr> strides = {},
                        PrimExpr elem_offset = PrimExpr(), String storage_scope = "",
                        int align = -1, int offset_factor = 0, String buffer_type_str = "default",
                        Array<IntImm> axis_separators = {}, Span span = Span());

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_PRIM_FUNC_FRAME_H_
