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
#ifndef TVM_SCRIPT_BUILDER_RELAX_FUNCTION_FRAME_H_
#define TVM_SCRIPT_BUILDER_RELAX_FUNCTION_FRAME_H_

#include <tvm/relax/block_builder.h>

#include "./base.h"

namespace tvm {
namespace script {
namespace builder {
namespace relax {

class FunctionFrameNode : public RelaxFrameNode {
 public:
  Optional<String> name;
  Array<tvm::relax::Var> params;
  Optional<Type> ret_type;
  Map<String, ObjectRef> attrs;
  Array<tvm::relax::BindingBlock> binding_blocks;
  Array<tvm::relax::Expr> outputs;
  tvm::relax::BlockBuilder block_builder;

  void VisitAttrs(tvm::AttrVisitor* v) {
    RelaxFrameNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("params", &params);
    v->Visit("ret_type", &ret_type);
    v->Visit("attrs", &attrs);
    v->Visit("binding_blocks", &binding_blocks);
    v->Visit("outputs", &outputs);
    // `block_builder` is not visited.
  }

  static constexpr const char* _type_key = "script.builder.relax.PrimFuncFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionFrameNode, RelaxFrameNode);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class FunctionFrame : public RelaxFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(FunctionFrame, RelaxFrame, FunctionFrameNode);
};

FunctionFrame Function();
tvm::relax::Var Arg(const String& name, const tvm::relax::Var& var);
void FuncName(const String& name);
void FuncAttrs(Map<String, ObjectRef> attrs);
tvm::Type FuncRet(tvm::Type ret_type);

}  // namespace relax
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_PRIM_FUNC_FRAME_H_
