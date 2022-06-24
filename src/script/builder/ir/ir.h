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
#ifndef TVM_SCRIPT_BUILDER_IR_IR_H_
#define TVM_SCRIPT_BUILDER_IR_IR_H_

#include "../frame.h"

namespace tvm {
namespace script {
namespace builder {
namespace ir {

class IRModuleFrameNode : public FrameNode {
 public:
  Array<GlobalVar> global_vars;
  Array<BaseFunc> functions;

  void VisitAttrs(tvm::AttrVisitor* v) {
    FrameNode::VisitAttrs(v);
    v->Visit("global_vars", &global_vars);
    v->Visit("functions", &functions);
  }

  static constexpr const char* _type_key = "script.builder.ir.IRModuleFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(IRModuleFrameNode, FrameNode);

 public:
  void ExitWithScope() final;
};

class IRModuleFrame : public Frame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRModuleFrame, Frame, IRModuleFrameNode);
};

IRModuleFrame IRModule();

}  // namespace ir
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_IR_IR_H_
