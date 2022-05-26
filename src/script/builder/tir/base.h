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
#ifndef TVM_SCRIPT_BUILDER_TIR_BASE_H_
#define TVM_SCRIPT_BUILDER_TIR_BASE_H_

#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/var.h>

#include "../builder.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

class TIRFrameNode : public FrameNode {
 public:
  Array<tvm::tir::Stmt> stmts;

  void VisitAttrs(tvm::AttrVisitor* v) {
    FrameNode::VisitAttrs(v);
    v->Visit("stmts", &stmts);
  }

  static constexpr const char* _type_key = "script.builder.tir.TIRFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(TIRFrameNode, FrameNode);
};

class TIRFrame : public Frame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRFrame, Frame, TIRFrameNode);

 protected:
  TIRFrame() = default;
};

inline void AddToParent(tvm::tir::Stmt stmt) {
  Builder builder = Builder::Current();
  if (builder->frames.empty()) {
    ICHECK(!builder->result.defined()) << "ValueError: Builder.result has already been set";
    builder->result = stmt;
  } else if (const auto* tir_frame = builder->frames.back().as<TIRFrameNode>()) {
    GetRef<TIRFrame>(tir_frame)->stmts.push_back(stmt);
  } else {
    LOG(FATAL) << "TypeError: Unsupported frame type: " << builder->frames.back();
  }
}

inline tvm::tir::Stmt AsStmt(const Array<tvm::tir::Stmt>& stmt) {
  using namespace tvm::tir;
  if (stmt.empty()) {
    return Evaluate(0);
  } else if (stmt.size() == 1) {
    return stmt[0];
  } else {
    return SeqStmt(stmt);
  }
}

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_BASE_H_
