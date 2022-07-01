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
#ifndef TVM_SCRIPT_BUILDER_TIR_STMT_H_
#define TVM_SCRIPT_BUILDER_TIR_STMT_H_

#include "./base.h"
#include "./var.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

class AssertFrameNode : public TIRFrameNode {
 public:
  PrimExpr condition;
  PrimExpr message;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("condition", &condition);
    v->Visit("message", &message);
  }

  static constexpr const char* _type_key = "script.builder.tir.AssertFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AssertFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class AssertFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AssertFrame, TIRFrame, AssertFrameNode);
};

class LetFrameNode : public TIRFrameNode {
 public:
  tvm::tir::Var var;
  PrimExpr value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("var", &var);
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "script.builder.tir.LetFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(LetFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class LetFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(LetFrame, TIRFrame, LetFrameNode);
};

class AllocateFrameNode : public TIRFrameNode {
 public:
  Array<PrimExpr> extents;
  DataType dtype;
  String storage_scope;
  PrimExpr condition;
  Map<String, ObjectRef> annotations;
  Buffer buffer{nullptr};

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("extents", &extents);
    v->Visit("dtype", &dtype);
    v->Visit("storage_scope", &storage_scope);
    v->Visit("condition", &condition);
    v->Visit("annotations", &annotations);
    v->Visit("buffer", &buffer);
  }

  static constexpr const char* _type_key = "script.builder.tir.AllocateFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocateFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class AllocateFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AllocateFrame, TIRFrame, AllocateFrameNode);
};

class AllocateConstFrameNode : public TIRFrameNode {
 public:
  DataType dtype;
  Array<PrimExpr> extents;
  tvm::runtime::NDArray data;
  Buffer buffer{nullptr};
  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("dtype", &dtype);
    v->Visit("extents", &extents);
    v->Visit("data", &data);
    v->Visit("buffer", &buffer);
  }

  static constexpr const char* _type_key = "script.builder.tir.AllocateConstFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocateConstFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class AllocateConstFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AllocateConstFrame, TIRFrame,
                                                    AllocateConstFrameNode);
};

class LaunchThreadFrameNode : public TIRFrameNode {
 public:
  PrimExpr extent;
  String attr_key;
  tvm::tir::IterVar iter_var;
  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("extent", &extent);
    v->Visit("attr_key", &attr_key);
    v->Visit("iter_var", &iter_var);
  }

  static constexpr const char* _type_key = "script.builder.tir.LaunchThreadFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(LaunchThreadFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class LaunchThreadFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(LaunchThreadFrame, TIRFrame,
                                                    LaunchThreadFrameNode);
};

class RealizeFrameNode : public TIRFrameNode {
 public:
  tvm::tir::BufferRegion buffer_slice;
  String storage_scope;
  PrimExpr condition;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("buffer_slice", &buffer_slice);
    v->Visit("storage_scope", &storage_scope);
    v->Visit("condition", &condition);
  }

  static constexpr const char* _type_key = "script.builder.tir.RealizeFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(RealizeFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class RealizeFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(RealizeFrame, TIRFrame, RealizeFrameNode);
};

class AttrFrameNode : public TIRFrameNode {
 public:
  ObjectRef node;
  String attr_key;
  PrimExpr value;
  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("node", &node);
    v->Visit("attr_key", &attr_key);
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "script.builder.tir.AttrFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class AttrFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AttrFrame, TIRFrame, AttrFrameNode);
};

class WhileFrameNode : public TIRFrameNode {
 public:
  PrimExpr condition;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("condition", &condition);
  }

  static constexpr const char* _type_key = "script.builder.tir.WhileFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(WhileFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class WhileFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(WhileFrame, TIRFrame, WhileFrameNode);
};

class IfFrameNode : public TIRFrameNode {
 public:
  PrimExpr condition;
  Optional<Array<tvm::tir::Stmt>> then_stmts;
  Optional<Array<tvm::tir::Stmt>> else_stmts;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("condition", &condition);
    v->Visit("then_stmts", &then_stmts);
    v->Visit("else_stmts", &else_stmts);
  }

  static constexpr const char* _type_key = "script.builder.tir.IfFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(IfFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class IfFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IfFrame, TIRFrame, IfFrameNode);
};

class ThenFrameNode : public TIRFrameNode {
 public:
  static constexpr const char* _type_key = "script.builder.tir.ThenFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ThenFrameNode, TIRFrameNode);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class ThenFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ThenFrame, TIRFrame, ThenFrameNode);
};

class ElseFrameNode : public TIRFrameNode {
 public:
  static constexpr const char* _type_key = "script.builder.tir.ElseFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ElseFrameNode, TIRFrameNode);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class ElseFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ElseFrame, TIRFrame, ElseFrameNode);
};

tvm::tir::IterVar EnvThread(String thread_tag);
void BufferStore(Buffer buffer, PrimExpr value, Array<PrimExpr> indices);
void Prefetch(Buffer buffer, Array<Range> bounds);
void Evaluate(PrimExpr value);

AssertFrame Assert(PrimExpr condition, String message);
LetFrame Let(tvm::tir::Var var, PrimExpr value);
AllocateFrame Allocate(Array<PrimExpr> extents, DataType dtype, String storage_scope = "",
                       Optional<PrimExpr> condition = NullOpt,
                       Optional<Map<String, ObjectRef>> annotations = NullOpt);
AllocateConstFrame AllocateConst(tvm::runtime::NDArray data, DataType dtype,
                                 Array<PrimExpr> extents);
LaunchThreadFrame LaunchThread(tvm::tir::IterVar iter_var, PrimExpr extent);
RealizeFrame Realize(tvm::tir::BufferRegion buffer_slice, String storage_scope, PrimExpr condition);
AttrFrame Attr(ObjectRef node, String attr_key, PrimExpr value);
WhileFrame While(PrimExpr condition);
IfFrame If(PrimExpr condition);
ThenFrame Then();
ElseFrame Else();
}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_STMT_H_
