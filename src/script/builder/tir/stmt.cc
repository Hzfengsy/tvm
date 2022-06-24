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
#include "./stmt.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>

#include "./prim_func_frame.h"
#include "./var.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

void AssertFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AssertStmt(condition, message, AsStmt(stmts)));
}

void LetFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::LetStmt(var, value, AsStmt(stmts)));
}

void AllocateFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  tvm::tir::Buffer flattened_buffer = buffer.GetFlattenedBuffer();
  AddToParent(tvm::tir::Allocate(buffer->data, flattened_buffer->dtype, flattened_buffer->shape,
                                 condition, AsStmt(stmts), annotations));
}

void AllocateConstFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AllocateConst(buffer->data, dtype, extents, data, AsStmt(stmts)));
}

void LaunchThreadFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AttrStmt(iter_var, attr_key, extent, AsStmt(stmts)));
}

void RealizeFrameNode::ExitWithScope() {
  using namespace tvm::tir;
  TIRFrameNode::ExitWithScope();
  AddToParent(AttrStmt(
      buffer_slice->buffer, "realize_scope", StringImm(storage_scope),
      BufferRealize(buffer_slice->buffer, buffer_slice->region, condition, AsStmt(stmts))));
}

void AttrFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AttrStmt(node, attr_key, value, AsStmt(stmts)));
}

void WhileFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::While(condition, AsStmt(stmts)));
}

void IfFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  if (!stmts.empty()) {
    LOG(FATAL) << "stmt within IfThenElse frame should be either in ThenFrame or ElseFrame";
  }
  if (!then_stmts.defined()) {
    LOG(FATAL) << "IfThenElse frame should have at least one then branch";
  }
  AddToParent(tvm::tir::IfThenElse(
      condition, AsStmt(then_stmts.value()),
      else_stmts.defined() ? AsStmt(else_stmts.value()) : tvm::tir::Stmt(nullptr)));
}

IfFrame FindIfFrame(const String& method) {
  if (Optional<IfFrame> if_frame = Builder::Current()->GetLastFrame<IfFrame>()) {
    return if_frame.value();
  } else {
    LOG(FATAL) << "ValueError: IfThenElse frame not find. Please ensure '" << method
               << "' is called under T.if_()";
  }
  throw;
}

void ThenFrameNode::EnterWithScope() {
  IfFrame frame = FindIfFrame("T.then_");
  if (frame->then_stmts.defined()) {
    LOG(FATAL) << "ValueError: Duplicate then branch declaration, previous one is "
               << frame->then_stmts.value();
  }
  TIRFrameNode::EnterWithScope();
}

void ThenFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  FindIfFrame("T.then_")->then_stmts = stmts;
}

void ElseFrameNode::EnterWithScope() {
  IfFrame frame = FindIfFrame("T.else_");
  if (!frame->then_stmts.defined()) {
    LOG(FATAL) << "The else branch should follow then branch";
  }
  if (frame->else_stmts.defined()) {
    LOG(FATAL) << "ValueError: Duplicate else branch declaration, previous one is "
               << frame->else_stmts.value();
  }
  TIRFrameNode::EnterWithScope();
}

void ElseFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  FindIfFrame("T.else_")->else_stmts = stmts;
}

AssertFrame Assert(PrimExpr condition, String message) {
  ObjectPtr<AssertFrameNode> n = make_object<AssertFrameNode>();
  n->condition = condition;
  n->message = tvm::tir::StringImm(message);
  return AssertFrame(n);
}

LetFrame Let(tvm::tir::Var var, PrimExpr value) {
  ObjectPtr<LetFrameNode> n = make_object<LetFrameNode>();
  n->var = var;
  n->value = value;
  return LetFrame(n);
}

AllocateFrame Allocate(Array<PrimExpr> extents, DataType dtype, String storage_scope,
                       Optional<PrimExpr> condition, Optional<Map<String, ObjectRef>> annotations) {
  ObjectPtr<AllocateFrameNode> n = make_object<AllocateFrameNode>();
  n->extents = extents;
  n->dtype = dtype;
  n->storage_scope = storage_scope;
  n->condition = condition.value_or(true);
  if (!n->condition->dtype.is_bool()) {
    n->condition = tvm::cast(DataType::Bool(), n->condition);
  }
  n->annotations = annotations.value_or(Map<String, ObjectRef>());
  n->buffer =
      DeclBuffer(extents, dtype, "", NullOpt, {}, PrimExpr(), storage_scope, 0, 0, "default", {});
  return AllocateFrame(n);
}

AllocateConstFrame AllocateConst(tvm::runtime::NDArray data, DataType dtype,
                                 Array<PrimExpr> extents) {
  ObjectPtr<AllocateConstFrameNode> n = make_object<AllocateConstFrameNode>();
  n->dtype = dtype;
  n->extents = extents;
  n->data = data;
  n->buffer = DeclBuffer(extents, dtype, "", NullOpt, {}, PrimExpr(), "", 0, 0, "default", {});
  return AllocateConstFrame(n);
}

LaunchThreadFrame LaunchThread(tvm::tir::IterVar iter_var, PrimExpr extent) {
  ObjectPtr<LaunchThreadFrameNode> n = make_object<LaunchThreadFrameNode>();
  if (!iter_var->dom.defined()) {
    const_cast<tvm::tir::IterVarNode*>(iter_var.get())->dom = Range(0, extent);
  } else if (!arith::Analyzer().CanProveEqual(iter_var->dom->extent, extent)) {
    LOG(FATAL) << "ValueError: Inconsistent extents of environment thread. "
               << iter_var->dom->extent << " vs " << extent;
  }
  n->iter_var = iter_var;
  n->extent = extent;
  n->attr_key = iter_var->thread_tag == "vthread" ? "virtual_thread" : "thread_extent";
  return LaunchThreadFrame(n);
}

RealizeFrame Realize(tvm::tir::BufferRegion buffer_slice, String storage_scope,
                     PrimExpr condition) {
  ObjectPtr<RealizeFrameNode> n = make_object<RealizeFrameNode>();
  n->buffer_slice = buffer_slice;
  n->storage_scope = storage_scope;
  n->condition = condition;
  return RealizeFrame(n);
}

AttrFrame Attr(ObjectRef node, String attr_key, PrimExpr value) {
  ObjectPtr<AttrFrameNode> n = make_object<AttrFrameNode>();
  n->node = node;
  n->attr_key = attr_key;
  n->value = value;
  return AttrFrame(n);
}

WhileFrame While(PrimExpr condition) {
  ObjectPtr<WhileFrameNode> n = make_object<WhileFrameNode>();
  n->condition = condition;
  return WhileFrame(n);
}

IfFrame If(PrimExpr condition) {
  ObjectPtr<IfFrameNode> n = make_object<IfFrameNode>();
  n->condition = condition;
  n->then_stmts = NullOpt;
  n->else_stmts = NullOpt;
  return IfFrame(n);
}

ThenFrame Then() {
  ObjectPtr<ThenFrameNode> n = make_object<ThenFrameNode>();
  return ThenFrame(n);
}

ElseFrame Else() {
  ObjectPtr<ElseFrameNode> n = make_object<ElseFrameNode>();
  return ElseFrame(n);
}

tvm::tir::IterVar EnvThread(String thread_tag) {
  using namespace tvm::tir;
  return IterVar(Range{nullptr}, Var("", DataType::Int(32)), IterVarType::kThreadIndex, thread_tag);
}

void BufferStore(tvm::tir::Buffer buffer, PrimExpr value, Array<PrimExpr> indices) {
  AddToParent(tvm::tir::BufferStore(buffer, value, indices));
}

void Prefetch(tvm::tir::Buffer buffer, Array<Range> bounds) {
  AddToParent(tvm::tir::Prefetch(buffer, bounds));
}

void Evaluate(PrimExpr value) { AddToParent(tvm::tir::Evaluate(value)); }

TVM_REGISTER_NODE_TYPE(AssertFrameNode);
TVM_REGISTER_NODE_TYPE(LetFrameNode);
TVM_REGISTER_NODE_TYPE(AllocateFrameNode);
TVM_REGISTER_NODE_TYPE(AllocateConstFrameNode);
TVM_REGISTER_NODE_TYPE(LaunchThreadFrameNode);
TVM_REGISTER_NODE_TYPE(RealizeFrameNode);
TVM_REGISTER_NODE_TYPE(AttrFrameNode);
TVM_REGISTER_NODE_TYPE(WhileFrameNode);
TVM_REGISTER_NODE_TYPE(IfFrameNode);
TVM_REGISTER_NODE_TYPE(ThenFrameNode);
TVM_REGISTER_NODE_TYPE(ElseFrameNode);
TVM_REGISTER_GLOBAL("script.builder.tir.AssertFrame").set_body_typed(Assert);
TVM_REGISTER_GLOBAL("script.builder.tir.LetFrame").set_body_typed(Let);
TVM_REGISTER_GLOBAL("script.builder.tir.AllocateFrame").set_body_typed(Allocate);
TVM_REGISTER_GLOBAL("script.builder.tir.AllocateConstFrame").set_body_typed(AllocateConst);
TVM_REGISTER_GLOBAL("script.builder.tir.RealizeFrame").set_body_typed(Realize);
TVM_REGISTER_GLOBAL("script.builder.tir.AttrFrame").set_body_typed(Attr);
TVM_REGISTER_GLOBAL("script.builder.tir.WhileFrame").set_body_typed(While);
TVM_REGISTER_GLOBAL("script.builder.tir.IfFrame").set_body_typed(If);
TVM_REGISTER_GLOBAL("script.builder.tir.ThenFrame").set_body_typed(Then);
TVM_REGISTER_GLOBAL("script.builder.tir.ElseFrame").set_body_typed(Else);
TVM_REGISTER_GLOBAL("script.builder.tir.LaunchThreadFrame").set_body_typed(LaunchThread);
TVM_REGISTER_GLOBAL("script.builder.tir.EnvThread").set_body_typed(EnvThread);
TVM_REGISTER_GLOBAL("script.builder.tir.BufferStore").set_body_typed(BufferStore);
TVM_REGISTER_GLOBAL("script.builder.tir.Prefetch").set_body_typed(Prefetch);
TVM_REGISTER_GLOBAL("script.builder.tir.Evaluate").set_body_typed(Evaluate);

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm
