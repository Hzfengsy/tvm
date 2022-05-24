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
#include "./for_frame.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

ForFrame::ForFrame(Array<tvm::tir::Var> loop_vars, ForFrame::FMakeForLoop f_make_for_loop) {
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->loop_vars = std::move(loop_vars);
  n->f_make_for_loop = std::move(f_make_for_loop);
  data_ = std::move(n);
}

#define TVM_SCRIPT_BUILDER_TIR_FOR_CREATE(Method, Kind)                                         \
  With<ForFrame> Method(PrimExpr min, PrimExpr extent, Map<String, ObjectRef> attrs) {          \
    ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();                                    \
    int bits = std::max(min.dtype().bits(), extent.dtype().bits());                             \
    n->loop_vars = {tvm::tir::Var("v", DataType::Int(bits))};                                   \
    n->f_make_for_loop = [=](Array<tvm::tir::Var> vars, tvm::tir::Stmt body) -> tvm::tir::For { \
      ICHECK_EQ(vars.size(), 1);                                                                \
      return tvm::tir::For(/*loop_var=*/vars[0], min, extent, Kind, body,                       \
                           /*thread_binding=*/NullOpt, attrs);                                  \
    };                                                                                          \
    return With<ForFrame>(n);                                                                   \
  }

TVM_SCRIPT_BUILDER_TIR_FOR_CREATE(Serial, tvm::tir::ForKind::kSerial);
TVM_SCRIPT_BUILDER_TIR_FOR_CREATE(Parallel, tvm::tir::ForKind::kParallel);
TVM_SCRIPT_BUILDER_TIR_FOR_CREATE(Vectorized, tvm::tir::ForKind::kVectorized);
TVM_SCRIPT_BUILDER_TIR_FOR_CREATE(Unroll, tvm::tir::ForKind::kUnrolled);

#undef TVM_SCRIPT_BUILDER_TIR_FOR_CREATE

With<ForFrame> ThreadBinding(PrimExpr min, PrimExpr extent, String thread,
                             Map<String, ObjectRef> attrs) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  int bits = std::max(min.dtype().bits(), extent.dtype().bits());
  n->loop_vars = {Var("v", DataType::Int(bits))};
  n->f_make_for_loop = [=](Array<Var> vars, Stmt body) -> For {
    ICHECK_EQ(vars.size(), 1);
    IterVar iter_var(Range(nullptr), Var(ObjectPtr<Object>(nullptr)), IterVarType::kThreadIndex,
                     thread);
    return For(vars[0], min, extent, tvm::tir::ForKind::kThreadBinding, body, iter_var, attrs);
  };
  return With<ForFrame>(n);
}

With<ForFrame> Grid(Array<PrimExpr> extents) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->loop_vars.reserve(extents.size());
  for (const auto& extent : extents) {
    n->loop_vars.push_back(Var("v", extent.dtype()));
  }
  n->f_make_for_loop = [=](Array<Var> vars, Stmt body) -> Stmt {
    ICHECK_EQ(extents.size(), vars.size());
    int n = extents.size();
    for (int i = n - 1; i >= 0; --i) {
      Var var = vars[i];
      PrimExpr extent = extents[i];
      body = For(var, Integer(0), extent, ForKind::kSerial, body, /*thread_binding=*/NullOpt, {});
    }
    return body;
  };
  return With<ForFrame>(n);
}

TVM_REGISTER_NODE_TYPE(ForFrameNode);

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm
