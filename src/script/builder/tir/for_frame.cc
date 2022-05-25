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

ForFrame::ForFrame(Array<tvm::tir::Var> vars, Array<Range> doms,
                   ForFrameNode::FMakeForLoop f_make_for_loop) {
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars = std::move(vars);
  n->doms = std::move(doms);
  n->f_make_for_loop = std::move(f_make_for_loop);
  data_ = std::move(n);
}

#define TVM_SCRIPT_BUILDER_TIR_FOR_CREATE(Method, Kind)                               \
  ForFrame Method(PrimExpr min, PrimExpr extent, Map<String, ObjectRef> attrs) {      \
    using namespace tvm::tir;                                                         \
    ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();                          \
    int bits = std::max(min.dtype().bits(), extent.dtype().bits());                   \
    n->vars = {Var("v", DataType::Int(bits))};                                        \
    n->doms = {Range(min, extent)};                                                   \
    n->f_make_for_loop = [attrs](Array<Var> vars, Array<Range> doms, Stmt body) {     \
      ICHECK_EQ(vars.size(), 1);                                                      \
      ICHECK_EQ(doms.size(), 1);                                                      \
      return For(vars[0], doms[0]->min, doms[0]->extent, Kind, body, NullOpt, attrs); \
    };                                                                                \
    return ForFrame(n);                                                               \
  }

TVM_SCRIPT_BUILDER_TIR_FOR_CREATE(Serial, tvm::tir::ForKind::kSerial);
TVM_SCRIPT_BUILDER_TIR_FOR_CREATE(Parallel, tvm::tir::ForKind::kParallel);
TVM_SCRIPT_BUILDER_TIR_FOR_CREATE(Vectorized, tvm::tir::ForKind::kVectorized);
TVM_SCRIPT_BUILDER_TIR_FOR_CREATE(Unroll, tvm::tir::ForKind::kUnrolled);

#undef TVM_SCRIPT_BUILDER_TIR_FOR_CREATE

ForFrame ThreadBinding(PrimExpr min, PrimExpr extent, String thread, Map<String, ObjectRef> attrs) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  int bits = std::max(min.dtype().bits(), extent.dtype().bits());
  n->vars = {Var("v", DataType::Int(bits))};
  n->doms = {Range(min, extent)};
  n->f_make_for_loop = [attrs, thread](Array<Var> vars, Array<Range> doms, Stmt body) -> For {
    ICHECK_EQ(vars.size(), 1);
    ICHECK_EQ(doms.size(), 1);
    IterVar iter_var(Range(nullptr), NullValue<Var>(), IterVarType::kThreadIndex, thread);
    return For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kThreadBinding, body, iter_var,
               attrs);
  };
  return ForFrame(n);
}

ForFrame Grid(Array<PrimExpr> extents) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  for (const auto& extent : extents) {
    DataType dtype = extent.dtype();
    n->vars.push_back(Var("v", extent.dtype()));
    n->doms.push_back(Range(make_const(dtype, 0), extent));
  }
  n->f_make_for_loop = [](Array<Var> vars, Array<Range> doms, Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      body = For(var, dom->min, dom->extent, ForKind::kSerial, std::move(body),
                 /*thread_binding=*/NullOpt, /*annotations=*/{});
    }
    return body;
  };
  return ForFrame(n);
}

TVM_REGISTER_NODE_TYPE(ForFrameNode);

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm
