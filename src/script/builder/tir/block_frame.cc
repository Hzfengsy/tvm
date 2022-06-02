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
#include "./block_frame.h"

#include <tvm/runtime/registry.h>

#include "./for_frame.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

BlockFrame Block_(String name) {
  ObjectPtr<BlockFrameNode> n = make_object<BlockFrameNode>();
  n->name = name;
  n->iter_vars.clear();
  n->reads.clear();
  n->writes.clear();
  n->init = NullOpt;
  n->alloc_buffers.clear();
  n->match_buffers.clear();
  n->annotations.clear();
  n->iter_values.clear();
  n->predicate = NullOpt;
  return BlockFrame(n);
}

void BlockFrameNode::ExitWithScope() {
  using namespace tvm::tir;
  TIRFrameNode::ExitWithScope();
  AddToParent(BlockRealize(iter_values,  //
                           predicate.value_or(Bool(true)),
                           Block(iter_vars,      //
                                 reads, writes,  //
                                 name,           //
                                 AsStmt(stmts),  //
                                 init,           //
                                 alloc_buffers,  //
                                 match_buffers,  //
                                 annotations)));
}

namespace axis {

// TODO(@junrushao1994): figure out the Block syntax without BlockRealize

tvm::tir::IterVar PushBlockVar(tvm::tir::IterVar iter_var, PrimExpr binding) {
  if (const BlockFrameNode* opt_frame = Builder::Current()->frames.back().as<BlockFrameNode>()) {
    BlockFrame frame = GetRef<BlockFrame>(opt_frame);
    frame->iter_vars.push_back(iter_var);
    frame->iter_values.push_back(binding);
  } else {
    LOG(FATAL) << "TypeError: The last frame is not BlockFrame";
  }
  return iter_var;
}

tvm::tir::IterVar Spatial(Range dom, PrimExpr binding, DataType dtype) {
  using namespace tvm::tir;
  ICHECK(dom.defined()) << "Spatial axis must have a domain";
  int bits = std::max({dom->min.dtype().bits(), dom->extent.dtype().bits(), dtype.bits()});
  return PushBlockVar(IterVar(/*dom=*/dom,                              //
                              /*var=*/Var("_", dtype.with_bits(bits)),  //
                              /*iter_type=*/IterVarType::kDataPar,      //
                              /*thread_tag=*/""),
                      binding);
}

tvm::tir::IterVar Reduce(Range dom, PrimExpr binding, DataType dtype) {
  using namespace tvm::tir;
  ICHECK(dom.defined()) << "Reduction axis must have a domain";
  int bits = std::max({dom->min.dtype().bits(), dom->extent.dtype().bits(), dtype.bits()});
  return PushBlockVar(IterVar(/*dom=*/dom,                              //
                              /*var=*/Var("_", dtype.with_bits(bits)),  //
                              /*iter_type=*/IterVarType::kCommReduce,   //
                              /*thread_tag=*/""),
                      binding);
}

Array<tvm::tir::IterVar> Remap(String kinds, Array<PrimExpr> bindings, DataType dtype) {
  using namespace tvm::tir;
  Array<IterVar> results;
  ICHECK_EQ(kinds.size(), bindings.size());
  int n = bindings.size();
  results.reserve(n);
  for (int i = 0; i < n; ++i) {
    char c = kinds.c_str()[i];
    PrimExpr e = bindings[i];
    const VarNode* v = e.as<VarNode>();
    ICHECK(v) << "TypeError: Only Var is supported in T.axis.remap";
    Range dom{nullptr};
    for (const auto& frame : Builder::Current()->frames) {
      if (const auto* for_frame = frame.as<ForFrameNode>()) {
        ICHECK_EQ(for_frame->doms.size(), for_frame->vars.size());
        int n = for_frame->doms.size();
        for (int i = 0; i < n; ++i) {
          if (for_frame->vars[i].get() == v) {
            dom = for_frame->doms[i];
            break;
          }
        }
        if (dom.defined()) {
          break;
        }
      }
    }
    ICHECK(dom.defined()) << "TypeError: Variable is not in the loop: " << GetRef<Var>(v);
    DataType dtype = v->dtype;
    if (c == 'S') {
      results.push_back(PushBlockVar(IterVar(/*dom=*/dom,
                                             /*var=*/Var("_", dtype),
                                             /*iter_type=*/IterVarType::kDataPar,
                                             /*thread_tag=*/""),
                                     e));
    } else if (c == 'R') {
      results.push_back(PushBlockVar(IterVar(/*dom=*/dom,
                                             /*var=*/Var("_", dtype),
                                             /*iter_type=*/IterVarType::kCommReduce,
                                             /*thread_tag=*/""),
                                     e));
    } else {
      LOG(FATAL) << "Unknown axis kind: " << c;
    }
  }
  return results;
}

}  // namespace axis

TVM_REGISTER_NODE_TYPE(BlockFrameNode);

TVM_REGISTER_GLOBAL("script.builder.tir.BlockFrame").set_body_typed(Block_);

TVM_REGISTER_GLOBAL("script.builder.tir.AxisSpatial").set_body_typed(axis::Spatial);

TVM_REGISTER_GLOBAL("script.builder.tir.AxisReduce").set_body_typed(axis::Reduce);

TVM_REGISTER_GLOBAL("script.builder.tir.AxisRemap").set_body_typed(axis::Remap);

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm
