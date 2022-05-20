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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

inline Optional<tir::LoopRV> FuseSpatialAndBindBlockIdx(const tir::Schedule& sch,
                                                        const tir::BlockRV& block_rv) {
  using namespace tvm::tir;
  Array<StmtSRef> loops = tir::GetLoops(sch->GetSRef(block_rv));
  int n = loops.size();
  if (n == 0) {
    return NullOpt;
  }
  int i_block_idx = -1;
  int i_thread_idx = -1;
  int i_multi_child = -1;
  int i_spatial_loop = -1;
  for (int i = 0; i < n; ++i) {
    const StmtSRef& loop_sref = loops[i];
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    runtime::ThreadScope thread_scope = GetThreadScope(loop);
    if (IsBlockIdx(thread_scope)) {
      if (i_block_idx == -1) {
        i_block_idx = i;
      }
    }
    if (IsThreadIdx(thread_scope)) {
      if (i_thread_idx == -1) {
        i_thread_idx = i;
      }
    }
    if (loop->kind != ForKind::kSerial) {
      if (i_multi_child == -1) {
        i_multi_child = i;
      }
    }
    if (!IsSingleStmt(loop->body)) {
      if (i_multi_child == -1) {
        i_multi_child = i + 1;
      }
    }
    if (GetLoopIterType(loop_sref) == IterVarType::kDataPar) {
      if (i_spatial_loop == i - 1) {
        ++i_spatial_loop;
      }
    }
  }
  if (i_multi_child == -1) {
    i_multi_child = n;
  }
  if ((i_block_idx != -1 && i_thread_idx != -1) || i_spatial_loop == -1) {
    return NullOpt;
  }
  if (i_block_idx != -1 && i_thread_idx == -1) {
    ICHECK(false) << "Unsupported case, where blockIdx is bound but threadIdx is not";
    throw;
  }
  Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
  if (i_block_idx == -1 && i_thread_idx != -1) {
    int num_fuse = std::min(std::min(i_multi_child, i_thread_idx), i_spatial_loop + 1);
    LoopRV fused = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + num_fuse});
    sch->Bind(fused, "blockIdx.x");
    return NullOpt;
  } else {  // i_block_idx == -1 && i_thread_idx == -1
    int num_fuse = std::min(i_multi_child, i_spatial_loop + 1);
    LoopRV fused = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + num_fuse});
    return fused;
  }
}

inline void BindBlockThreadIdx(const tir::Schedule& sch, const tir::LoopRV& loop_rv,
                               int64_t max_threadblocks, int64_t max_threads_per_block,
                               std::function<tir::ExprRV(int64_t)> get_factor) {
  using namespace tvm::tir;
  int64_t extent = -1;
  if (const int64_t* e = GetLoopIntExtent(sch->Get(loop_rv).get())) {
    extent = *e;
    extent = std::min(extent, max_threads_per_block);
  } else {
    extent = max_threads_per_block;
  }
  if (extent <= max_threadblocks * max_threads_per_block) {
    ExprRV factor = get_factor(extent);
    Array<LoopRV> splits = sch->Split(loop_rv, {NullOpt, factor});
    ICHECK_EQ(splits.size(), 2);
    sch->Bind(splits[0], "blockIdx.x");
    sch->Bind(splits[1], "threadIdx.x");
  } else {
    Array<LoopRV> splits = sch->Split(loop_rv, {NullOpt,
                                                Integer(max_threadblocks),  //
                                                Integer(max_threads_per_block)});
    ICHECK_EQ(splits.size(), 3);
    sch->Reorder({splits[1], splits[2], splits[0]});
    sch->Bind(splits[1], "blockIdx.x");
    sch->Bind(splits[2], "threadIdx.x");
  }
}

inline std::function<tir::ExprRV(int64_t)> MakeFactorSampler(tir::Schedule sch,
                                                             Array<Integer> thread_extents) {
  return [sch = std::move(sch),
          thread_extents = std::move(thread_extents)](int64_t max_extent) -> tir::ExprRV {
    Array<Integer> extents;
    extents.reserve(thread_extents.size());
    for (const Integer extent : thread_extents) {
      if (extent->value <= max_extent) {
        extents.push_back(extent);
      }
    }
    int n = extents.size();
    if (n == 0) {
      return Integer(max_extent);
    }
    if (n == 1) {
      return Integer(extents[0]);
    }
    Array<FloatImm> probs(n, FloatImm(DataType::Float(64), 1.0 / n));
    return sch->SampleCategorical(extents, probs);
  };
}

}  // namespace meta_schedule
}  // namespace tvm
