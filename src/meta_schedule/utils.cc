
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

#include "./utils.h"

namespace tvm {
namespace tir {

bool IsUnboundBlock(const StmtSRef& block_sref) {
  for (const StmtSRefNode* p = block_sref->parent; p != nullptr; p = p->parent) {
    if (p->stmt->IsInstance<ForNode>()) {
      For loop = Downcast<For>(GetRef<Stmt>(p->stmt));
      if (loop->kind == ForKind::kThreadBinding) return false;
    }
  }
  return true;
}

/*!
 * \brief Check the combination of bindings to be added to the block
 * \param block_sref The block to be checked
 * \param fuse_first_num The number of loops to be fused
 * \return The type of binding to be added to the block
 */
BindType GetBindType(const StmtSRef& block_sref, int* fuse_first_num) {
  Array<StmtSRef> loops = tir::GetLoops(block_sref);
  int n = loops.size();
  if (n == 0) {
    return BindType::kNoBind;
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
    if (loop->kind != tir::ForKind::kSerial) {
      if (i_multi_child == -1) {
        i_multi_child = i;
      }
    }
    if (!IsSingleStmt(loop->body)) {
      if (i_multi_child == -1) {
        i_multi_child = i + 1;
      }
    }
    if (tir::GetLoopIterType(loop_sref) == IterVarType::kDataPar) {
      if (i_spatial_loop == i - 1) {
        ++i_spatial_loop;
      }
    }
  }
  if (i_multi_child == -1) {
    i_multi_child = n;
  }
  if ((i_block_idx != -1 && i_thread_idx != -1) || i_spatial_loop == -1) {
    return BindType::kNoBind;
  } else if (i_block_idx != -1 && i_thread_idx == -1) {
    ICHECK(false) << "Unsupported case, where blockIdx is bound but threadIdx is not";
    throw;
  } else if (i_block_idx == -1 && i_thread_idx != -1) {
    *fuse_first_num = std::min(std::min(i_multi_child, i_thread_idx), i_spatial_loop + 1);
    return BindType::kBindBlock;
  } else {  // i_block_idx == -1 && i_thread_idx == -1
    *fuse_first_num = std::min(i_multi_child, i_spatial_loop + 1);
    return BindType::kBindBlockThread;
  }
}

Schedule BindThreadsForUnboundBlock(const Schedule& sch,      //
                                    const BlockRV& block_rv,  //
                                    int max_threadblock,      //
                                    int max_num_threads,      //
                                    Array<Integer> thread_extents) {
  tir::StmtSRef block_sref = sch->GetSRef(block_rv);

  int fuse_first_num = 0;
  tir::BindType bind_type = tir::GetBindType(block_sref, &fuse_first_num);
  if (bind_type == tir::BindType::kNoBind) {
    return {sch};
  }

  Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
  LoopRV fused = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + fuse_first_num});
  if (bind_type == tir::BindType::kBindBlock) {
    sch->Bind(fused, "blockIdx.x");
  } else if (bind_type == tir::BindType::kBindBlockThread) {
    int64_t extent_size = int64_t(1) << 60;
    if (const int64_t* extent_ptr = tir::GetLoopIntExtent(sch->Get(fused).get())) {
      extent_size = *extent_ptr;
    }

    Array<Integer> updated_extents;
    for (const Integer extent : thread_extents) {
      if (extent->value <= extent_size) updated_extents.push_back(extent);
    }

    if (extent_size <= max_threadblock * max_num_threads) {
      tir::ExprRV factor;
      if (updated_extents.empty()) {
        factor = Integer(std::min(static_cast<int64_t>(max_num_threads), extent_size));
      } else if (updated_extents.size() == 1) {
        factor = updated_extents[0];
      } else {
        // Sample a factor
        int n = updated_extents.size();
        Array<FloatImm> probs(n, FloatImm(DataType::Float(64), 1.0 / n));
        factor = sch->SampleCategorical(updated_extents, probs);
      }
      Array<LoopRV> splits = sch->Split(fused, {NullOpt, factor});
      ICHECK_EQ(splits.size(), 2);
      sch->Bind(splits[0], "blockIdx.x");
      sch->Bind(splits[1], "threadIdx.x");
    } else {
      Array<LoopRV> splits =
          sch->Split(fused, {NullOpt, Integer(max_threadblock), Integer(max_num_threads)});
      ICHECK_EQ(splits.size(), 3);
      sch->Reorder({splits[1], splits[2], splits[0]});
      sch->Bind(splits[1], "blockIdx.x");
      sch->Bind(splits[2], "threadIdx.x");
    }
  }
  return {sch};
}
}  // namespace tir
}  // namespace tvm
