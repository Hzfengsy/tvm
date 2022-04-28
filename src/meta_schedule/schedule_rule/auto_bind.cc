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

class AutoBindNode : public ScheduleRuleNode {
 public:
  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {
    CHECK(context->target.defined()) << "ValueError: target is not defined";
    Optional<Integer> max_num_threads =
        context->target.value()->GetAttr<Integer>("max_threads_per_block");
    CHECK(max_num_threads.defined())
        << "ValueError: missing attribute `max_threads_per_block` in the target";
    this->max_num_threads_ = max_num_threads.value();
  }

  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final;

 public:
  /*! \brief The max number of threads per block from Target */
  int max_num_threads_ = -1;
  /*! \brief The max number of threadblocks in the cuda device */
  int max_threadblock_ = -1;
  /*!
   * \brief thread_extents Candidates of thread axis extent. Use `max_num_threads_` if it's empty.
   */
  Array<Integer> thread_extents;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `max_num_threads_` is not visited
    // `max_threadblock_` is not visited
    v->Visit("thread_extents", &thread_extents);
  }

  static constexpr const char* _type_key = "meta_schedule.AutoBind";
  TVM_DECLARE_FINAL_OBJECT_INFO(AutoBindNode, ScheduleRuleNode);
};

Array<tir::Schedule> AutoBindNode::Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) {
  ICHECK_NE(this->max_num_threads_, -1);

  return {BindThreadsForUnboundBlock(sch, block_rv, max_num_threads_, max_threadblock_,
                                     thread_extents)};
}

ScheduleRule ScheduleRule::AutoBind(int max_threadblock,  //
                                    Array<Integer> thread_extents) {
  ObjectPtr<AutoBindNode> n = make_object<AutoBindNode>();
  n->max_threadblock_ = max_threadblock;
  n->max_num_threads_ = -1;
  n->thread_extents = std::move(thread_extents);
  return ScheduleRule(n);
}

TVM_REGISTER_NODE_TYPE(AutoBindNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleAutoBind").set_body_typed(ScheduleRule::AutoBind);

}  // namespace meta_schedule
}  // namespace tvm
