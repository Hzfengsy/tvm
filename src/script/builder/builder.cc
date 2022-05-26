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
#include "./builder.h"

namespace tvm {
namespace script {
namespace builder {

Builder::Builder() {
  ObjectPtr<BuilderNode> n = make_object<BuilderNode>();
  n->frames.clear();
  n->result = NullOpt;
  data_ = n;
}

std::vector<Builder>* ThreadLocalBuilderStack() {
  thread_local std::vector<Builder> stack;
  return &stack;
}

void Builder::EnterWithScope() {
  BuilderNode* n = this->get();
  CHECK(n->frames.empty()) << "ValueError: There are frame(s) left in the builder: "
                           << n->frames.size()
                           << ". Please use a fresh new builder every time building IRs";
  n->frames.push_back(IRModuleFrame());
  std::vector<Builder>* stack = ThreadLocalBuilderStack();
  stack->push_back(*this);
}

void Builder::ExitWithScope() {
  BuilderNode* n = this->get();
  ICHECK_EQ(n->frames.size(), 1);
  IRModuleFrame frame = Downcast<IRModuleFrame>(n->frames.back());
  n->frames.pop_back();
  std::vector<Builder>* stack = ThreadLocalBuilderStack();
  ICHECK(!stack->empty());
  stack->pop_back();
  if (!frame->stmts.empty()) {
    ICHECK(frame->global_vars.empty());
    ICHECK(frame->functions.empty());
    n->result = frame->stmts;
  } else {
    Map<GlobalVar, BaseFunc> func_map;
    ICHECK_EQ(frame->functions.size(), frame->global_vars.size());
    int m = frame->functions.size();
    for (int i = 0; i < m; ++i) {
      func_map.Set(frame->global_vars[i], frame->functions[i]);
    }
  }
}

Builder Builder::Current() {
  std::vector<Builder>* stack = ThreadLocalBuilderStack();
  CHECK(!stack->empty()) << "ValueError: No builder in current scope";
  return stack->back();
}

TVM_REGISTER_NODE_TYPE(BuilderNode);

}  // namespace builder
}  // namespace script
}  // namespace tvm
