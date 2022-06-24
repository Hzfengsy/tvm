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
#include "./ir.h"

#include "../builder.h"

namespace tvm {
namespace script {
namespace builder {
namespace ir {

IRModuleFrame IRModule() {
  ObjectPtr<IRModuleFrameNode> n = make_object<IRModuleFrameNode>();
  n->global_vars.clear();
  n->functions.clear();
  return IRModuleFrame(n);
}

void IRModuleFrameNode::ExitWithScope() {
  ICHECK_EQ(functions.size(), global_vars.size());
  int n = functions.size();
  Map<GlobalVar, BaseFunc> func_map;
  for (int i = 0; i < n; ++i) {
    func_map.Set(global_vars[i], functions[i]);
  }
  Builder builder = Builder::Current();
  ICHECK(!builder->result.defined()) << "ValueError: Builder.result has already been set";
  builder->result = tvm::IRModule(func_map);
}

TVM_REGISTER_NODE_TYPE(IRModuleFrameNode);
TVM_REGISTER_GLOBAL("script.builder.ir.IRModule").set_body_typed(IRModule);

}  // namespace ir
}  // namespace builder
}  // namespace script
}  // namespace tvm
