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
#include "./base.h"

#include <tvm/support/with.h>

#include "./block_frame.h"
#include "./for_frame.h"
#include "./prim_func_frame.h"
#include "./var.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

TVM_REGISTER_NODE_TYPE(TIRFrameNode);

void TestPOC() {
  namespace T = tvm::script::builder::tir;
  using namespace ::tvm::tir;

  With<Builder> builder;
  {
    With<PrimFuncFrame> _{T::PrimFunc_("main")};
    Buffer A = T::Buffer_({128, 128, 128}, DataType::Float(32));
    Buffer B = T::Buffer_({128, 128, 128}, DataType::Float(32));
    {
      With<ForFrame> _{T::Grid({128, 128, 128})};
      Var i = _()->vars[0];
      Var j = _()->vars[1];
      Var k = _()->vars[2];
      {
        With<BlockFrame> _{T::Block_("block")};
        IterVar vi = T::axis::Spatial(Range(0, 128), i);
        IterVar vj = T::axis::Spatial(Range(0, 128), j);
        IterVar vk = T::axis::Spatial(Range(0, 128), k);
      }
    }
  }
}

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm
