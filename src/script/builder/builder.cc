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

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/registry.h>

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
  n->result = NullOpt;
  std::vector<Builder>* stack = ThreadLocalBuilderStack();
  stack->push_back(*this);
}

void Builder::ExitWithScope() {
  std::vector<Builder>* stack = ThreadLocalBuilderStack();
  ICHECK(!stack->empty());
  stack->pop_back();
}

Builder Builder::Current() {
  std::vector<Builder>* stack = ThreadLocalBuilderStack();
  CHECK(!stack->empty()) << "ValueError: No builder in current scope";
  return stack->back();
}

Namer::FType& Namer::vtable() {
  static FType inst;
  return inst;
}

void Namer::Name(ObjectRef node, String name) {
  static const FType& f = vtable();
  CHECK(node.defined()) << "ValueError: Cannot name nullptr with: " << name;
  CHECK(f.can_dispatch(node)) << "ValueError: Do not know how to name type \""
                              << node->GetTypeKey();
  f(node, name);
}

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::runtime::ArrayNode>([](const ObjectRef& node, String name) -> void {
      using namespace tvm::runtime;
      ArrayNode* array = const_cast<ArrayNode*>(node.as<ArrayNode>());
      ICHECK(array);
      int n = array->size();
      for (int i = 0; i < n; ++i) {
        Namer::Name(array->at(i), name + std::to_string(i));
      }
    });

namespace details {

ObjectRef DefImpl(String name, ObjectRef obj) {
  Namer::Name(obj, name);
  return obj;
}

}  // namespace details

TVM_REGISTER_NODE_TYPE(BuilderNode);

TVM_REGISTER_GLOBAL("script.builder.Builder").set_body_typed([]() { return Builder(); });
TVM_REGISTER_GLOBAL("script.builder.BuilderEnter").set_body_method(&Builder::EnterWithScope);
TVM_REGISTER_GLOBAL("script.builder.BuilderExit").set_body_method(&Builder::ExitWithScope);
TVM_REGISTER_GLOBAL("script.builder.BuilderCurrent").set_body_typed(Builder::Current);
TVM_REGISTER_GLOBAL("script.builder.BuilderGet")
    .set_body_method<Builder>(&BuilderNode::Get<ObjectRef>);
TVM_REGISTER_GLOBAL("script.builder.Def").set_body_typed(Def<ObjectRef>);

}  // namespace builder
}  // namespace script
}  // namespace tvm
