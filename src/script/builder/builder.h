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
#ifndef TVM_SCRIPT_BUILDER_BUILDER_H_
#define TVM_SCRIPT_BUILDER_BUILDER_H_

#include <tvm/node/node.h>

#include "./frame.h"

namespace tvm {
namespace script {
namespace builder {

class BuilderNode : public runtime::Object {
 public:
  runtime::Array<Frame> frames;
  Optional<ObjectRef> result;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("frames", &frames);
    v->Visit("result", &result);
  }

  static constexpr const char* _type_key = "script.builder.Builder";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuilderNode, runtime::Object);

 public:
  template <typename TFrame>
  inline Optional<TFrame> FindFrame() const;
  template <typename TFrame>
  inline Optional<TFrame> GetLastFrame() const;

  template <typename TObjectRef>
  inline TObjectRef Get() const;
};

class Builder : public runtime::ObjectRef {
 public:
  Builder();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Builder, ObjectRef, BuilderNode);

 public:
  void EnterWithScope();
  void ExitWithScope();
  static Builder Current();
};

template <class TObjectRef>
inline TObjectRef Def(String name, TObjectRef obj);

namespace details {
ObjectRef DefImpl(String name, ObjectRef obj);
}

class Namer {
 public:
  using FType = NodeFunctor<void(const ObjectRef&, String)>;
  static FType& vtable();

  static void Name(ObjectRef node, String name);
};

template <class TObjectRef>
inline TObjectRef Def(String name, TObjectRef obj) {
  return Downcast<TObjectRef>(details::DefImpl(name, obj));
}

template <typename TFrame>
inline Optional<TFrame> BuilderNode::FindFrame() const {
  using TFrameNode = typename TFrame::ContainerType;
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    if (const TFrameNode* p = (*it).template as<TFrameNode>()) {
      return GetRef<TFrame>(p);
    }
  }
  return NullOpt;
}

template <typename TFrame>
inline Optional<TFrame> BuilderNode::GetLastFrame() const {
  using TFrameNode = typename TFrame::ContainerType;
  if (!frames.empty() && frames.back()->IsInstance<TFrameNode>()) {
    return Downcast<TFrame>(frames.back());
  }
  return NullOpt;
}

template <typename TObjectRef>
inline TObjectRef BuilderNode::Get() const {
  using TObject = typename TObjectRef::ContainerType;
  CHECK(result.defined()) << "IndexError: No result exists in IRBuilder yet";
  const auto* n = result.as<TObject>();
  CHECK(n != nullptr) << "IndexError: IRBuilder result is not of type: " << TObject::_type_key;
  return GetRef<TObjectRef>(n);
}

}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_BUILDER_H_
