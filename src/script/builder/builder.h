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

namespace tvm {
namespace script {
namespace builder {

class FrameNode : public runtime::Object {
 public:
  std::vector<runtime::TypedPackedFunc<void()>> callbacks;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `callbacks` is not visited.
  }

  void AddCallback(runtime::TypedPackedFunc<void()> callback) { callbacks.push_back(callback); }

  static constexpr const char* _type_key = "script.builder.Frame";
  TVM_DECLARE_BASE_OBJECT_INFO(FrameNode, runtime::Object);

 public:
  virtual ~FrameNode() {
    for (auto it = callbacks.rbegin(); it != callbacks.rend(); ++it) {
      (*it)();
    }
  }
};

class Frame : public runtime::ObjectRef {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Frame, ObjectRef, FrameNode);

 protected:
  Frame() = default;
};

class BuilderNode : public runtime::Object {
 public:
  runtime::Array<Frame> frames;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("frames", &frames);  //
  }

  static constexpr const char* _type_key = "script.builder.Builder";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuilderNode, runtime::Object);

 public:
  template <typename TFrame>
  TFrame FindFrame() const {
    using TFrameNode = typename TFrame::ContainerType;
    for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
      if (const TFrameNode* p = (*it).template as<TFrameNode>()) {
        return GetRef<TFrame>(p);
      }
    }
    LOG(FATAL) << "IndexError: Cannot find frame: " << TFrameNode::_type_key;
    throw;
  }
};

class Builder : public runtime::ObjectRef {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Builder, ObjectRef, BuilderNode);

 public:
  void EnterWithScope();

  void ExitWithScope();

  static Builder Current();
};

}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_BUILDER_H_
