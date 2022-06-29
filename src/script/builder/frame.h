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
#ifndef TVM_SCRIPT_BUILDER_FRAME_H_
#define TVM_SCRIPT_BUILDER_FRAME_H_

#include <tvm/ir/module.h>
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

  static constexpr const char* _type_key = "script.builder.Frame";
  TVM_DECLARE_BASE_OBJECT_INFO(FrameNode, runtime::Object);

 public:
  virtual ~FrameNode() = default;
  virtual void EnterWithScope();
  virtual void ExitWithScope();

  void AddCallback(runtime::TypedPackedFunc<void()> callback);
};

class Frame : public runtime::ObjectRef {
 public:
  virtual ~Frame() = default;
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Frame, ObjectRef, FrameNode);

 protected:
  Frame() = default;

 public:
  inline void EnterWithScope();
  inline void ExitWithScope();
};

inline void Frame::EnterWithScope() {
  ICHECK(data_ != nullptr);
  static_cast<FrameNode*>(data_.get())->EnterWithScope();
}

inline void Frame::ExitWithScope() {
  ICHECK(data_ != nullptr);
  static_cast<FrameNode*>(data_.get())->ExitWithScope();
  data_.reset();
}

}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_FRAME_H_
