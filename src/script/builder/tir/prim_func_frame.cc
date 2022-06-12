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

#include "./prim_func_frame.h"

#include <tvm/tir/function.h>

#include "./block_frame.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

void PrimFuncFrameNode::ExitWithScope() {
  using namespace tvm::tir;
  TIRFrameNode::ExitWithScope();
  Builder builder = Builder::Current();
  PrimFunc func(/*params=*/args,
                /*body=*/AsStmt(stmts),
                /*ret_type=*/ret_type,
                /*buffer_map=*/buffer_map,
                /*preflattened_buffer_map=*/preflattened_buffer_map,
                /*attrs=*/DictAttrs(attrs));
  if (builder->frames.empty()) {
    ICHECK(!builder->result.defined()) << "ValueError: Builder.result has already been set";
    builder->result = func;
  } else if (Optional<IRModuleFrame> opt_frame = builder->FindFrame<IRModuleFrame>()) {
    IRModuleFrame frame = opt_frame.value();
    frame->global_vars.push_back(GlobalVar(name));
    frame->functions.push_back(func);
  } else {
    LOG(FATAL) << "ValueError: Cannot find where to insert PrimFunc";
  }
}

PrimFuncFrame PrimFunc_() {
  ObjectPtr<PrimFuncFrameNode> n = make_object<PrimFuncFrameNode>();
  n->name = "";
  n->args.clear();
  n->ret_type = TupleType::Empty();
  n->buffer_map.clear();
  n->preflattened_buffer_map.clear();
  n->attrs.clear();
  return PrimFuncFrame(n);
}

tvm::tir::Var Arg(String name, tvm::tir::Var var) {
  Namer::Name(var, name);
  PrimFuncFrame frame = Builder::Current()->FindFrame<PrimFuncFrame>().value();
  frame->args.push_back(var);
  return var;
}

tvm::tir::Buffer Arg(String name, tvm::tir::Buffer buffer) {
  using namespace tvm::tir;
  Namer::Name(buffer, name);
  PrimFuncFrame frame = Builder::Current()->FindFrame<PrimFuncFrame>().value();
  Var handle(buffer->name + "_handle", DataType::Handle());
  frame->args.push_back(handle);
  frame->buffer_map.Set(handle, buffer);
  return buffer;
}

void FuncName(String name) {
  PrimFuncFrame frame = Builder::Current()->FindFrame<PrimFuncFrame>().value();
  frame->name = name;
}

void FuncAttrs(Map<String, ObjectRef> attrs) {
  using namespace tvm::tir;
  PrimFuncFrame frame = Builder::Current()->FindFrame<PrimFuncFrame>().value();
  frame->attrs = attrs;
}

tvm::Type FuncRet(tvm::Type ret_type) {
  PrimFuncFrame frame = Builder::Current()->FindFrame<PrimFuncFrame>().value();
  frame->ret_type = ret_type;
  return ret_type;
}

tvm::tir::Buffer MatchBuffer(ObjectRef param, Array<PrimExpr> shape, DataType dtype,
                             Optional<tvm::tir::Var> data, Array<PrimExpr> strides,
                             PrimExpr elem_offset, String storage_scope, int align,
                             int offset_factor, String buffer_type_str,
                             Array<IntImm> axis_separators, Span span) {
  using namespace tvm::tir;
  Var buffer_data;
  if (!data.defined()) {
    DataType storage_dtype = dtype;
    if (storage_dtype == DataType::Bool()) {
      storage_dtype = DataType::Int(8);
    }
    buffer_data = Var("", PointerType(PrimType(storage_dtype), storage_scope), span);
  } else {
    buffer_data = data.value();
  }
  BufferType buffer_type = (buffer_type_str == "auto_broadcast") ? kAutoBroadcast : kDefault;
  Buffer buffer(buffer_data, dtype, shape, strides, elem_offset, "", align, offset_factor,
                buffer_type, axis_separators, span);
  PrimFuncFrame frame = Builder::Current()->FindFrame<PrimFuncFrame>().value();
  if (const auto* var = param.as<VarNode>()) {
    Var v = GetRef<Var>(var);
    for (auto const& arg : frame->args) {
      if (arg.same_as(v)) {
        frame->buffer_map.Set(v, buffer);
        return buffer;
      }
    }
    LOG(FATAL) << "ValueError: Can not bind non-input param to buffer.";
  } else if (const auto* buffer_region = param.as<BufferRegionNode>()) {
    BlockFrame block_frame = Builder::Current()->FindFrame<BlockFrame>().value();
    block_frame->match_buffers.push_back(
        MatchBufferRegion(buffer, GetRef<BufferRegion>(buffer_region)));
  } else {
    LOG(FATAL) << "ValueError: Unexpected type for TIR MatchBuffer.";
  }
  return buffer;
};

void PreflattenedBuffer(tvm::tir::Buffer postflattened_buffer, Array<PrimExpr> shape,
                        DataType dtype, Optional<tvm::tir::Var> data, Array<PrimExpr> strides,
                        PrimExpr elem_offset, String storage_scope, int align, int offset_factor,
                        String buffer_type_str, Array<IntImm> axis_separators, Span span) {
  using namespace tvm::tir;
  PrimFuncFrame frame = Builder::Current()->FindFrame<PrimFuncFrame>().value();
  for (auto const& p : frame->buffer_map) {
    if (p.second.same_as(postflattened_buffer)) {
      Var buffer_data = (data.defined()) ? data.value() : frame->buffer_map.at(p.first)->data;
      String buffer_name(postflattened_buffer->name + "_preflatten");
      BufferType buffer_type = (buffer_type_str == "auto_broadcast") ? kAutoBroadcast : kDefault;
      Buffer buffer(buffer_data, dtype, shape, strides, elem_offset, buffer_name, align,
                    offset_factor, buffer_type, axis_separators, span);
      Namer::Name(buffer, buffer_name);
      frame->preflattened_buffer_map.Set(p.first, buffer);
      return;
    }
  }
  LOG(FATAL) << "ValueError: postflattened buffer " << postflattened_buffer->name
             << " does not exist.";
};

TVM_REGISTER_NODE_TYPE(PrimFuncFrameNode);
TVM_REGISTER_GLOBAL("script.builder.tir.PrimFuncFrame").set_body_typed(PrimFunc_);
TVM_REGISTER_GLOBAL("script.builder.tir.Arg")
    .set_body_typed([](String name, ObjectRef obj) -> ObjectRef {
      using namespace tvm::tir;
      if (const auto* var = obj.as<VarNode>()) {
        return Arg(name, GetRef<Var>(var));
      }
      if (const auto* buffer = obj.as<BufferNode>()) {
        return Arg(name, GetRef<Buffer>(buffer));
      }
      LOG(FATAL) << "ValueError: Unexpected type for TIR Arg.";
      throw;
    });
TVM_REGISTER_GLOBAL("script.builder.tir.FuncName").set_body_typed(FuncName);
TVM_REGISTER_GLOBAL("script.builder.tir.FuncAttrs").set_body_typed(FuncAttrs);
TVM_REGISTER_GLOBAL("script.builder.tir.FuncRet").set_body_typed(FuncRet);
TVM_REGISTER_GLOBAL("script.builder.tir.MatchBuffer").set_body_typed(MatchBuffer);
TVM_REGISTER_GLOBAL("script.builder.tir.PreflattenedBuffer").set_body_typed(PreflattenedBuffer);

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm
