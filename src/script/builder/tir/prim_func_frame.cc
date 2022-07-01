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

#include "../ir/ir.h"
#include "./block_frame.h"
#include "./var.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

PrimFuncFrame FindPrimFuncFrame(const String& method) {
  Builder builder = Builder::Current();
  if (Optional<PrimFuncFrame> prim_func_frame = builder->FindFrame<PrimFuncFrame>()) {
    if (Optional<BlockFrame> block_frame = builder->GetLastFrame<BlockFrame>()) {
      if (prim_func_frame.value()->root_block_frame.get() == block_frame.get()) {
        return prim_func_frame.value();
      }
    }
  } else {
    LOG(FATAL) << "ValueError: PrimFunc frame not find. Please ensure '" << method
               << "' is called under T.prim_func()";
  }
  LOG(FATAL) << "ValueError: '" << method << "' must be called immediately under T.prim_func()";
  throw;
}

void PrimFuncFrameNode::EnterWithScope() {
  TIRFrameNode::EnterWithScope();
  // add implicit root block
  root_block_frame->EnterWithScope();
}

void PrimFuncFrameNode::ExitWithScope() {
  using ir::IRModuleFrame;
  root_block_frame->ExitWithScope();
  TIRFrameNode::ExitWithScope();
  Builder builder = Builder::Current();
  if (!(stmts.size() == 1 && stmts[0]->IsInstance<tvm::tir::BlockRealizeNode>())) {
    LOG(FATAL) << "ValueError: PrimFuncFrame shoulde have one and only one root block.";
  }
  tvm::tir::BlockRealize root_block_realize = Downcast<tvm::tir::BlockRealize>(stmts[0]);
  tvm::tir::Block root_block = root_block_realize->block;
  // remove redundant implicit root block
  if (root_block->alloc_buffers.empty() &&
      root_block->body->IsInstance<tvm::tir::BlockRealizeNode>() &&
      root_block->annotations.empty() && root_block->reads.empty() && root_block->writes.empty()) {
    stmts.Set(0, root_block->body);
  }
  Map<tvm::tir::Var, tvm::tir::Buffer> tir_buffer_map;
  Map<tvm::tir::Var, tvm::tir::Buffer> tir_preflattened_buffer_map;
  for (auto const& p : buffer_map) {
    tir_buffer_map.Set(p.first, p.second->buffer);
  }
  for (auto const& p : preflattened_buffer_map) {
    tir_preflattened_buffer_map.Set(p.first, p.second->buffer);
  }
  tvm::tir::PrimFunc func(/*params=*/args,
                          /*body=*/AsStmt(stmts),
                          /*ret_type=*/ret_type.value_or(TupleType::Empty()),
                          /*buffer_map=*/tir_buffer_map,
                          /*preflattened_buffer_map=*/tir_preflattened_buffer_map,
                          /*attrs=*/DictAttrs(attrs));
  if (builder->frames.empty()) {
    ICHECK(!builder->result.defined()) << "ValueError: Builder.result has already been set";
    builder->result = func;
  } else if (Optional<IRModuleFrame> opt_frame = builder->FindFrame<IRModuleFrame>()) {
    IRModuleFrame frame = opt_frame.value();
    frame->global_vars.push_back(GlobalVar(name.value_or("")));
    frame->functions.push_back(func);
  } else {
    LOG(FATAL) << "ValueError: Cannot find where to insert PrimFunc";
  }
}

PrimFuncFrame PrimFunc() {
  ObjectPtr<PrimFuncFrameNode> n = make_object<PrimFuncFrameNode>();
  n->name = NullOpt;
  n->args.clear();
  n->ret_type = NullOpt;
  n->buffer_map.clear();
  n->preflattened_buffer_map.clear();
  n->attrs.clear();
  n->root_block_frame = Block("root");
  return PrimFuncFrame(n);
}

tvm::tir::Var Arg(String name, tvm::tir::Var var) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.Arg");
  Namer::Name(var, name);
  frame->args.push_back(var);
  return var;
}

Buffer Arg(String name, Buffer buffer) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.Arg");
  Namer::Name(buffer, name);
  tvm::tir::Var handle(buffer->buffer->name + "_handle", DataType::Handle());
  frame->args.push_back(handle);
  frame->buffer_map.Set(handle, buffer);
  return buffer;
}

void FuncName(String name) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.func_name");
  if (frame->name.defined()) {
    LOG(FATAL) << "ValueError: Duplicate prim func name, previous one is " << frame->name.value();
  }
  frame->name = name;
}

void FuncAttrs(Map<String, ObjectRef> attrs) {
  using namespace tvm::tir;
  PrimFuncFrame frame = FindPrimFuncFrame("T.func_attr");
  if (!frame->attrs.empty()) {
    LOG(FATAL) << "ValueError: Duplicate prim func annotations, previous one is " << frame->attrs;
  }
  frame->attrs = attrs;
}

tvm::Type FuncRet(tvm::Type ret_type) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.ret_type");
  if (frame->ret_type.defined()) {
    LOG(FATAL) << "ValueError: Duplicate prim func return type, previous one is "
               << frame->ret_type.value();
  }
  frame->ret_type = ret_type;
  return ret_type;
}

Buffer MatchBuffer(ObjectRef param, Array<PrimExpr> shape, DataType dtype,
                   Optional<tvm::tir::Var> data, Array<PrimExpr> strides, PrimExpr elem_offset,
                   String storage_scope, int align, int offset_factor, String buffer_type_str,
                   Array<IntImm> axis_separators) {
  Buffer buffer(shape, dtype, "", data, strides, elem_offset, storage_scope, align, offset_factor,
                buffer_type_str, axis_separators);
  if (const auto* var = param.as<tvm::tir::VarNode>()) {
    PrimFuncFrame frame = FindPrimFuncFrame("T.match_buffer");
    tvm::tir::Var v = GetRef<tvm::tir::Var>(var);
    for (auto const& arg : frame->args) {
      if (arg.same_as(v)) {
        frame->buffer_map.Set(v, buffer);
        return buffer;
      }
    }
    LOG(FATAL) << "ValueError: Can not bind non-input param to buffer.";
  } else if (const auto* buffer_region = param.as<tvm::tir::BufferRegionNode>()) {
    BlockFrame frame = FindBlockFrame("T.match_buffer");
    frame->match_buffers.push_back(
        tvm::tir::MatchBufferRegion(buffer->buffer, GetRef<tvm::tir::BufferRegion>(buffer_region)));
  } else {
    LOG(FATAL) << "ValueError: Unexpected type for TIR MatchBuffer.";
  }
  return buffer;
};

void PreflattenedBuffer(Buffer postflattened_buffer, Array<PrimExpr> shape, DataType dtype,
                        Optional<tvm::tir::Var> data, Array<PrimExpr> strides, PrimExpr elem_offset,
                        String storage_scope, int align, int offset_factor, String buffer_type_str,
                        Array<IntImm> axis_separators) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.preflattened_buffer");
  for (auto const& p : frame->buffer_map) {
    if (p.second.same_as(postflattened_buffer)) {
      String buffer_name(postflattened_buffer->buffer->name + "_preflatten");
      Buffer buffer(shape, dtype, buffer_name, data, strides, elem_offset, storage_scope, align,
                    offset_factor, buffer_type_str, axis_separators);
      Namer::Name(buffer, buffer_name);
      frame->preflattened_buffer_map.Set(p.first, buffer);
      return;
    }
  }
  LOG(FATAL) << "ValueError: postflattened buffer " << postflattened_buffer->buffer->name
             << " does not exist.";
};

TVM_REGISTER_NODE_TYPE(PrimFuncFrameNode);
TVM_REGISTER_GLOBAL("script.builder.tir.PrimFunc").set_body_typed(PrimFunc);
TVM_REGISTER_GLOBAL("script.builder.tir.Arg")
    .set_body_typed([](String name, ObjectRef obj) -> ObjectRef {
      using namespace tvm::tir;
      if (const auto* var = obj.as<VarNode>()) {
        return Arg(name, GetRef<tvm::tir::Var>(var));
      }
      if (const auto* buffer = obj.as<BufferNode>()) {
        return Arg(name, GetRef<Buffer>(buffer));
      }
      LOG(FATAL) << "ValueError: Unexpected type for TIR Arg: " << obj->GetTypeKey();
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
