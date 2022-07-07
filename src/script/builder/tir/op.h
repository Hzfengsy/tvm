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
#ifndef TVM_SCRIPT_BUILDER_TIR_OP_H_
#define TVM_SCRIPT_BUILDER_TIR_OP_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include "../builder.h"

namespace tvm {
namespace script {
namespace builder {
namespace tir {

inline PrimExpr Int8(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::Int(8);
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr Int16(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::Int(16);
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr Int32(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::Int(32);
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr Int64(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::Int(64);
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr UInt8(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::UInt(8);
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr UInt16(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::UInt(16);
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr UInt32(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::UInt(32);
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr UInt64(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::UInt(64);
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr Float8(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::Float(8);
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr Float16(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::Float(16);
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr Float32(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::Float(32);
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr Float64(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::Float(64);
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr Boolean(Optional<PrimExpr> expr = NullOpt) {
  DataType dtype = DataType::Bool();
  return expr.defined() ? tvm::cast(dtype, expr.value()) : tvm::tir::Var("", dtype);
}

inline PrimExpr Ptr(Type type, String storate_scope = "global") {
  return tvm::tir::Var("", tvm::PointerType(type, storate_scope));
}

inline tvm::tir::Var Handle() { return tvm::tir::Var("", DataType::Handle()); }

inline PrimExpr PrimType(DataType dtype, PrimExpr expr) { return tvm::cast(dtype, expr); }

using tvm::cast;
using tvm::if_then_else;
using tvm::infinity;
using tvm::max;
using tvm::max_value;
using tvm::min;
using tvm::min_value;
using tvm::reinterpret;

using tvm::ceil;
using tvm::floor;
using tvm::floordiv;
using tvm::floormod;
using tvm::nearbyint;
using tvm::round;
using tvm::trunc;
using tvm::truncdiv;
using tvm::truncmod;

using tvm::abs;
using tvm::copysign;
using tvm::fmod;
using tvm::nextafter;
using tvm::popcount;

using tvm::erf;
using tvm::exp;
using tvm::exp10;
using tvm::exp2;
using tvm::hypot;
using tvm::ldexp;
using tvm::log;
using tvm::log10;
using tvm::log1p;
using tvm::log2;
using tvm::pow;
using tvm::rsqrt;
using tvm::sigmoid;
using tvm::sqrt;

using tvm::acos;
using tvm::acosh;
using tvm::asin;
using tvm::asinh;
using tvm::atan;
using tvm::atan2;
using tvm::atanh;
using tvm::clz;
using tvm::cos;
using tvm::cosh;
using tvm::sin;
using tvm::sinh;
using tvm::tan;
using tvm::tanh;

using tvm::isfinite;
using tvm::isinf;
using tvm::isnan;

using tvm::tir::Broadcast;
using tvm::tir::CommReducer;
using tvm::tir::Ramp;
using tvm::tir::Select;
using tvm::tir::Shuffle;

}  // namespace tir
}  // namespace builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_TIR_OP_H_
