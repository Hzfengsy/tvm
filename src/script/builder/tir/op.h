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

PrimExpr int8(PrimExpr expr) { return cast(DataType::Int(8), expr); }
PrimExpr int16(PrimExpr expr) { return cast(DataType::Int(16), expr); }
PrimExpr int32(PrimExpr expr) { return cast(DataType::Int(32), expr); }
PrimExpr int64(PrimExpr expr) { return cast(DataType::Int(64), expr); }

PrimExpr uint8(PrimExpr expr) { return cast(DataType::UInt(8), expr); }
PrimExpr uint16(PrimExpr expr) { return cast(DataType::UInt(16), expr); }
PrimExpr uint32(PrimExpr expr) { return cast(DataType::UInt(32), expr); }
PrimExpr uint64(PrimExpr expr) { return cast(DataType::UInt(64), expr); }

PrimExpr float8(PrimExpr expr) { return cast(DataType::Float(8), expr); }
PrimExpr float16(PrimExpr expr) { return cast(DataType::Float(16), expr); }
PrimExpr float32(PrimExpr expr) { return cast(DataType::Float(32), expr); }
PrimExpr float64(PrimExpr expr) { return cast(DataType::Float(64), expr); }

PrimExpr bool_(PrimExpr expr) { return cast(DataType::Bool(), expr); }

PrimExpr prim_type(String type_name, PrimExpr expr);

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
