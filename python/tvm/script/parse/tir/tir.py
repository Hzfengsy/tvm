# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import contextlib

from ...builder import Frame
from ...builder import tir as T
from .. import dispatch, doc
from ..parser import Parser

from functools import partial


@dispatch.register(token="tir", type_name="For")
def visit_for(self: Parser, node: doc.For) -> None:
    for_frame = self.eval_expr(node.iter)
    if not isinstance(for_frame, T.ForFrame):
        self.report_error(
            node.iter,
            "Expect the for loop to be one of the following: "
            "range, T.serial, T.grid, T.parallel, T.vectorized, T.unroll, T.thread_binding",
        )
    with self.var_table.with_frame():
        with for_frame as iters:
            self.eval_assign(target=node.target, source=iters)
            self.visit_body(node.body)


@dispatch.register(token="tir", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0]
    rhs = self.eval_expr(node.value)
    if isinstance(rhs, Frame):
        rhs.add_callback(partial(rhs.__exit__, None, None, None))
        res = rhs.__enter__()
        self.eval_assign(target=lhs, source=res)
    elif isinstance(lhs, doc.Subscript):
        if isinstance(lhs.slice, doc.Tuple):
            indices = []
            for index in lhs.slice.elts:
                indices.append(self.eval_expr(index))
        else:
            indices = [self.eval_expr(lhs.slice)]
        T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
    else:
        self.eval_assign(target=lhs, source=rhs)


@dispatch.register(token="tir", type_name="With")
def visit_with(self: Parser, node: doc.With) -> None:
    with contextlib.ExitStack() as stack:
        stack.enter_context(self.var_table.with_frame())
        for item in node.items:
            frame = self.eval_expr(item.context_expr)
            if not isinstance(frame, Frame):
                self.report_error(
                    item.context_expr, "Invalid context expression in the with-statement."
                )
            rhs = stack.enter_context(frame)
            if item.optional_vars is not None:
                self.eval_assign(
                    target=item.optional_vars,
                    source=rhs,
                )
        self.visit_body(node.body)


@dispatch.register(token="tir", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    with self.var_table.with_frame():
        self.var_table.add("range", T.serial)
        with T.prim_func():
            T.func_name(node.name)
            with self.with_dispatch_token("tir"):
                # TODO: define the GlobalVar, handle the return value
                self.visit(node.args)
                self.visit_body(node.body)


@dispatch.register(token="tir", type_name="arguments")
def visit_arguments(self: Parser, node: doc.arguments) -> None:
    # TODO: handle different types of arguments:
    # - vararg: arg | None
    # - kwonlyargs: list[arg]
    # - kw_defaults: list[expr | None]
    # - kwarg: arg | None
    # - defaults: list[expr]
    # - posonlyargs: list[arg]
    arg: doc.arg
    for arg in node.args:
        if arg.annotation is None:
            self.report_error(arg, "Type annotation is required for function parameters.")
        param = T.arg(arg.arg, self.visit_tvm_annotation(arg.annotation))
        self.var_table.add(arg.arg, param)


@dispatch.register(token="tir", type_name="tvm_annotation")
def visit_tvm_annotation(self: Parser, node: doc.expr):
    annotation = self.eval_expr(node)
    if callable(annotation):
        annotation = annotation()
    return annotation


@dispatch.register(token="tir", type_name="Expr")
def visit_expr_stmt(self: Parser, node: doc.Expr) -> None:
    res = self.eval_expr(node.value)
    if isinstance(res, Frame):
        res.add_callback(partial(res.__exit__, None, None, None))
        res.__enter__()
