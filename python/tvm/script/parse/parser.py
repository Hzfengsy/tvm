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
"""The core parser"""
from typing import Any, Dict, List, Optional, Union

from ..builder import def_
from . import dispatch, doc
from .evaluator import eval_assign, eval_expr
from .utils import deferred
from .var_table import VarTable


def _dispatch(self: "Parser", type_name: str) -> dispatch.ParseMethod:
    for token in [self.dispatch_tokens[-1], "default"]:
        func = dispatch.get(token=token, type_name=type_name, default=None)
        if func is not None:
            return func
    return lambda self, node: self.generic_visit(node)


class Parser(doc.NodeVisitor):
    """The TVMScript parser"""

    dispatch_tokens: List[str]
    var_table: VarTable

    def __init__(self) -> None:
        self.dispatch_tokens = ["default"]
        self.var_table = VarTable()

    def with_dispatch_token(self, token: str):
        def pop_token():
            self.dispatch_tokens.pop()

        self.dispatch_tokens.append(token)
        return deferred(pop_token)

    def eval_expr(
        self,
        node: Union[doc.Expression, doc.expr],
        extra_vars: Optional[Dict[str, Any]] = None,
    ) -> Any:
        var_values = self.var_table.get()
        if extra_vars is not None:
            for k, v in extra_vars.items():
                var_values[k] = v
        return eval_expr(node, var_values)

    def eval_assign(
        self,
        target: doc.expr,
        source: Any,
    ) -> Dict[str, Any]:
        var_values = eval_assign(target, source)
        for k, v in var_values.items():
            def_(k, v)
            self.var_table.add(k, v)
        return var_values

    def report_error(self, node: doc.AST, msg: str) -> None:  # pylint: disable=no-self-use
        raise SyntaxError(f"At {node.lineno}:{node.col_offset}: {msg}")

    def visit_FunctionDef(self, node: doc.FunctionDef) -> Any:  # pylint: disable=invalid-name
        _handle_function(self, node)

    def visit_ClassDef(self, node: doc.ClassDef) -> Any:  # pylint: disable=invalid-name
        _handle_class(self, node)

    def visit_body(self, node: List[doc.stmt]) -> Any:
        for stmt in node:
            self.visit(stmt)

    def visit_arguments(self, node: doc.arguments) -> Any:
        _dispatch(self, "arguments")(self, node)

    def visit_For(self, node: doc.For) -> Any:  # pylint: disable=invalid-name
        _dispatch(self, "For")(self, node)

    def visit_With(self, node: doc.With) -> Any:  # pylint: disable=invalid-name
        _dispatch(self, "With")(self, node)

    def visit_Assign(self, node: doc.Assign) -> Any:  # pylint: disable=invalid-name
        _dispatch(self, "Assign")(self, node)


def _handle_function(self: Parser, node: doc.FunctionDef) -> None:
    if not node.decorator_list:
        self.report_error(node, "Function must be decorated")
    # TODO: only the last decorator is parsed
    decorator = self.eval_expr(node.decorator_list[-1])
    if hasattr(decorator, "dispatch_token"):
        token = decorator.dispatch_token
        func = dispatch.get(token=token, type_name="FunctionDef", default=None)
        if func is not None:
            func(self, node)
            return
    self.report_error(node, "The parser does not understand the decorator")


def _handle_class(self: Parser, node: doc.ClassDef) -> None:
    # TODO: assume IRModule
    func = dispatch.get(token="ir", type_name="ClassDef", default=None)
    if func is not None:
        func(self, node)
        return
    self.report_error(node, "The parser does not understand the decorator")
