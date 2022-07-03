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
"""AST Evaluation"""
import ast
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from . import doc

if TYPE_CHECKING:
    from .parser import Parser


class ExprEvaluator:

    parser: "Parser"
    value_table: Dict[str, Any]
    new_value_count: int

    def __init__(self, parser: "Parser", value_table: Dict[str, Any]) -> None:
        super().__init__()
        self.parser = parser
        self.value_table = value_table
        self.new_value_count = 0

    @staticmethod
    def eval(parser: "Parser", value_table: Dict[str, Any], node: doc.AST) -> Any:
        self = ExprEvaluator(parser, value_table)
        result = self._visit(node)
        if isinstance(result, doc.Name):
            if result.id not in self.value_table:
                self.parser.report_error(result, "Undefined variable: %s" % result.id)
            return self.value_table[result.id]
        if isinstance(result, doc.Constant):
            return result.value
        raise TypeError("Unexpected result type: %s" % type(result))

    def _add_intermediate_result(self, value: Any) -> doc.Name:
        name = f"__tvm_tmp_value_{self.new_value_count}"
        self.new_value_count += 1
        self.value_table[name] = value
        lineno = 0
        col_offset = 0
        return doc.Name(
            id=name,
            ctx=doc.Load(
                lineno=lineno,
                col_offset=col_offset,
                end_lineno=None,
                end_col_offset=None,
            ),
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=None,
            end_col_offset=None,
        )

    def _visit(self, node: doc.AST) -> Any:
        if isinstance(node, list):
            return [self._visit(n) for n in node]
        if isinstance(node, tuple):
            return tuple(self._visit(n) for n in node)
        assert isinstance(node, doc.AST)
        if isinstance(node, doc.Name):
            if node.id not in self.value_table:
                self.parser.report_error(node, "Undefined variable: %s" % node.id)
            return node
        if (not isinstance(node, doc.expr)) or isinstance(
            node,
            (
                doc.Constant,
                doc.expr_context,
                doc.operator,
                doc.boolop,
                doc.unaryop,
                doc.cmpop,
                doc.slice,
            ),
        ):
            return node
        fields = {}
        for field in node.__class__._FIELDS:  # pylint: disable=protected-access
            attr = getattr(node, field)
            if isinstance(attr, (doc.AST, tuple, list)):
                fields[field] = self._visit(attr)
            else:
                fields[field] = attr
        try:
            if isinstance(node, doc.BoolOp) and isinstance(fields["op"], doc.And):
                value = self._eval_binary(
                    fields["values"],
                    lhs_func_name="__tvm_logical_and__",
                    rhs_func_name="__tvm_r_logical_and__",
                    default_func=lambda lhs, rhs: lhs and rhs,
                )
            elif isinstance(node, doc.BoolOp) and isinstance(fields["op"], doc.Or):
                value = self._eval_binary(
                    fields["values"],
                    lhs_func_name="__tvm_logical_or__",
                    rhs_func_name="__tvm_r_logical_or__",
                    default_func=lambda lhs, rhs: lhs or rhs,
                )
            elif isinstance(node, doc.UnaryOp) and isinstance(fields["op"], doc.Not):
                value = self._eval_unary(
                    fields["operand"],
                    func_name="__tvm_logical_not__",
                    default_func=lambda v: not v,
                )
            else:
                value = _eval_expr(node.__class__(**fields), self.value_table)
        except Exception as e:
            self.parser.report_error(node, str(e))
        return self._add_intermediate_result(value)

    def _eval_unary(
        self,
        value: Any,
        func_name: str,
        default_func: Callable,
    ):
        value = _eval_expr(value, self.value_table)
        method = getattr(value, func_name, None)
        if method is not None:
            return method(value)
        return default_func(value)

    def _eval_binary(
        self,
        values: List[Any],
        lhs_func_name: str,
        rhs_func_name: str,
        default_func: Callable,
    ):
        assert len(values) > 0
        values = [_eval_expr(v, self.value_table) for v in values if v is not None]
        lhs = values[0]
        for rhs in values[1:]:
            method = getattr(lhs, lhs_func_name, None)
            if method is not None:
                lhs = method(rhs)
                continue
            method = getattr(rhs, rhs_func_name, None)
            if method is not None:
                lhs = method(lhs)
                continue
            lhs = default_func(lhs, rhs)
        return lhs


def eval_expr(
    parser: "Parser",
    node: Union[doc.expr, doc.Expression],
    dict_globals: Optional[Dict[str, Any]],
) -> Any:
    value_table = {}
    if dict_globals is not None:
        value_table.update(dict_globals)
    return ExprEvaluator.eval(parser, value_table, node)


def eval_assign(
    parser: "Parser",
    target: doc.expr,
    source: Any,
) -> Dict[str, Any]:
    try:
        return _eval_assign(target, source)
    except Exception as e:
        parser.report_error(target, "Failed to evaluate assignment: %s" % str(e))
        raise


def _eval_expr(
    node: Union[doc.expr, doc.Expression],
    dict_globals: Optional[Dict[str, Any]],
) -> Any:
    node = doc.from_doc(node)
    if isinstance(node, ast.expr):
        node = ast.Expression(body=node)
    assert isinstance(node, ast.Expression)
    if dict_globals is None:
        dict_globals = {}
    node = ast.fix_missing_locations(node)
    exe = compile(node, filename="<ast>", mode="eval")
    return eval(exe, dict_globals)  # pylint: disable=eval-used


def _eval_assign(
    target: doc.expr,
    source: Any,
) -> Dict[str, Any]:
    target = doc.from_doc(target)
    assert isinstance(target, ast.expr)
    RHS_VAR_NAME = "__tvm_rhs_var__"  # pylint: disable=invalid-name
    rhs_var_name = RHS_VAR_NAME
    dict_locals = {rhs_var_name: source}
    mod = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.Assign(
                    targets=[target],
                    value=ast.Name(
                        id=rhs_var_name,
                        ctx=ast.Load(),
                    ),
                )
            ],
            type_ignores=[],
        )
    )
    exe = compile(mod, filename="<ast>", mode="exec")
    exec(exe, {}, dict_locals)  # pylint: disable=exec-used
    del dict_locals[rhs_var_name]
    return dict_locals
