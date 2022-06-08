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
from typing import Any, Dict, Optional, Union


def eval_expr(
    node: Union[ast.expr, ast.Expression],
    dict_globals: Optional[Dict[str, Any]],
) -> Any:
    if isinstance(node, ast.expr):
        node = ast.Expression(body=node)
    assert isinstance(node, ast.Expression)
    if dict_globals is None:
        dict_globals = {}
    node = ast.fix_missing_locations(node)
    exe = compile(node, filename="<ast>", mode="eval")
    return eval(exe, dict_globals)  # pylint: disable=eval-used


def eval_assign(
    target: ast.expr,
    source: Any,
) -> Dict[str, Any]:
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
