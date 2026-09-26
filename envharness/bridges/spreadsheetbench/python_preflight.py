"""Cheap static checks for common SpreadsheetBench run_python mistakes."""
from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class PreflightResult:
    error: str = ""
    error_type: str = ""
    warnings: tuple[str, ...] = ()


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.formula_assignment = False
        self.bare_except = False
        self.loads_input_for_edit = False
        self.saves_workbook = False
        self.mutates_cells = False

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "formula":
                self.formula_assignment = True
            if isinstance(target, (ast.Subscript, ast.Attribute)):
                self.mutates_cells = True
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.bare_except = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name == "save_workbook":
            self.saves_workbook = True
        if name in {"load_workbook", "load_workbook_for_edit"} and node.args:
            first = node.args[0] if name == "load_workbook" else None
            if isinstance(first, ast.Name) and first.id == "input_path":
                self.loads_input_for_edit = True
        self.generic_visit(node)


def preflight_python(code: str) -> PreflightResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return PreflightResult(
            error=f"Python syntax error at line {exc.lineno}: {exc.msg}",
            error_type="SyntaxError",
        )
    visitor = _Visitor()
    visitor.visit(tree)
    if visitor.formula_assignment:
        return PreflightResult(
            error=(
                "openpyxl Cell objects have no .formula attribute; assign an "
                "Excel formula through cell.value = '=...' instead"
            ),
            error_type="PreflightError",
        )
    warnings: list[str] = []
    if visitor.bare_except:
        warnings.append("bare except may hide workbook or formula errors")
    if visitor.loads_input_for_edit:
        warnings.append("loading input_path may discard edits from earlier turns")
    if visitor.mutates_cells and not visitor.saves_workbook:
        warnings.append("code appears to mutate workbook objects without save_workbook(wb)")
    return PreflightResult(warnings=tuple(warnings))
