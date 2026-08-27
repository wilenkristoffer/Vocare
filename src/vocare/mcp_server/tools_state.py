from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

_ALLOWED_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculationError(ValueError):
    pass


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculationError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise CalculationError(f"unsupported expression element: {ast.dump(node)}")


def calculate(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression (+ - * / ** % //, parens).

    Deliberately does NOT use eval() - parses to an AST and only walks a small
    allow-listed set of numeric operations, so it can't execute arbitrary code
    even if a prompt tries to smuggle something through the model.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"could not parse expression: {exc}") from exc
    return _eval_node(tree.body)


def get_current_time(timezone: str = "UTC") -> str:
    try:
        tz = ZoneInfo(timezone)
    except Exception as exc:
        raise ValueError(f"unknown timezone: {timezone!r}") from exc
    return datetime.now(tz).isoformat()


# Mock device registry - stands in for "real hardware/CRM integration" so the
# agent has a tool with side effects to call, without pretending to talk to
# actual lab/pharmacy robotics hardware. In-memory only: state resets each run.
_DEVICES: dict[str, dict[str, str]] = {
    "autodose-01": {"status": "idle", "location": "Pharmacy A - Bay 1"},
    "autodose-02": {"status": "dispensing", "location": "Pharmacy A - Bay 2"},
    "autodose-03": {"status": "paused", "location": "Pharmacy B - Bay 1"},
}

_VALID_ACTIONS = {"pause", "resume"}
_VALID_STATUSES = {"idle", "dispensing", "paused", "error"}


def device_status(device_id: str) -> dict[str, str]:
    if device_id not in _DEVICES:
        raise KeyError(f"unknown device_id: {device_id!r}. Known devices: {sorted(_DEVICES)}")
    return {"device_id": device_id, **_DEVICES[device_id]}


def device_control(device_id: str, action: str) -> dict[str, str]:
    if device_id not in _DEVICES:
        raise KeyError(f"unknown device_id: {device_id!r}. Known devices: {sorted(_DEVICES)}")
    if action not in _VALID_ACTIONS:
        raise ValueError(f"unsupported action: {action!r}. Valid actions: {sorted(_VALID_ACTIONS)}")
    new_status = "paused" if action == "pause" else "idle"
    _DEVICES[device_id]["status"] = new_status
    return {"device_id": device_id, **_DEVICES[device_id]}


def list_devices() -> list[dict[str, str]]:
    return [{"device_id": device_id, **state} for device_id, state in _DEVICES.items()]
