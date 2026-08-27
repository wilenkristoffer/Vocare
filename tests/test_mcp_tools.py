from __future__ import annotations

import pytest

from vocare.mcp_server import tools_state


def test_calculate_basic_arithmetic() -> None:
    assert tools_state.calculate("2 + 3 * 4") == 14
    assert tools_state.calculate("(2 + 3) * 4") == 20
    assert tools_state.calculate("2 ** 8") == 256
    assert tools_state.calculate("-5 + 2") == -3


def test_calculate_rejects_non_arithmetic() -> None:
    with pytest.raises(tools_state.CalculationError):
        tools_state.calculate("__import__('os').system('echo hi')")
    with pytest.raises(tools_state.CalculationError):
        tools_state.calculate("open('f.txt')")


def test_calculate_rejects_bad_syntax() -> None:
    with pytest.raises(tools_state.CalculationError):
        tools_state.calculate("2 + ")


def test_get_current_time_valid_timezone() -> None:
    result = tools_state.get_current_time("UTC")
    assert "T" in result  # ISO format


def test_get_current_time_unknown_timezone() -> None:
    with pytest.raises(ValueError):
        tools_state.get_current_time("Not/A_Real_Zone")


def test_device_status_known_device() -> None:
    status = tools_state.device_status("autodose-01")
    assert status["device_id"] == "autodose-01"
    assert "status" in status


def test_device_status_unknown_device() -> None:
    with pytest.raises(KeyError):
        tools_state.device_status("does-not-exist")


def test_device_control_pause_and_resume() -> None:
    paused = tools_state.device_control("autodose-01", "pause")
    assert paused["status"] == "paused"
    resumed = tools_state.device_control("autodose-01", "resume")
    assert resumed["status"] == "idle"


def test_device_control_invalid_action() -> None:
    with pytest.raises(ValueError):
        tools_state.device_control("autodose-01", "explode")


def test_list_devices_returns_all() -> None:
    devices = tools_state.list_devices()
    assert len(devices) == 3
    assert {d["device_id"] for d in devices} == {"autodose-01", "autodose-02", "autodose-03"}
