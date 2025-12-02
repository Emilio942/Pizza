"""Unit tests for SD card logging in the RP2040 emulator."""

from pathlib import Path
from typing import List

from src.emulation.emulator import RP2040Emulator
from src.emulation.logging_system import LogLevel, LogType


def _collect_lines(path: Path) -> List[str]:
    return path.read_text().strip().splitlines()


def test_sd_card_records_performance_metrics(tmp_path: Path) -> None:
    sd_root = tmp_path / "sd"
    log_dir = tmp_path / "logs"

    emulator = RP2040Emulator(sd_root_dir=sd_root, log_dir=log_dir)

    try:
        emulator.log_performance_metrics(
            inference_time_ms=12.5,
            peak_ram_kb=48.0,
            cpu_load=55.0,
            prediction=1,
            confidence=0.87,
        )
        emulator.log_performance_metrics(
            inference_time_ms=15.2,
            peak_ram_kb=52.0,
            cpu_load=62.0,
            prediction=0,
            confidence=0.34,
        )
        emulator.log_temperature()
    finally:
        emulator.close()

    sd_logs = sd_root / "logs"
    metrics_files = sorted(sd_logs.glob("performance_metrics_*.csv"))
    assert metrics_files, "Performance metrics file missing on SD card"

    for metrics_file in metrics_files:
        lines = _collect_lines(metrics_file)
        assert lines[0].startswith("Timestamp,"), "CSV header missing in metrics file"
        assert len(lines) >= 2, "Metrics file should contain at least one data row"

    aggregated_logs = sorted(sd_logs.glob("performance_log_*.csv"))
    assert aggregated_logs, "Aggregated performance log not written to SD card"
    assert len(_collect_lines(aggregated_logs[0])) >= 2


def test_sd_card_records_system_logs(tmp_path: Path) -> None:
    sd_root = tmp_path / "sd"
    log_dir = tmp_path / "logs"

    emulator = RP2040Emulator(sd_root_dir=sd_root, log_dir=log_dir)

    try:
        emulator.logging_system.log("SD card logging functional", LogLevel.INFO, LogType.SYSTEM)
    finally:
        emulator.close()

    sd_logs = sd_root / "logs"
    system_logs = sorted(sd_logs.glob("system_log_*.log"))
    assert system_logs, "System log file not found on SD card"

    content = system_logs[0].read_text()
    assert "SD card logging functional" in content
