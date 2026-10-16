"""Mocked CUDA probe contracts; these tests do NOT claim real GPU execution."""

from __future__ import annotations

from types import SimpleNamespace

from cds2 import gpu


def _fake_cupy(device_count: int):
    return SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: device_count))
    )


def test_importable_backend_with_no_devices_is_not_usable(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "_ensure_cupy", lambda: _fake_cupy(0))
    assert gpu.is_usable() is False


def test_runtime_reports_devices_only_after_probing(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "_ensure_cupy", lambda: _fake_cupy(2))
    assert gpu.is_usable() is True


def test_cuda_driver_error_does_not_claim_gpu(monkeypatch) -> None:
    def broken_cuda():
        raise RuntimeError("CUDA driver unavailable")

    monkeypatch.setattr(gpu, "_ensure_cupy", broken_cuda)
    assert gpu.is_usable() is False


def test_cuda_query_error_does_not_claim_gpu(monkeypatch) -> None:
    def broken_query():
        raise OSError("CUDA runtime failed")

    fake = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=broken_query))
    )
    monkeypatch.setattr(gpu, "_ensure_cupy", lambda: fake)
    assert gpu.is_usable() is False
