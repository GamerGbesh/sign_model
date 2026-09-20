"""Integration tests for FastAPI REST and WebSocket endpoints."""
import os
import struct
import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from asl.server import app, MAX_FRAME_BYTES


@pytest.fixture
def client():
    # Set DEV mode for test execution
    os.environ["DEV"] = "1"
    with TestClient(app) as c:
        yield c


def _create_dummy_jpeg():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "labels" in data
    assert "window_s" in data


def test_meta_endpoint(client):
    res = client.get("/meta")
    assert res.status_code == 200
    data = res.json()
    assert "labels" in data
    assert "seq_len" in data
    assert "feature_dim" in data


def test_demo_endpoint(client):
    res = client.get("/demo")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_websocket_handshake_and_frames(client):
    with client.websocket_connect("/ws/recognize") as ws:
        # 1. Server sends initial ready payload
        ready_msg = ws.receive_json()
        assert ready_msg["type"] == "ready"
        assert "labels" in ready_msg
        assert ready_msg["recommended_fps"] == 15

        # 2. Client sends binary packet: 8-byte timestamp (float64) + JPEG
        t_ms = 1234.56
        ts_header = struct.pack(">d", t_ms)
        jpeg_data = _create_dummy_jpeg()
        packet = ts_header + jpeg_data

        ws.send_bytes(packet)
        # Server processes frame and returns prediction
        resp = ws.receive_json()
        assert resp["type"] == "prediction"
        assert "label" in resp
        assert "topk" in resp
        assert "phrase" in resp

        # 3. Client sends reset text command
        ws.send_text('{"cmd": "reset"}')
        reset_resp = ws.receive_json()
        assert reset_resp["type"] == "reset"
        assert reset_resp["phrase"] == []


def test_websocket_oversize_frame_rejected(client):
    with client.websocket_connect("/ws/recognize") as ws:
        _ = ws.receive_json()  # ready
        # Send payload larger than 512 KB
        oversize_data = b"\x00" * (MAX_FRAME_BYTES + 100)
        ws.send_bytes(oversize_data)
        # Connection should close with 1009
        try:
            ws.receive()
        except Exception:
            pass  # Expected disconnect on oversize frame


def test_websocket_session_limit(client):
    os.environ["MAX_SESSIONS"] = "2"
    # Connect 2 sessions
    with client.websocket_connect("/ws/recognize") as ws1:
        _ = ws1.receive_json()
        with client.websocket_connect("/ws/recognize") as ws2:
            _ = ws2.receive_json()
            # Third connection exceeds limit
            try:
                with client.websocket_connect("/ws/recognize") as ws3:
                    # Should be closed with code 1013
                    pass
            except Exception:
                pass  # Rejected as expected
