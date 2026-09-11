import json
import socket
import struct
from concurrent.futures import ThreadPoolExecutor

import pytest

from agenticrobot_bridge.physx_client import PhysXClient, recv_message, send_message


def test_fragmented_message_and_roundtrip():
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(1) as executor:
        payload = json.dumps({"op": "hello"}).encode()
        def write():
            for byte in struct.pack("<I", len(payload)) + payload:
                left.sendall(bytes([byte]))
        pending = executor.submit(write)
        assert recv_message(right) == {"op": "hello"}
        pending.result()
        send_message(right, {"engine": "PhysX"})
        assert recv_message(left) == {"engine": "PhysX"}


def test_rejects_oversized_and_disconnected_frames():
    left, right = socket.socketpair()
    with left, right:
        left.sendall(struct.pack("<I", 50_000_000))
        with pytest.raises(ValueError, match="size"):
            recv_message(right)
    left, right = socket.socketpair()
    left.close()
    with right, pytest.raises(ConnectionError):
        recv_message(right)


def test_client_refuses_non_loopback_target():
    with pytest.raises(ValueError, match="loopback"):
        PhysXClient(host="0.0.0.0")
