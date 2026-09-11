"""Bounded loopback protocol for the independent PhysX Player sampler."""

import json
import socket
import struct

MAX_MESSAGE_BYTES = 8 * 1024 * 1024


def _receive_exact(connection, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("PhysX Player closed the sampler connection")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_message(connection):
    size = struct.unpack("<I", _receive_exact(connection, 4))[0]
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("Sampler message size is outside the protocol limit")
    return json.loads(_receive_exact(connection, size))


def send_message(connection, message):
    encoded = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if not 0 < len(encoded) <= MAX_MESSAGE_BYTES:
        raise ValueError("Sampler message size is outside the protocol limit")
    connection.sendall(struct.pack("<I", len(encoded)) + encoded)


class PhysXClient:
    def __init__(self, host="127.0.0.1", port=62101, timeout=120):
        if host != "127.0.0.1":
            raise ValueError("Sampler connections must use explicit IPv4 loopback")
        self.connection = socket.create_connection((host, port), timeout=timeout)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.identity = self.request("hello")
        if self.identity.get("engine") != "PhysX":
            self.connection.close()
            raise RuntimeError("The target is not a PhysX sampler")

    def request(self, op, **fields):
        send_message(self.connection, {"op": op, **fields})
        result = recv_message(self.connection)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Unknown PhysX sampler failure"))
        return result

    def reset(self, slot, external=True, indices=None):
        return self.request("reset", slot=slot, external=external, indices=indices)["results"]

    def step(self, actions=None, *, record_physics_trace=False):
        packed = None if actions is None else [{"values": list(row)} for row in actions]
        return self.request("step", actions=packed, recordPhysicsTrace=record_physics_trace)["results"]

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
