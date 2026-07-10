"""Peer plane: daemon-to-daemon object transfer + (later) group comm.

Layering: ``protocol`` defines the wire vocabulary; ``core`` holds the
rendezvous STATE MACHINES as deterministic, thread-free logic driven by
explicit message delivery and a caller-owned clock (unit-testable
against the in-memory transport, including every fault path);
``transports`` move payload bytes. The threaded NetworkManager that
pumps these cores against real sockets arrives with the loopback-daemon
work; nothing in this package touches the catalog — commits are the
embedder's job through its dispatcher.
"""
from .core import PeerCore, ReceiverEnv, SenderTicket
from .protocol import NackCode, TransferState

__all__ = ["PeerCore", "ReceiverEnv", "SenderTicket", "NackCode",
           "TransferState"]
