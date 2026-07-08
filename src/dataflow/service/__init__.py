"""Engine service: persistent daemon (server), object store, wire
protocol, and the client library. Design + API reference:
docs/notes/engine_service_design.md, docs/notes/server_engine_api_s1.md."""

from .client import DEFAULT_SOCKET, EngineClient
from .server import EngineConfig, Server
from .wire import SCHEMA_VERSION, ServiceError

__all__ = ["EngineClient", "EngineConfig", "Server", "ServiceError",
           "SCHEMA_VERSION", "DEFAULT_SOCKET"]
