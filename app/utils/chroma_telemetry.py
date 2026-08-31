"""No-op Chroma telemetry implementation for local/private deployments."""
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpTelemetry(ProductTelemetryClient):
    """Accept telemetry events without sending them anywhere."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None
