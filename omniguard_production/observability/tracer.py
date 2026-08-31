"""
tracer.py — Production Distributed Tracing & Ring Latency Telemetry.

Captures end-to-end trace context, span lifecycles, and sub-millisecond execution metrics.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TelemetrySpan:
    """Represents a discrete unit of execution within the RAG defense pipeline."""
    span_id: str = field(default_factory=lambda: f"span_{uuid.uuid4().hex[:8]}")
    name: str = "operation"
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"  # 'OK', 'ERROR', 'FLAGGED'

    def finish(self, status: str = "OK", **extra_attrs):
        self.end_time = time.perf_counter()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 3)
        self.status = status
        self.attributes.update(extra_attrs)

    def log_event(self, name: str, data: Optional[Dict[str, Any]] = None):
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "data": data or {}
        })


class PipelineTracer:
    """Manages request-scoped trace contexts and span collection."""

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
        self.spans: List[TelemetrySpan] = []
        self.start_time = time.perf_counter()
        self.end_time: Optional[float] = None

    def start_span(self, name: str, **attributes) -> TelemetrySpan:
        """Starts and registers a new telemetry span."""
        span = TelemetrySpan(name=name, attributes=attributes)
        self.spans.append(span)
        return span

    def finish_trace(self) -> Dict[str, Any]:
        """Finalizes the trace context and calculates total pipeline latency."""
        self.end_time = time.perf_counter()
        total_duration_ms = round((self.end_time - self.start_time) * 1000.0, 3)

        return {
            "trace_id": self.trace_id,
            "total_duration_ms": total_duration_ms,
            "span_count": len(self.spans),
            "spans": [
                {
                    "span_id": s.span_id,
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "status": s.status,
                    "attributes": s.attributes,
                    "events": s.events
                }
                for s in self.spans
            ]
        }
