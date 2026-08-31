"""
trust_store.py — Persistent, Event-Driven Multi-Signal Trust Ledger & Reputation Engine.

Decouples publisher/source domain reputation from specific chunk content validity,
maintains an append-only audit ledger, and supports temporal decay and state snapshots.
"""
from __future__ import annotations
import math
import time
import uuid
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple


class TrustEventType(str, Enum):
    SECURITY_SCAN_PASS = "SECURITY_SCAN_PASS"
    INJECTION_FLAGGED = "INJECTION_FLAGGED"
    DRS_ANOMALY = "DRS_ANOMALY"
    NLI_CONTRADICTION = "NLI_CONTRADICTION"
    GWCC_QUARANTINE = "GWCC_QUARANTINE"
    GWCC_VERIFIED = "GWCC_VERIFIED"
    USER_POSITIVE_FEEDBACK = "USER_POSITIVE_FEEDBACK"
    USER_NEGATIVE_FEEDBACK = "USER_NEGATIVE_FEEDBACK"
    ADMIN_OVERRIDE = "ADMIN_OVERRIDE"
    ROLLBACK = "ROLLBACK"


@dataclass
class TrustEvent:
    """Immutable audit record of a trust update."""
    event_id: str = field(default_factory=lambda: f"te_{uuid.uuid4().hex[:12]}")
    event_type: TrustEventType = TrustEventType.SECURITY_SCAN_PASS
    tenant_id: str = "default"
    entity_type: str = "chunk"  # 'domain', 'source', 'chunk', 'content_hash'
    entity_id: str = ""
    delta: float = 0.0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    actor: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustSnapshot:
    """Snapshot of trust tables for rollback capability."""
    snapshot_id: str
    timestamp: float
    domain_scores: Dict[str, Dict[str, float]]
    source_scores: Dict[str, Dict[str, float]]
    content_scores: Dict[str, Dict[str, float]]
    event_index: int


class PersistentTrustStore:
    """Multi-tenant, append-only trust ledger with exponential decay and rollback."""

    def __init__(self, half_life_seconds: float = 86400.0 * 7, default_baseline: float = 0.75):
        self.half_life_seconds = half_life_seconds
        self.decay_lambda = math.log(2.0) / max(1.0, half_life_seconds)
        self.default_baseline = default_baseline

        # Append-only audit ledger
        self.events: List[TrustEvent] = []

        # Tables: tenant_id -> entity_id -> (current_score, last_update_time)
        self._domain_trust: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._source_trust: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._content_trust: Dict[str, Dict[str, Tuple[float, float]]] = {}

        # Snapshots
        self._snapshots: Dict[str, TrustSnapshot] = {}

    def _get_decayed_score(self, current_score: float, last_time: float, current_time: float) -> float:
        """Applies exponential decay pulling toward default baseline over time."""
        dt = max(0.0, current_time - last_time)
        decay_factor = math.exp(-self.decay_lambda * dt)
        return float(current_score * decay_factor + self.default_baseline * (1.0 - decay_factor))

    def record_event(self, event: TrustEvent) -> float:
        """Appends a trust event and updates the relevant entity score atomically."""
        self.events.append(event)
        now = event.timestamp
        tenant = event.tenant_id

        table = self._get_table_for_entity(event.entity_type)
        if tenant not in table:
            table[tenant] = {}

        current_score, last_time = table[tenant].get(event.entity_id, (self.default_baseline, now))
        decayed = self._get_decayed_score(current_score, last_time, now)
        new_score = max(0.0, min(1.0, decayed + event.delta))

        table[tenant][event.entity_id] = (new_score, now)
        return new_score

    def get_effective_trust(self,
                            tenant_id: str,
                            publisher_domain: str = "internal",
                            source_id: str = "src_default",
                            content_hash: str = "") -> Dict[str, float]:
        """Calculates multi-tiered composite trust score decoupled into domain, source, and content."""
        now = time.time()

        # 1. Domain trust
        dom_score, dom_t = self._domain_trust.get(tenant_id, {}).get(publisher_domain, (self.default_baseline, now))
        dom_effective = self._get_decayed_score(dom_score, dom_t, now)

        # 2. Source trust
        src_score, src_t = self._source_trust.get(tenant_id, {}).get(source_id, (self.default_baseline, now))
        src_effective = self._get_decayed_score(src_score, src_t, now)

        # 3. Content hash trust
        cnt_score, cnt_t = self._content_trust.get(tenant_id, {}).get(content_hash, (self.default_baseline, now))
        cnt_effective = self._get_decayed_score(cnt_score, cnt_t, now)

        # Composite score: 40% domain, 30% source, 30% content validity
        composite = 0.40 * dom_effective + 0.30 * src_effective + 0.30 * cnt_effective

        return {
            "composite_trust": round(composite, 4),
            "domain_trust": round(dom_effective, 4),
            "source_trust": round(src_effective, 4),
            "content_trust": round(cnt_effective, 4)
        }

    def create_snapshot(self, snapshot_id: Optional[str] = None) -> str:
        """Captures a snapshot of all trust score states for instant rollback."""
        sid = snapshot_id or f"snap_{uuid.uuid4().hex[:8]}"
        now = time.time()

        snap = TrustSnapshot(
            snapshot_id=sid,
            timestamp=now,
            domain_scores={t: {k: v[0] for k, v in d.items()} for t, d in self._domain_trust.items()},
            source_scores={t: {k: v[0] for k, v in d.items()} for t, d in self._source_trust.items()},
            content_scores={t: {k: v[0] for k, v in d.items()} for t, d in self._content_trust.items()},
            event_index=len(self.events)
        )
        self._snapshots[sid] = snap
        return sid

    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """Rolls back score states and marks ledger with a ROLLBACK event."""
        if snapshot_id not in self._snapshots:
            return False

        snap = self._snapshots[snapshot_id]
        now = time.time()

        # Restore tables
        self._domain_trust = {t: {k: (v, now) for k, v in d.items()} for t, d in snap.domain_scores.items()}
        self._source_trust = {t: {k: (v, now) for k, v in d.items()} for t, d in snap.source_scores.items()}
        self._content_trust = {t: {k: (v, now) for k, v in d.items()} for t, d in snap.content_scores.items()}

        # Record rollback event
        self.events.append(TrustEvent(
            event_type=TrustEventType.ROLLBACK,
            tenant_id="global",
            entity_type="system",
            entity_id=snapshot_id,
            delta=0.0,
            reason=f"Rolled back to snapshot {snapshot_id} (captured at {snap.timestamp})",
            actor="admin"
        ))
        return True

    def _get_table_for_entity(self, entity_type: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
        if entity_type == "domain":
            return self._domain_trust
        elif entity_type == "source":
            return self._source_trust
        else:
            return self._content_trust
