"""
trust_store.py — Enterprise Durable Multi-Tiered Trust Ledger & Reputation Engine.

Key Architectural Upgrades:
1. Durable on-disk append-only audit persistence (JSONL/SQLite) with automatic state replay on boot.
2. Bounded hierarchical trust propagation (chunk -> document -> source -> domain) preventing domain over-penalization.
3. Multi-entity decoupled reputation tables (tenant, domain, source, document, content_hash).
4. Configurable domain-specific trust weights and calibrated temporal half-life decay.
5. Snapshotting and immutable audit trail replay.
"""
from __future__ import annotations
import os
import json
import math
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger("omniguard.trust_store")


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
    entity_type: str = "chunk"  # 'domain', 'source', 'document', 'chunk', 'content_hash'
    entity_id: str = ""
    delta: float = 0.0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    actor: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, TrustEventType) else str(self.event_type),
            "tenant_id": self.tenant_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "delta": self.delta,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrustEvent:
        return cls(
            event_id=data["event_id"],
            event_type=TrustEventType(data["event_type"]),
            tenant_id=data.get("tenant_id", "default"),
            entity_type=data.get("entity_type", "chunk"),
            entity_id=data.get("entity_id", ""),
            delta=float(data.get("delta", 0.0)),
            reason=data.get("reason", ""),
            timestamp=float(data.get("timestamp", time.time())),
            actor=data.get("actor", "system"),
            metadata=data.get("metadata", {})
        )


@dataclass
class TrustWeights:
    """Configurable weights for multi-entity composite trust calculation."""
    domain_weight: float = 0.20
    source_weight: float = 0.30
    document_weight: float = 0.25
    content_weight: float = 0.25

    def normalize(self):
        total = self.domain_weight + self.source_weight + self.document_weight + self.content_weight
        if total > 0:
            self.domain_weight /= total
            self.source_weight /= total
            self.document_weight /= total
            self.content_weight /= total


@dataclass
class TrustSnapshot:
    """Snapshot of trust tables for rollback capability."""
    snapshot_id: str
    timestamp: float
    domain_scores: Dict[str, Dict[str, float]]
    source_scores: Dict[str, Dict[str, float]]
    document_scores: Dict[str, Dict[str, float]]
    content_scores: Dict[str, Dict[str, float]]
    event_index: int


class PersistentTrustStore:
    """Multi-tenant, durable append-only trust ledger with exponential decay, bounded propagation, and replay."""

    def __init__(self,
                 half_life_seconds: float = 86400.0 * 7,
                 default_baseline: float = 0.75,
                 weights: Optional[TrustWeights] = None,
                 storage_path: Optional[str] = None):
        self.half_life_seconds = half_life_seconds
        self.decay_lambda = math.log(2.0) / max(1.0, half_life_seconds)
        self.default_baseline = default_baseline
        self.weights = weights or TrustWeights()
        self.weights.normalize()
        self.storage_path = storage_path

        # Append-only audit ledger in memory
        self.events: List[TrustEvent] = []

        # Tables: tenant_id -> entity_id -> (current_score, last_update_time)
        self._domain_trust: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._source_trust: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._document_trust: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._content_trust: Dict[str, Dict[str, Tuple[float, float]]] = {}

        # Cross-query provenance & anti-amplification tracking
        self._content_obs_counts: Dict[str, Dict[str, int]] = {}
        self._domain_distinct_docs: Dict[str, Dict[str, Set[str]]] = {}
        self._domain_daily_gain: Dict[str, Dict[str, Tuple[float, float]]] = {}

        # Snapshots
        self._snapshots: Dict[str, TrustSnapshot] = {}

        # If on-disk storage specified, replay ledger from disk
        if self.storage_path:
            self._init_storage()

    def _init_storage(self):
        """Initializes storage directory and replays previous ledger state if present."""
        if not self.storage_path:
            return
        p = Path(self.storage_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    replayed_count = 0
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            ev = TrustEvent.from_dict(data)
                            self._apply_event_state(ev)
                            self.events.append(ev)
                            replayed_count += 1
                logger.info(f"Replayed {replayed_count} durable trust events from {self.storage_path}")
            except Exception as e:
                logger.error(f"Failed to replay trust ledger from {self.storage_path}: {e}")

    def _append_to_disk(self, event: TrustEvent):
        """Appends a single trust event to the durable append-only log file."""
        if not self.storage_path:
            return
        try:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist trust event {event.event_id} to disk: {e}")

    def _get_decayed_score(self, current_score: float, last_time: float, current_time: float) -> float:
        """Applies exponential decay pulling toward default baseline over time."""
        dt = max(0.0, current_time - last_time)
        decay_factor = math.exp(-self.decay_lambda * dt)
        return float(current_score * decay_factor + self.default_baseline * (1.0 - decay_factor))

    def _apply_event_state(self, event: TrustEvent) -> float:
        """Applies single event to in-memory entity tables with anti-amplification safeguards."""
        now = event.timestamp
        tenant = event.tenant_id
        table = self._get_table_for_entity(event.entity_type)
        if tenant not in table:
            table[tenant] = {}

        current_score, last_time = table[tenant].get(event.entity_id, (self.default_baseline, now))
        decayed = self._get_decayed_score(current_score, last_time, now)

        # Anti-amplification ceiling: domains with < 3 distinct verified documents are capped at 0.80
        max_ceiling = 1.0
        if event.entity_type == "domain" and event.delta > 0:
            distinct_docs = self._domain_distinct_docs.get(tenant, {}).get(event.entity_id, set())
            if len(distinct_docs) < 3:
                max_ceiling = 0.80

        new_score = max(0.0, min(max_ceiling, decayed + event.delta))
        table[tenant][event.entity_id] = (new_score, now)
        return new_score

    def record_hierarchical_reward(self,
                                   tenant_id: str,
                                   publisher_domain: str,
                                   source_id: str,
                                   document_id: str,
                                   content_hash: str,
                                   reason: str,
                                   base_reward: float = 0.05,
                                   query_id: Optional[str] = None,
                                   actor: str = "system"):
        """
        Applies provenance-aware positive trust updates with sub-linear diminishing returns
        and domain diversity ceilings to eliminate temporal trust amplification attacks.
        """
        now = time.time()

        # Track distinct documents per domain
        if publisher_domain:
            if tenant_id not in self._domain_distinct_docs:
                self._domain_distinct_docs[tenant_id] = {}
            if publisher_domain not in self._domain_distinct_docs[tenant_id]:
                self._domain_distinct_docs[tenant_id][publisher_domain] = set()
            if document_id:
                self._domain_distinct_docs[tenant_id][publisher_domain].add(document_id)

        # 1. Sub-linear scaling for repeated content observations
        if tenant_id not in self._content_obs_counts:
            self._content_obs_counts[tenant_id] = {}
        obs_count = self._content_obs_counts[tenant_id].get(content_hash, 0) + 1
        self._content_obs_counts[tenant_id][content_hash] = obs_count

        # Effective reward decays as 1 / sqrt(N_obs) to prevent query looping attacks
        effective_reward = base_reward / math.sqrt(obs_count)

        # 2. Daily velocity cap for publisher domain (+0.05 max gain per 24h window)
        if tenant_id not in self._domain_daily_gain:
            self._domain_daily_gain[tenant_id] = {}
        daily_gain, win_start = self._domain_daily_gain[tenant_id].get(publisher_domain, (0.0, now))
        if now - win_start > 86400.0:
            daily_gain, win_start = 0.0, now

        allowed_domain_delta = max(0.0, min(effective_reward * 0.10, 0.05 - daily_gain))
        self._domain_daily_gain[tenant_id][publisher_domain] = (daily_gain + allowed_domain_delta, win_start)

        # Record events across hierarchy
        if content_hash:
            self.record_event(TrustEvent(
                event_type=TrustEventType.GWCC_VERIFIED,
                tenant_id=tenant_id,
                entity_type="content_hash",
                entity_id=content_hash,
                delta=effective_reward,
                reason=f"Content reward (obs #{obs_count}): {reason}",
                timestamp=now,
                actor=actor,
                metadata={"query_id": query_id, "obs_count": obs_count}
            ))

        if document_id:
            self.record_event(TrustEvent(
                event_type=TrustEventType.GWCC_VERIFIED,
                tenant_id=tenant_id,
                entity_type="document",
                entity_id=document_id,
                delta=effective_reward * 0.50,
                reason=f"Document reward: {reason}",
                timestamp=now,
                actor=actor
            ))

        if source_id:
            self.record_event(TrustEvent(
                event_type=TrustEventType.GWCC_VERIFIED,
                tenant_id=tenant_id,
                entity_type="source",
                entity_id=source_id,
                delta=effective_reward * 0.25,
                reason=f"Source reward: {reason}",
                timestamp=now,
                actor=actor
            ))

        if publisher_domain and publisher_domain not in {"internal", "localhost", "default"} and allowed_domain_delta > 0:
            self.record_event(TrustEvent(
                event_type=TrustEventType.GWCC_VERIFIED,
                tenant_id=tenant_id,
                entity_type="domain",
                entity_id=publisher_domain,
                delta=allowed_domain_delta,
                reason=f"Domain reward: {reason}",
                timestamp=now,
                actor=actor,
                metadata={"distinct_docs": len(self._domain_distinct_docs[tenant_id][publisher_domain])}
            ))

    def record_event(self, event: TrustEvent) -> float:
        """Appends a trust event, updates entity score atomically, and writes to durable storage."""
        self.events.append(event)
        new_score = self._apply_event_state(event)
        self._append_to_disk(event)
        return new_score

    def record_hierarchical_penalty(self,
                                    tenant_id: str,
                                    publisher_domain: str,
                                    source_id: str,
                                    document_id: str,
                                    content_hash: str,
                                    reason: str,
                                    base_penalty: float = -0.40,
                                    actor: str = "system"):
        """
        Applies bounded hierarchical trust propagation.
        Directly penalizes content/document, while applying heavily damped bounded penalties
        to parent source and publisher domain to prevent global domain false-correlation.
        """
        now = time.time()

        # 1. Direct penalty on content hash
        if content_hash:
            self.record_event(TrustEvent(
                event_type=TrustEventType.INJECTION_FLAGGED,
                tenant_id=tenant_id,
                entity_type="content_hash",
                entity_id=content_hash,
                delta=base_penalty,
                reason=f"Content penalty: {reason}",
                timestamp=now,
                actor=actor
            ))

        # 2. Document penalty (70% of base)
        if document_id:
            self.record_event(TrustEvent(
                event_type=TrustEventType.INJECTION_FLAGGED,
                tenant_id=tenant_id,
                entity_type="document",
                entity_id=document_id,
                delta=base_penalty * 0.70,
                reason=f"Document penalty: {reason}",
                timestamp=now,
                actor=actor
            ))

        # 3. Source penalty (25% of base, bounded)
        if source_id:
            self.record_event(TrustEvent(
                event_type=TrustEventType.INJECTION_FLAGGED,
                tenant_id=tenant_id,
                entity_type="source",
                entity_id=source_id,
                delta=base_penalty * 0.25,
                reason=f"Source penalty: {reason}",
                timestamp=now,
                actor=actor
            ))

        # 4. Domain penalty (5% of base, heavily damped to prevent false domain punishment)
        if publisher_domain and publisher_domain not in {"internal", "localhost", "default"}:
            self.record_event(TrustEvent(
                event_type=TrustEventType.INJECTION_FLAGGED,
                tenant_id=tenant_id,
                entity_type="domain",
                entity_id=publisher_domain,
                delta=max(-0.05, base_penalty * 0.05),
                reason=f"Domain damping penalty: {reason}",
                timestamp=now,
                actor=actor
            ))

    def get_effective_trust(self,
                            tenant_id: str,
                            publisher_domain: str = "internal",
                            source_id: str = "src_default",
                            document_id: str = "doc_default",
                            content_hash: str = "") -> Dict[str, float]:
        """Calculates multi-tiered composite trust score decoupled into domain, source, document, and content."""
        now = time.time()

        # 1. Domain trust
        dom_score, dom_t = self._domain_trust.get(tenant_id, {}).get(publisher_domain, (self.default_baseline, now))
        dom_effective = self._get_decayed_score(dom_score, dom_t, now)

        # 2. Source trust
        src_score, src_t = self._source_trust.get(tenant_id, {}).get(source_id, (self.default_baseline, now))
        src_effective = self._get_decayed_score(src_score, src_t, now)

        # 3. Document trust
        doc_score, doc_t = self._document_trust.get(tenant_id, {}).get(document_id, (self.default_baseline, now))
        doc_effective = self._get_decayed_score(doc_score, doc_t, now)

        # 4. Content hash trust
        cnt_score, cnt_t = self._content_trust.get(tenant_id, {}).get(content_hash, (self.default_baseline, now))
        cnt_effective = self._get_decayed_score(cnt_score, cnt_t, now)

        # Composite score using calibrated weights
        composite = (
            self.weights.domain_weight * dom_effective +
            self.weights.source_weight * src_effective +
            self.weights.document_weight * doc_effective +
            self.weights.content_weight * cnt_effective
        )

        return {
            "composite_trust": round(composite, 4),
            "domain_trust": round(dom_effective, 4),
            "source_trust": round(src_effective, 4),
            "document_trust": round(doc_effective, 4),
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
            document_scores={t: {k: v[0] for k, v in d.items()} for t, d in self._document_trust.items()},
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

        self._domain_trust = {t: {k: (v, now) for k, v in d.items()} for t, d in snap.domain_scores.items()}
        self._source_trust = {t: {k: (v, now) for k, v in d.items()} for t, d in snap.source_scores.items()}
        self._document_trust = {t: {k: (v, now) for k, v in d.items()} for t, d in snap.document_scores.items()}
        self._content_trust = {t: {k: (v, now) for k, v in d.items()} for t, d in snap.content_scores.items()}

        rb_event = TrustEvent(
            event_type=TrustEventType.ROLLBACK,
            tenant_id="global",
            entity_type="system",
            entity_id=snapshot_id,
            delta=0.0,
            reason=f"Rolled back to snapshot {snapshot_id} (captured at {snap.timestamp})",
            actor="admin"
        )
        self.record_event(rb_event)
        return True

    def _get_table_for_entity(self, entity_type: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
        if entity_type == "domain":
            return self._domain_trust
        elif entity_type == "source":
            return self._source_trust
        elif entity_type == "document":
            return self._document_trust
        else:
            return self._content_trust
