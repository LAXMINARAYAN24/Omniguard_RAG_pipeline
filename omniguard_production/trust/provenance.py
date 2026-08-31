"""
provenance.py — Enterprise Document Provenance, Cryptographic Attribution & Lifecycle State Machine.

Every document chunk in OmniGuard-RAG v2 is an immutable, attributed entity
with full cryptographic provenance tracking:
  Tenant -> Publisher/Domain -> Document -> Chunk -> Claim
"""
from __future__ import annotations
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class DocumentState(str, Enum):
    STAGED = "STAGED"                   # Uploaded/Received, awaiting security scan
    SCANNED = "SCANNED"                 # Passed pre-index security & injection scan
    INDEXED = "INDEXED"                 # Embeddings generated & indexed in vector/sparse store
    ACTIVE = "ACTIVE"                   # Live in retrieval pool
    SUSPICIOUS = "SUSPICIOUS"           # Flagged by GWCC or NLI contention (quarantined)
    SUPERSEDED = "SUPERSEDED"           # Replaced by newer document version
    ARCHIVED = "ARCHIVED"               # Removed from live retrieval pool


@dataclass
class DocumentMetadata:
    tenant_id: str = "default"
    source_id: str = "src_default"
    publisher_domain: str = "internal"
    author: Optional[str] = None
    title: Optional[str] = None
    version: str = "1.0"
    created_at: float = field(default_factory=time.time)
    mime_type: str = "text/plain"
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionChunk:
    chunk_id: str
    doc_id: str
    text: str
    clean_text: str
    embedding: Optional[np.ndarray] = None
    sparse_vector: Optional[Dict[int, float]] = None
    char_start: int = 0
    char_end: int = 0
    chunk_index: int = 0
    total_chunks: int = 1
    content_hash: str = ""
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    state: DocumentState = DocumentState.STAGED
    trust_score: float = 1.0
    security_flags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.content_hash and self.text:
            self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def citation_label(self) -> str:
        """Returns deterministic citation reference string."""
        title = self.metadata.title or self.doc_id
        return f"[Doc: {title} | Chunk: {self.chunk_index} | Hash: {self.content_hash[:8]}]"


@dataclass
class ProductionDocument:
    doc_id: str
    raw_content: str
    metadata: DocumentMetadata
    state: DocumentState = DocumentState.STAGED
    chunks: List[ProductionChunk] = field(default_factory=list)
    doc_hash: str = ""
    ingestion_time: float = field(default_factory=time.time)
    security_scan_report: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.doc_hash and self.raw_content:
            self.doc_hash = hashlib.sha256(self.raw_content.encode("utf-8")).hexdigest()
