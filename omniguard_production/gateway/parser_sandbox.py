"""
parser_sandbox.py — Ingestion Parser Sandbox & Document Normalizer.

Provides unprivileged, memory-bounded text normalization, dangerous character stripping,
homoglyph sanitization, and deterministic sliding-window chunking.
"""
from __future__ import annotations
import hashlib
import json
import re
import unicodedata
import uuid
from typing import List, Optional, Dict, Any, Tuple
from ..trust.provenance import ProductionDocument, ProductionChunk, DocumentMetadata, DocumentState

# Regex patterns for dangerous/evasion text patterns
ZERO_WIDTH_CHARS = re.compile(r"[​-‍﻿‪-‮⁠-⁯]")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
HTML_SCRIPT_TAGS = re.compile(r"<\s*(script|style|iframe|object|embed)[^>]*>.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL)
HTML_TAGS = re.compile(r"<[^>]+>")
MULTIPLE_WHITESPACE = re.compile(r"\s+")


class ParserSandbox:
    """Sanitizes raw document input and performs structured chunking."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80, max_doc_size: int = 10_000_000):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_doc_size = max_doc_size

    def sanitize_text(self, raw_text: str) -> Tuple[str, List[str]]:
        """Cleanses input text of zero-width evasion artifacts, scripts, and control codes."""
        flags = []
        if len(raw_text) > self.max_doc_size:
            raw_text = raw_text[:self.max_doc_size]
            flags.append("DOC_TRUNCATED_MAX_SIZE")

        # Check for zero-width characters commonly used in steganographic prompt injection
        if ZERO_WIDTH_CHARS.search(raw_text):
            raw_text = ZERO_WIDTH_CHARS.sub("", raw_text)
            flags.append("ZERO_WIDTH_CHARS_REMOVED")

        # Strip control characters
        if CONTROL_CHARS.search(raw_text):
            raw_text = CONTROL_CHARS.sub(" ", raw_text)
            flags.append("CONTROL_CHARS_SANITIZED")

        # Strip dangerous HTML elements
        if HTML_SCRIPT_TAGS.search(raw_text):
            raw_text = HTML_SCRIPT_TAGS.sub(" ", raw_text)
            flags.append("ACTIVE_SCRIPT_TAGS_STRIPPED")

        # Strip remaining HTML markup while preserving plain text
        if "<" in raw_text and ">" in raw_text:
            raw_text = HTML_TAGS.sub(" ", raw_text)

        # Unicode NFKC normalization (combats lookalike homoglyph evasion)
        norm_text = unicodedata.normalize("NFKC", raw_text)
        clean_text = MULTIPLE_WHITESPACE.sub(" ", norm_text).strip()

        return clean_text, flags

    def chunk_text(self, text: str, doc_id: str, metadata: DocumentMetadata) -> List[ProductionChunk]:
        """Splits sanitized text into overlapping semantic passages."""
        # Simple sentence-boundary aware sliding window
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[ProductionChunk] = []

        current_words: List[str] = []
        current_len = 0
        char_offset = 0

        for sentence in sentences:
            s_words = sentence.split()
            if not s_words:
                continue

            if current_len + len(s_words) > self.chunk_size and current_words:
                chunk_str = " ".join(current_words)
                chunk_id = f"{doc_id}_chk_{len(chunks)}"
                chunks.append(ProductionChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=chunk_str,
                    clean_text=chunk_str,
                    char_start=char_offset,
                    char_end=char_offset + len(chunk_str),
                    chunk_index=len(chunks),
                    metadata=metadata,
                    state=DocumentState.SCANNED
                ))
                char_offset += len(chunk_str) + 1
                # Retain overlap words
                overlap_count = min(len(current_words), self.chunk_overlap)
                current_words = current_words[-overlap_count:] + s_words
                current_len = len(current_words)
            else:
                current_words.extend(s_words)
                current_len += len(s_words)

        # Append final chunk
        if current_words:
            chunk_str = " ".join(current_words)
            chunk_id = f"{doc_id}_chk_{len(chunks)}"
            chunks.append(ProductionChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_str,
                clean_text=chunk_str,
                char_start=char_offset,
                char_end=char_offset + len(chunk_str),
                chunk_index=len(chunks),
                metadata=metadata,
                state=DocumentState.SCANNED
            ))

        # Update total chunks count
        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks

    def process_document(self, raw_content: str, doc_id: Optional[str] = None,
                         metadata: Optional[DocumentMetadata] = None) -> ProductionDocument:
        """End-to-end ingest, sanitize, and chunk a document."""
        if metadata is None:
            metadata = DocumentMetadata()
        if doc_id is None:
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"

        clean_text, flags = self.sanitize_text(raw_content)
        doc = ProductionDocument(
            doc_id=doc_id,
            raw_content=raw_content,
            metadata=metadata,
            state=DocumentState.SCANNED,
            security_scan_report={"flags": flags, "sanitized_length": len(clean_text)}
        )

        chunks = self.chunk_text(clean_text, doc_id=doc_id, metadata=metadata)
        for c in chunks:
            c.security_flags.extend(flags)
        doc.chunks = chunks
        return doc

    # Convenience alias for parse_and_chunk
    parse_and_chunk = process_document
