"""
prompt_assembler.py — Source-Anchored Prompt Synthesizer with Strict Negative Constraints & Citation Formatting.

Constructs grounded system prompts that eliminate fabrication by enforcing strict contextual boundaries.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from ..trust.provenance import ProductionChunk


STRICT_GROUNDED_SYSTEM_PROMPT = """You are a strictly grounded, factual enterprise intelligence assistant operating under zero-fabrication constraints.

OPERATING CONSTRAINTS:
1. SOURCE ANCHORING: Answer the query using ONLY the verified evidence passages provided below. If the provided evidence is insufficient to answer the query truthfully, state: "Information not available in provided sources."
2. NEGATIVE CONSTRAINTS: Do NOT extrapolate, speculate, or introduce external facts, libraries, or methodologies not explicitly stated in the source text.
3. CITATION MANDATES: Every factual claim must include an inline citation in the exact format [Doc: <Title> | Chunk: <Idx> | Hash: <Hash>].
4. CONFLICT RESOLUTION: If the provided evidence contains contradictory assertions, explicitly cite both viewpoints and declare the evidence in conflict."""


class PromptAssembler:
    """Assembles prompt payloads with grounded evidence formatting and cryptographic citation tags."""

    def __init__(self, system_prompt: str = STRICT_GROUNDED_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def assemble_prompt(self,
                        query_text: str,
                        verified_chunks: List[ProductionChunk],
                        include_cot_instructions: bool = True) -> Dict[str, Any]:
        """Formats the query and verified chunks into a strict grounded prompt payload."""
        evidence_blocks = []
        for c in verified_chunks:
            title = c.metadata.title or c.doc_id
            h_short = c.content_hash[:8] if c.content_hash else "nohash"
            label = f"[Doc: {title} | Chunk: {c.chunk_index} | Hash: {h_short}]"
            block = f"--- BEGIN EVIDENCE {label} ---\n{c.clean_text}\n--- END EVIDENCE {label} ---"
            evidence_blocks.append(block)

        joined_evidence = "\n\n".join(evidence_blocks) if evidence_blocks else "No evidence provided."

        cot_prompt = ""
        if include_cot_instructions:
            cot_prompt = (
                "\nINSTRUCTIONS:\n"
                "1. Read the provided evidence carefully.\n"
                "2. Identify the specific sentences directly answering the query.\n"
                "3. Formulate a precise, grounded answer.\n"
                "4. Attach the exact evidence citation tag after each claim.\n"
            )

        user_content = (
            f"VERIFIED EVIDENCE CONTEXT:\n\n{joined_evidence}\n\n"
            f"USER QUERY:\n{query_text}\n"
            f"{cot_prompt}"
        )

        return {
            "system_prompt": self.system_prompt,
            "user_prompt": user_content,
            "evidence_chunk_count": len(verified_chunks),
            "expected_citations": [
                f"[Doc: {c.metadata.title or c.doc_id} | Chunk: {c.chunk_index} | Hash: {c.content_hash[:8]}]"
                for c in verified_chunks
            ]
        }
