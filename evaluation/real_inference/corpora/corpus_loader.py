"""
Corpus loader for Track B Real-Inference Evaluation.
Loads multi-domain documents into OmniGuardProductionPipeline with zero shortcuts.
"""

from typing import List, Dict, Any, Optional
import os
import sys

from omniguard_production.pipeline import OmniGuardProductionPipeline
from omniguard_production.models import DocumentMetadata
from evaluation.real_inference.corpora.real_documents_data import REAL_DOMAINS_DATA


class RealCorpusLoader:
    """
    Manages loading and indexing of realistic multi-domain corpora into the
    production defense pipeline.
    """

    def __init__(self, topics_data: Optional[List[Dict[str, Any]]] = None):
        self.topics_data = topics_data or REAL_DOMAINS_DATA

    def populate_pipeline(
        self,
        pipeline: OmniGuardProductionPipeline,
        tenant_id: Optional[str] = None,
        inject_distractors: bool = True
    ) -> Dict[str, Any]:
        """
        Ingests all clean real documents (and optional distractors) into the provided pipeline.
        Returns indexing statistics.
        """
        total_docs = 0
        total_chunks = 0
        tenant_counts: Dict[str, int] = {}

        for topic in self.topics_data:
            topic_tenant = topic.get("clean_documents", [{}])[0].get("tenant_id", "default")
            if tenant_id and topic_tenant != tenant_id:
                continue

            for doc in topic.get("clean_documents", []):
                doc_tenant = doc.get("tenant_id", "default")
                if tenant_id and doc_tenant != tenant_id:
                    continue

                metadata = DocumentMetadata(
                    title=doc["title"],
                    publisher_domain=doc["publisher_domain"],
                    source_id=doc["source_id"],
                    tenant_id=doc_tenant
                )

                chunks = pipeline.ingest_document(
                    raw_text=doc["text"],
                    metadata=metadata
                )
                total_docs += 1
                total_chunks += len(chunks)
                tenant_counts[doc_tenant] = tenant_counts.get(doc_tenant, 0) + 1

        return {
            "indexed_documents": total_docs,
            "indexed_chunks": total_chunks,
            "tenants_populated": tenant_counts,
            "drs_calibrated": pipeline.drs_engine.is_calibrated()
        }

    def get_topic_by_id(self, topic_id: str) -> Optional[Dict[str, Any]]:
        for topic in self.topics_data:
            if topic["topic_id"] == topic_id:
                return topic
        return None

    def get_all_topics(self) -> List[Dict[str, Any]]:
        return list(self.topics_data)
