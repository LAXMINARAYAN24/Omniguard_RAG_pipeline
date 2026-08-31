"""
evidence_graph.py — Graph-Theoretic Evidence Network & Source Lineage Independence Matrix.

Builds an Evidence Independence Graph over candidate chunks incorporating
cross-source lineage independence matrices to prevent homogeneous collusion amplification,
and partitions them into consensus groups and adversarial candidate clusters.
"""
from __future__ import annotations
import numpy as np
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Tuple, Optional
import networkx as nx
from ..trust.provenance import ProductionChunk
from ..embeddings.base import EmbeddingProvider

_WORD_REGEX = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class EvidenceCluster:
    """Represents a coherent cluster of mutually supporting evidence chunks."""
    cluster_id: int
    chunks: List[ProductionChunk]
    average_trust: float = 1.0
    intra_cluster_density: float = 0.0
    domain_diversity: int = 1
    source_diversity: int = 1
    lineage_independence_score: float = 1.0
    is_adversarial_candidate: bool = False
    evidence_weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


class EvidenceGraph:
    """
    Constructs and analyzes the semantic affinity and contradiction graph across retrieved chunks,
    gating support edges with source lineage independence weights.
    """

    def __init__(self,
                 similarity_threshold: float = 0.45,
                 contradiction_threshold: float = 0.55,
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 same_domain_discount: float = 0.65,
                 same_source_discount: float = 0.35):
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.embedding_provider = embedding_provider
        self.same_domain_discount = same_domain_discount
        self.same_source_discount = same_source_discount

    def compute_source_independence_matrix(self, chunks: List[ProductionChunk]) -> np.ndarray:
        """
        Computes an NxN matrix M where M[i, j] in [0, 1] represents the lineage independence
        between chunk i and chunk j.
        - Different domains & independent sources: ~1.0
        - Same domain, different source: ~0.65
        - Same source document: ~0.35
        - Verbatim template n-gram repetition discount applied.
        """
        n = len(chunks)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)

        M = np.ones((n, n), dtype=np.float64)
        for i in range(n):
            M[i, i] = 1.0
            for j in range(i + 1, n):
                c_i = chunks[i]
                c_j = chunks[j]

                indep = 1.0

                # 1. Source Document lineage
                doc_id_i = getattr(c_i, "doc_id", getattr(c_i, "document_id", ""))
                doc_id_j = getattr(c_j, "doc_id", getattr(c_j, "document_id", ""))
                if doc_id_i and doc_id_j and doc_id_i == doc_id_j:
                    indep *= self.same_source_discount
                elif c_i.metadata.source_id and c_j.metadata.source_id and c_i.metadata.source_id == c_j.metadata.source_id:
                    indep *= self.same_source_discount
                # 2. Publisher Domain lineage
                elif c_i.metadata.publisher_domain and c_j.metadata.publisher_domain and \
                        c_i.metadata.publisher_domain == c_j.metadata.publisher_domain and \
                        c_i.metadata.publisher_domain not in {"internal", "localhost", "default"}:
                    indep *= self.same_domain_discount

                # 3. Lexical template duplication discount (detect near-identical injected sentences)
                tokens_i = set(_WORD_REGEX.findall(c_i.clean_text.lower()))
                tokens_j = set(_WORD_REGEX.findall(c_j.clean_text.lower()))
                if tokens_i and tokens_j:
                    jaccard = len(tokens_i & tokens_j) / max(1, len(tokens_i | tokens_j))
                    if jaccard > 0.80 and indep > 0.5:
                        # Near-verbatim text across different document IDs indicates coordinated copy-paste attack
                        indep *= 0.50

                M[i, j] = round(indep, 4)
                M[j, i] = round(indep, 4)

        return M

    def build_graph(self,
                    chunks: List[ProductionChunk],
                    contradiction_matrix: Optional[np.ndarray] = None) -> nx.Graph:
        """
        Constructs an undirected weighted graph where nodes are chunk_ids,
        edges reflect semantic support discounted by source lineage independence or contradiction.
        """
        G = nx.Graph()

        # Add nodes with metadata attributes
        for c in chunks:
            G.add_node(
                c.chunk_id,
                chunk=c,
                trust=c.trust_score,
                domain=c.metadata.publisher_domain,
                source_id=c.metadata.source_id,
                document_id=getattr(c, "doc_id", getattr(c, "document_id", "")),
                flags=c.security_flags
            )

        n = len(chunks)
        if n <= 1:
            return G

        # Compute Lineage Independence Matrix
        indep_matrix = self.compute_source_independence_matrix(chunks)

        # Compute pairwise semantic similarities
        sim_matrix = np.zeros((n, n), dtype=np.float64)

        has_embeddings = all(c.embedding is not None for c in chunks)
        if has_embeddings or self.embedding_provider is not None:
            embeddings = []
            for c in chunks:
                if c.embedding is not None:
                    embeddings.append(c.embedding)
                else:
                    emb = self.embedding_provider.embed_text(c.clean_text)
                    c.embedding = emb
                    embeddings.append(emb)
            embeddings = np.array(embeddings, dtype=np.float64)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            embeddings = embeddings / norms
            sim_matrix = np.dot(embeddings, embeddings.T)
        else:
            # High-resolution token overlap similarity fallback
            token_sets = [set(_WORD_REGEX.findall(c.clean_text.lower())) for c in chunks]
            for i in range(n):
                sim_matrix[i, i] = 1.0
                for j in range(i + 1, n):
                    union_sz = len(token_sets[i] | token_sets[j])
                    if union_sz > 0:
                        jaccard = len(token_sets[i] & token_sets[j]) / union_sz
                    else:
                        jaccard = 0.0
                    sim_matrix[i, j] = jaccard
                    sim_matrix[j, i] = jaccard

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                indep = float(indep_matrix[i, j])
                c_score = 0.0
                if contradiction_matrix is not None and i < contradiction_matrix.shape[0] and j < contradiction_matrix.shape[1]:
                    c_score = float(contradiction_matrix[i, j])

                # An edge is positive support if similarity >= threshold and low contradiction
                if sim >= self.similarity_threshold and c_score < self.contradiction_threshold:
                    # Gated support weight = similarity * (1 - contradiction) * independence
                    weight = sim * (1.0 - c_score) * indep
                    G.add_edge(chunks[i].chunk_id, chunks[j].chunk_id, weight=weight, relation="support", indep=indep)
                elif c_score >= self.contradiction_threshold:
                    G.add_edge(chunks[i].chunk_id, chunks[j].chunk_id, weight=c_score, relation="contradiction", indep=indep)

        return G

    def detect_communities(self, G: nx.Graph) -> List[EvidenceCluster]:
        """Partitions the evidence graph into discrete communities / collusion clusters."""
        if len(G) == 0:
            return []

        # Create support subgraph for positive community detection
        support_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("relation") == "support"]
        support_G = nx.Graph()
        support_G.add_nodes_from(G.nodes(data=True))
        support_G.add_edges_from(support_edges)

        # Modularity community detection
        try:
            communities_gen = nx.community.greedy_modularity_communities(support_G, weight="weight")
            raw_communities = [list(c) for c in communities_gen]
        except Exception:
            # Fallback to connected components
            raw_communities = list(nx.connected_components(support_G))

        clusters: List[EvidenceCluster] = []
        node_map = {n: G.nodes[n]["chunk"] for n in G.nodes}

        for c_idx, node_ids in enumerate(raw_communities):
            cluster_chunks = [node_map[nid] for nid in node_ids if nid in node_map]
            if not cluster_chunks:
                continue

            # Calculate cluster metrics
            avg_trust = float(np.mean([c.trust_score for c in cluster_chunks]))
            unique_domains = len(set(c.metadata.publisher_domain for c in cluster_chunks if c.metadata.publisher_domain))
            unique_domains = max(1, unique_domains)

            unique_sources = len(set(c.metadata.source_id for c in cluster_chunks if c.metadata.source_id))
            unique_sources = max(1, unique_sources)

            # Intra-cluster density & lineage independence
            sub_G = support_G.subgraph(node_ids)
            density = float(nx.density(sub_G)) if len(node_ids) > 1 else 1.0

            # Lineage independence score for cluster
            if len(node_ids) > 1:
                edge_indeps = [d.get("indep", 1.0) for _, _, d in sub_G.edges(data=True)]
                cluster_indep = float(np.mean(edge_indeps)) if edge_indeps else 0.5
            else:
                cluster_indep = 1.0

            # Collusion indicator: multiple chunks with low domain/source diversity and security flags
            any_flags = any(c.security_flags for c in cluster_chunks)
            is_adversarial = (
                (len(cluster_chunks) > 1 and unique_domains == 1 and cluster_indep < 0.5 and any_flags) or
                (avg_trust < 0.6) or
                any_flags
            )

            # Effective evidence weight: size discounted by independence and trust
            diversity_boost = 1.0 + 0.35 * (unique_domains - 1) + 0.20 * (unique_sources - 1)
            evidence_weight = len(cluster_chunks) * avg_trust * cluster_indep * diversity_boost
            if is_adversarial:
                evidence_weight *= 0.15

            clusters.append(EvidenceCluster(
                cluster_id=c_idx,
                chunks=cluster_chunks,
                average_trust=round(avg_trust, 4),
                intra_cluster_density=round(density, 4),
                domain_diversity=unique_domains,
                source_diversity=unique_sources,
                lineage_independence_score=round(cluster_indep, 4),
                is_adversarial_candidate=is_adversarial,
                evidence_weight=round(evidence_weight, 4),
                details={
                    "node_count": len(node_ids),
                    "unique_domains": unique_domains,
                    "unique_sources": unique_sources,
                    "cluster_indep": round(cluster_indep, 4)
                }
            ))

        # Sort clusters by evidence weight descending
        clusters.sort(key=lambda x: x.evidence_weight, reverse=True)
        return clusters
