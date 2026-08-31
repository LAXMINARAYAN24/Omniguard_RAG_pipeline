"""
evidence_graph.py — Graph-Theoretic Evidence Network & Collusion Cluster Detection.

Builds an Evidence Independence Graph over candidate chunks and partitions them
into consensus groups and adversarial collusion clusters using modularity community detection.
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
    is_adversarial_candidate: bool = False
    evidence_weight: float = 1.0


class EvidenceGraph:
    """Constructs and analyzes the semantic affinity and contradiction graph across retrieved chunks."""

    def __init__(self,
                 similarity_threshold: float = 0.50,
                 contradiction_threshold: float = 0.60,
                 embedding_provider: Optional[EmbeddingProvider] = None):
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.embedding_provider = embedding_provider

    def build_graph(self, chunks: List[ProductionChunk],
                    contradiction_matrix: Optional[np.ndarray] = None) -> nx.Graph:
        """Constructs an undirected weighted graph where nodes are chunk_ids."""
        G = nx.Graph()

        # Add nodes with metadata attributes
        for c in chunks:
            G.add_node(
                c.chunk_id,
                chunk=c,
                trust=c.trust_score,
                domain=c.metadata.publisher_domain,
                flags=c.security_flags
            )

        n = len(chunks)
        if n <= 1:
            return G

        # Compute pairwise similarities
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
                c_score = 0.0
                if contradiction_matrix is not None and i < contradiction_matrix.shape[0] and j < contradiction_matrix.shape[1]:
                    c_score = float(contradiction_matrix[i, j])

                # An edge is positive support if similarity >= threshold and low contradiction
                if sim >= self.similarity_threshold and c_score < self.contradiction_threshold:
                    weight = sim * (1.0 - c_score)
                    G.add_edge(chunks[i].chunk_id, chunks[j].chunk_id, weight=weight, relation="support")
                elif c_score >= self.contradiction_threshold:
                    G.add_edge(chunks[i].chunk_id, chunks[j].chunk_id, weight=c_score, relation="contradiction")

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

            # Check if domain diversity is suspiciously 1 with high security flags (collusion indicator)
            any_flags = any(c.security_flags for c in cluster_chunks)
            is_adversarial = (len(cluster_chunks) > 1 and unique_domains == 1 and any_flags) or (avg_trust < 0.6) or any_flags

            # Intra-cluster density
            sub_G = support_G.subgraph(node_ids)
            density = float(nx.density(sub_G)) if len(node_ids) > 1 else 1.0

            # Evidence weight = size * avg_trust * domain_diversity factor
            evidence_weight = len(cluster_chunks) * avg_trust * (1.0 + 0.3 * (unique_domains - 1))
            if is_adversarial:
                evidence_weight *= 0.1

            clusters.append(EvidenceCluster(
                cluster_id=c_idx,
                chunks=cluster_chunks,
                average_trust=round(avg_trust, 4),
                intra_cluster_density=round(density, 4),
                domain_diversity=unique_domains,
                is_adversarial_candidate=is_adversarial,
                evidence_weight=round(evidence_weight, 4)
            ))

        # Sort clusters by evidence weight descending
        clusters.sort(key=lambda x: x.evidence_weight, reverse=True)
        return clusters
