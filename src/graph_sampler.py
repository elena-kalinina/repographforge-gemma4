#!/usr/bin/env python3
"""
Milestone 1 & 3: Load pre-computed competition code graphs and 256-dim .npz embeddings
to sample multi-file connected subgraphs for RepoGraphForge task synthesis.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np

try:
    from src.data_loader import RepoGraph
except ImportError:
    from data_loader import RepoGraph

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("graph_sampler")


class RepoGraphSampler:
    """
    Samples multi-hop structural subgraphs enriched with semantic embedding neighbors.
    """

    def __init__(self, graph_json_path: Path, embedding_npz_path: Optional[Path] = None):
        self.graph_path = graph_json_path
        self.repo_graph = RepoGraph(graph_path=graph_json_path, embedding_path=embedding_npz_path)
        self.graph = self.repo_graph.graph
        self.embeddings = self.repo_graph.embeddings

    def find_candidate_anchors(
        self, min_degree: int = 1, max_degree: int = 25
    ) -> List[str]:
        """
        Filter for non-trivial function/method/class nodes with structural connections.
        Detects nodes by 'text' attribute starting with def/class or non-empty source.
        """
        candidates = []
        for node_id, attrs in self.graph.nodes(data=True):
            text = attrs.get("text", "")
            # Check if text is a code definition or if symbol name indicates a callable
            is_code = False
            if text:
                stripped = text.lstrip()
                if stripped.startswith(("def ", "async def ", "class ", "@")):
                    is_code = True
            elif "." in str(node_id):
                is_code = True

            if not is_code:
                continue

            deg = self.graph.in_degree(node_id) + self.graph.out_degree(node_id)
            if min_degree <= deg <= max_degree:
                candidates.append(node_id)
        return candidates

    def semantic_top_k(
        self, anchor_id: str, exclude_ids: Set[str], top_k: int = 2
    ) -> List[str]:
        """
        Find top-k cosine-similar nodes across the repository using 256-dim embeddings.
        """
        if anchor_id not in self.embeddings:
            return []

        anchor_vec = self.embeddings[anchor_id]
        anchor_norm = float(np.linalg.norm(anchor_vec)) + 1e-8

        scored = []
        for node_id, vec in self.embeddings.items():
            if node_id == anchor_id or node_id in exclude_ids or node_id not in self.graph:
                continue
            sim = float(np.dot(anchor_vec, vec) / (anchor_norm * (np.linalg.norm(vec) + 1e-8)))
            scored.append((sim, node_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [nid for _, nid in scored[:top_k]]

    def sample_subgraph(
        self,
        anchor_id: str,
        max_nodes: int = 6,
        semantic_additions: int = 1,
    ) -> Dict[str, Any]:
        """
        Extracts a 2-hop structural ego-graph + semantic neighbors for GraphForge synthesis.
        """
        undirected = self.graph.to_undirected()
        ego_nodes = list(nx.ego_graph(undirected, anchor_id, radius=2).nodes())
        random.shuffle(ego_nodes)

        selected: List[str] = [anchor_id]
        for nid in ego_nodes:
            if nid != anchor_id and len(selected) < (max_nodes - semantic_additions):
                selected.append(nid)

        sem_nodes = self.semantic_top_k(anchor_id, set(selected), top_k=semantic_additions)
        selected.extend(sem_nodes)

        sub_g = self.graph.subgraph(selected)
        files_touched = set()
        for n in sub_g.nodes():
            node_data = self.graph.nodes[n]
            f = node_data.get("file", node_data.get("file_path"))
            if not f:
                parts = str(n).split(".")
                f = ("/".join(parts[:-1]) + ".py") if len(parts) >= 2 else (parts[0] + ".py")
            files_touched.add(f)

        nodes_payload = [
            {"id": n, **self.graph.nodes[n]} for n in sub_g.nodes()
        ]
        edges_payload = [
            {"source": u, "target": v, "key": k, **d}
            for u, v, k, d in sub_g.edges(keys=True, data=True)
        ]

        return {
            "graph_file": self.graph_path.name,
            "anchor_node": anchor_id,
            "num_nodes": len(nodes_payload),
            "num_files_touched": len(files_touched),
            "files": sorted(list(files_touched)),
            "nodes": nodes_payload,
            "edges": edges_payload,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample multi-node subgraphs from competition graphs."
    )
    parser.add_argument("--graphs-dir", type=Path, default=Path("data/raw/graphs"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/raw/embeddings"))
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/synthetic_tasks/sampled_subgraphs.jsonl"),
    )
    parser.add_argument("--samples-per-graph", type=int, default=10)
    parser.add_argument(
        "--exclude-prefix",
        type=str,
        default="httpx",
        help="Holdout repository prefix to skip (default: httpx)",
    )
    args = parser.parse_args()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    graph_files = sorted(args.graphs_dir.glob("*.json"))

    if not graph_files:
        logger.warning("No .json graph files found in %s", args.graphs_dir)
        return

    total_written = 0
    with args.output_file.open("w", encoding="utf-8") as out_f:
        for g_path in graph_files:
            if args.exclude_prefix and g_path.name.lower().startswith(args.exclude_prefix.lower()):
                continue
            emb_path = args.embeddings_dir / g_path.with_suffix(".npz").name
            sampler = RepoGraphSampler(g_path, emb_path if emb_path.exists() else None)
            anchors = sampler.find_candidate_anchors()
            if not anchors:
                continue

            chosen = random.sample(anchors, min(len(anchors), args.samples_per_graph))
            for anchor in chosen:
                sub = sampler.sample_subgraph(anchor)
                # Filter for subgraphs with at least 3 nodes to ensure multi-node reasoning
                if sub["num_nodes"] >= 3:
                    out_f.write(json.dumps(sub) + "\n")
                    total_written += 1

    logger.info("Sampled %d multi-node subgraphs -> %s", total_written, args.output_file)


if __name__ == "__main__":
    main()
