#!/usr/bin/env python3
"""
Milestone 1: Unified Data Loader for Competition Tasks, Code Graphs, and Node Embeddings.

Loads and maps:
  1. tasks.jsonl (or train_tasks.jsonl / val_ood_tasks.jsonl)
  2. Directed multigraphs (NetworkX node-link JSON in graphs/)
  3. 256-dimensional node embeddings (.npz in embeddings/)
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_loader")


@dataclass
class Task:
    """Represents a single bug-fixing or feature benchmark task."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str = ""
    patch: str = ""
    test_patch: str = ""
    created_at: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            instance_id=data.get("instance_id", ""),
            repo=data.get("repo", ""),
            base_commit=data.get("base_commit", ""),
            problem_statement=data.get("problem_statement", ""),
            hints_text=data.get("hints_text", "") or "",
            patch=data.get("patch", "") or "",
            test_patch=data.get("test_patch", "") or "",
            created_at=data.get("created_at", "") or "",
            raw=data,
        )

    @property
    def repo_short(self) -> str:
        """Extract short name from 'owner/repo' (e.g. 'fastapi/fastapi' -> 'fastapi')."""
        return self.repo.split("/")[-1] if "/" in self.repo else self.repo


class RepoGraph:
    """
    Wraps a NetworkX directed multigraph and corresponding 256-dim node embeddings.
    """

    def __init__(
        self,
        graph_path: Path,
        embedding_path: Optional[Path] = None,
    ):
        self.graph_path = Path(graph_path)
        self.embedding_path = Path(embedding_path) if embedding_path else None
        self.graph: nx.MultiDiGraph = self._load_graph(self.graph_path)
        self.embeddings: Dict[str, np.ndarray] = (
            self._load_embeddings(self.embedding_path) if self.embedding_path else {}
        )

    @staticmethod
    def _load_graph(path: Path) -> nx.MultiDiGraph:
        """Loads NetworkX node-link JSON graph, handling schema variations."""
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle compatibility across networkx versions (links vs edges key)
        edges_key = "edges" if "edges" in data else "links"
        if "edges" in data and "links" not in data:
            data["links"] = data["edges"]
        elif "links" in data and "edges" not in data:
            data["edges"] = data["links"]

        try:
            # Standard node-link format in NetworkX
            try:
                g = json_graph.node_link_graph(data, directed=True, multigraph=True, edges=edges_key)
            except TypeError:
                g = json_graph.node_link_graph(data, directed=True, multigraph=True)

            if not isinstance(g, nx.MultiDiGraph):
                g = nx.MultiDiGraph(g)
            return g
        except Exception as e:
            logger.warning(
                "Standard node_link_graph failed for %s (%s). Falling back to manual construction.",
                path.name,
                e,
            )
            g = nx.MultiDiGraph()
            nodes = data.get("nodes", [])
            for n in nodes:
                node_id = n.get("id") if isinstance(n, dict) else str(n)
                attrs = n if isinstance(n, dict) else {}
                g.add_node(node_id, **attrs)

            edges = data.get("links", data.get("edges", []))
            for edge in edges:
                if isinstance(edge, dict):
                    src = edge.get("source")
                    dst = edge.get("target")
                    key = edge.get("key", None)
                    attr_dict = {k: v for k, v in edge.items() if k not in ("source", "target", "key")}
                    g.add_edge(src, dst, key=key, **attr_dict)
            return g

    @staticmethod
    def _load_embeddings(path: Optional[Path]) -> Dict[str, np.ndarray]:
        """Loads 256-dimensional vectors from an .npz archive."""
        if not path or not path.exists():
            return {}

        archive = np.load(path, allow_pickle=True)
        # Case A: Separate arrays for node_ids and embeddings matrix
        if "node_ids" in archive and "embeddings" in archive:
            node_ids = archive["node_ids"].tolist()
            vectors = archive["embeddings"]
            return {str(nid): np.asarray(vec, dtype=np.float32) for nid, vec in zip(node_ids, vectors)}

        # Case B: Key-value archive where each file is a node_id
        return {k: np.asarray(archive[k], dtype=np.float32) for k in archive.files}

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Returns node attributes if present."""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Returns embedding vector for node_id if present."""
        return self.embeddings.get(node_id)

    def get_neighbors(
        self, node_id: str, direction: str = "both"
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Retrieves structural neighbors and edge attributes.
        direction: 'in', 'out', or 'both'
        """
        if node_id not in self.graph:
            return []

        neighbors: List[Tuple[str, Dict[str, Any]]] = []
        if direction in ("out", "both"):
            for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                neighbors.append((target, {"direction": "out", **edge_data}))
        if direction in ("in", "both"):
            for source, _, edge_data in self.graph.in_edges(node_id, data=True):
                neighbors.append((source, {"direction": "in", **edge_data}))
        return neighbors

    def summary(self) -> Dict[str, Any]:
        """Returns structural statistics for this repository snapshot graph."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()

        node_types: Dict[str, int] = {}
        files_set: Set[str] = set()
        test_nodes_count = 0

        for nid, attrs in self.graph.nodes(data=True):
            ntype = str(attrs.get("type", attrs.get("kind", "unknown"))).lower()
            node_types[ntype] = node_types.get(ntype, 0) + 1

            file_path = attrs.get("file", attrs.get("file_path", ""))
            if file_path:
                files_set.add(file_path)
                if "test" in file_path.lower():
                    test_nodes_count += 1

        return {
            "graph_file": self.graph_path.name,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_embeddings": len(self.embeddings),
            "unique_files": len(files_set),
            "test_nodes": test_nodes_count,
            "node_type_breakdown": node_types,
        }


class RepoDataManager:
    """
    Coordinates access to competition tasks, code graphs, and embeddings.
    """

    def __init__(
        self,
        tasks_file: Path,
        graphs_dir: Optional[Path] = None,
        embeddings_dir: Optional[Path] = None,
    ):
        self.tasks_file = Path(tasks_file)
        self.graphs_dir = Path(graphs_dir) if graphs_dir else self.tasks_file.parent / "graphs"
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else self.tasks_file.parent / "embeddings"

        self.tasks: List[Task] = self._load_tasks(self.tasks_file)
        self._graph_cache: Dict[str, RepoGraph] = {}

        # Index graph files by commit hash for quick lookup
        self._graph_index = self._index_files(self.graphs_dir, extension=".json")
        self._embedding_index = self._index_files(self.embeddings_dir, extension=".npz")

    @staticmethod
    def _load_tasks(path: Path) -> List[Task]:
        if not path.exists():
            raise FileNotFoundError(f"Tasks file not found at: {path}")

        tasks = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tasks.append(Task.from_dict(json.loads(line)))
        return tasks

    @staticmethod
    def _index_files(directory: Optional[Path], extension: str) -> Dict[str, Path]:
        """Indexes files in a directory by commit SHA or stem."""
        index: Dict[str, Path] = {}
        if not directory or not directory.exists():
            return index

        for f in directory.glob(f"*{extension}"):
            index[f.name] = f
            index[f.stem] = f
            # Also index by commit if filename pattern is <repo>_<commit>
            parts = f.stem.split("_")
            if len(parts) >= 2:
                commit_part = parts[-1]
                index[commit_part] = f
        return index

    def get_task(self, instance_id: str) -> Optional[Task]:
        """Finds task by instance_id."""
        for t in self.tasks:
            if t.instance_id == instance_id:
                return t
        return None

    def find_graph_path(self, task: Task) -> Optional[Path]:
        """Resolves the graph JSON path for a task."""
        # Candidate 1: direct commit index match
        if task.base_commit in self._graph_index:
            return self._graph_index[task.base_commit]

        # Candidate 2: <repo_short>_<base_commit>.json
        cand_name = f"{task.repo_short}_{task.base_commit}.json"
        if cand_name in self._graph_index:
            return self._graph_index[cand_name]

        # Candidate 3: search prefix of commit
        for stem, path in self._graph_index.items():
            if task.base_commit.startswith(stem) or stem.endswith(task.base_commit):
                return path

        return None

    def find_embedding_path(self, task: Task) -> Optional[Path]:
        """Resolves the embedding NPZ path for a task."""
        if task.base_commit in self._embedding_index:
            return self._embedding_index[task.base_commit]

        cand_name = f"{task.repo_short}_{task.base_commit}.npz"
        if cand_name in self._embedding_index:
            return self._embedding_index[cand_name]

        for stem, path in self._embedding_index.items():
            if task.base_commit.startswith(stem) or stem.endswith(task.base_commit):
                return path

        return None

    def load_graph_for_task(self, task: Task) -> Optional[RepoGraph]:
        """Loads and caches the RepoGraph for a given task."""
        cache_key = f"{task.repo_short}_{task.base_commit}"
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]

        g_path = self.find_graph_path(task)
        if not g_path:
            logger.warning("No graph JSON found for task %s (commit %s)", task.instance_id, task.base_commit)
            return None

        e_path = self.find_embedding_path(task)
        repo_graph = RepoGraph(graph_path=g_path, embedding_path=e_path)
        self._graph_cache[cache_key] = repo_graph
        return repo_graph

    def audit_coverage(self) -> Dict[str, Any]:
        """
        Audits how many tasks have matching graph JSON and embedding NPZ files.
        """
        matched_graphs = 0
        matched_embeddings = 0
        missing_graphs: List[str] = []
        missing_embeddings: List[str] = []

        for task in self.tasks:
            g_path = self.find_graph_path(task)
            e_path = self.find_embedding_path(task)

            if g_path:
                matched_graphs += 1
            else:
                missing_graphs.append(task.instance_id)

            if e_path:
                matched_embeddings += 1
            else:
                missing_embeddings.append(task.instance_id)

        return {
            "total_tasks": len(self.tasks),
            "matched_graphs": matched_graphs,
            "missing_graphs_count": len(missing_graphs),
            "matched_embeddings": matched_embeddings,
            "missing_embeddings_count": len(missing_embeddings),
            "missing_graphs_instances": missing_graphs[:10],
            "missing_embeddings_instances": missing_embeddings[:10],
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 1: Ingest and inspect tasks, code graphs, and embeddings."
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=Path("data/raw/tasks.jsonl"),
        help="Path to tasks.jsonl (or train_tasks.jsonl / val_ood_tasks.jsonl)",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("data/raw/graphs"),
        help="Path to graphs directory (default: data/raw/graphs)",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("data/raw/embeddings"),
        help="Path to embeddings directory (default: data/raw/embeddings)",
    )
    parser.add_argument(
        "--sample-instance",
        type=str,
        default=None,
        help="Optional instance_id to inspect in detail",
    )
    args = parser.parse_args()

    if not args.tasks_file.exists():
        logger.error("Tasks file not found: %s", args.tasks_file)
        logger.info("Place competition data in data/raw/ or specify --tasks-file.")
        return

    manager = RepoDataManager(
        tasks_file=args.tasks_file,
        graphs_dir=args.graphs_dir,
        embeddings_dir=args.embeddings_dir,
    )

    logger.info("Loaded %d tasks from %s", len(manager.tasks), args.tasks_file)
    audit = manager.audit_coverage()
    logger.info(
        "Coverage Audit: %d/%d graphs matched, %d/%d embeddings matched.",
        audit["matched_graphs"],
        audit["total_tasks"],
        audit["matched_embeddings"],
        audit["total_tasks"],
    )

    if args.sample_instance:
        task = manager.get_task(args.sample_instance)
        if task:
            logger.info("Inspecting Task: %s", task.instance_id)
            logger.info("Repo: %s | Base Commit: %s", task.repo, task.base_commit)
            logger.info("Problem Statement Preview: %s...", task.problem_statement[:200].replace("\n", " "))
            repo_graph = manager.load_graph_for_task(task)
            if repo_graph:
                logger.info("Graph Summary: %s", json.dumps(repo_graph.summary(), indent=2))
        else:
            logger.warning("Instance %s not found in tasks.", args.sample_instance)
    elif manager.tasks:
        first_task = manager.tasks[0]
        logger.info("First Task Preview: %s (%s)", first_task.instance_id, first_task.repo)
        repo_graph = manager.load_graph_for_task(first_task)
        if repo_graph:
            logger.info("Graph Summary: %s", json.dumps(repo_graph.summary(), indent=2))


if __name__ == "__main__":
    main()
