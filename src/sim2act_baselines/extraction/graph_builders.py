from __future__ import annotations

import itertools
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.distance import cdist

from .extraction_result import GraphBuildMetadata


def _validated_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(array).all():
        raise ValueError("points contain non-finite coordinates")
    return array


class _UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, item: int) -> int:
        root = item
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while item != root:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return True


def _canonical_index_edges(edges: np.ndarray) -> np.ndarray:
    array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(array):
        return np.empty((0, 2), dtype=np.int64)
    array = np.sort(array, axis=1)
    return np.unique(array, axis=0)


def _kruskal(
    points: np.ndarray,
    candidate_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edges = _canonical_index_edges(candidate_edges)
    if len(points) <= 1:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    if not len(edges):
        raise ValueError("candidate graph is disconnected")
    delta = points[edges[:, 0]] - points[edges[:, 1]]
    weights = np.linalg.norm(delta, axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0], weights))
    union_find = _UnionFind(len(points))
    selected: list[int] = []
    for index in order:
        left, right = edges[index]
        if union_find.union(int(left), int(right)):
            selected.append(int(index))
            if len(selected) == len(points) - 1:
                break
    if len(selected) != len(points) - 1:
        raise ValueError("candidate graph is disconnected")
    selected_array = np.asarray(selected, dtype=np.int64)
    return edges[selected_array], weights[selected_array]


def _graph_from_tree(points: np.ndarray, edges: np.ndarray, weights: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    for index, position in enumerate(points):
        graph.add_node(int(index), pos=np.asarray(position, dtype=np.float64))
    for (left, right), weight in zip(edges, weights):
        graph.add_edge(int(left), int(right), weight=float(weight))
    return graph


def build_original_complete_graph_mst(
    points: np.ndarray,
) -> tuple[nx.Graph, GraphBuildMetadata]:
    """Reproduce ENSTRECT's complete-graph construction for controlled tests only."""
    points = _validated_points(points)
    count = len(points)
    if count <= 1:
        graph = _graph_from_tree(points, np.empty((0, 2), dtype=np.int64), np.empty(0))
        return graph, GraphBuildMetadata(
            implementation="original_upstream",
            equivalence_status="verified_exact",
            algorithm="complete_cdist_networkx_kruskal",
            input_nodes=count,
            unique_nodes=len(np.unique(points, axis=0)),
            candidate_edges=0,
            mst_edges=0,
            affine_rank=0,
        )
    distances = cdist(points, points)
    combinations = np.asarray(list(itertools.combinations(range(count), 2)), dtype=np.int64)
    entries = np.append(
        combinations,
        distances[combinations[:, 0], combinations[:, 1]][:, None],
        axis=-1,
    )
    complete = nx.Graph()
    complete.add_weighted_edges_from(entries)
    nx.set_node_attributes(complete, {i: points[i] for i in range(count)}, "pos")
    graph = nx.minimum_spanning_tree(complete)
    return graph, GraphBuildMetadata(
        implementation="original_upstream",
        equivalence_status="verified_exact",
        algorithm="complete_cdist_networkx_kruskal",
        input_nodes=count,
        unique_nodes=len(np.unique(points, axis=0)),
        candidate_edges=count * (count - 1) // 2,
        mst_edges=len(graph.edges),
        affine_rank=int(np.linalg.matrix_rank(points - points[:1])),
    )


def _affine_projection(points: np.ndarray) -> tuple[np.ndarray, int]:
    if len(points) <= 1:
        return np.empty((len(points), 0), dtype=np.float64), 0
    centered = points - points[0]
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0:
        return np.empty((len(points), 0), dtype=np.float64), 0
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    projected = centered @ vh[:rank].T
    reconstructed = projected @ vh[:rank]
    residual = float(np.max(np.abs(reconstructed - centered))) if len(centered) else 0.0
    scale = max(1.0, float(np.max(np.abs(centered))))
    if residual > 256 * np.finfo(np.float64).eps * scale:
        # Do not discard a numerically small but real distance dimension.
        rank = min(3, max(rank + 1, 1))
        projected = centered @ vh[:rank].T
    return projected, rank


def _delaunay_candidate_edges(projected: np.ndarray, rank: int) -> np.ndarray:
    count = len(projected)
    if count <= 1:
        return np.empty((0, 2), dtype=np.int64)
    if rank == 1:
        order = np.argsort(projected[:, 0], kind="stable")
        return _canonical_index_edges(np.column_stack((order[:-1], order[1:])))
    triangulation = Delaunay(projected)
    present = np.unique(triangulation.simplices)
    if len(present) != count or not np.array_equal(present, np.arange(count)):
        raise QhullError("Delaunay omitted one or more unique input vertices")
    simplex_size = triangulation.simplices.shape[1]
    pieces = [
        triangulation.simplices[:, [left, right]]
        for left, right in itertools.combinations(range(simplex_size), 2)
    ]
    return _canonical_index_edges(np.concatenate(pieces, axis=0))


def _dense_prim_exact(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact Euclidean MST with O(N) auxiliary memory and O(N^2) time."""
    count = len(points)
    if count <= 1:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    in_tree = np.zeros(count, dtype=bool)
    best = np.full(count, np.inf, dtype=np.float64)
    parent = np.full(count, -1, dtype=np.int64)
    best[0] = 0.0
    chosen: list[tuple[int, int]] = []
    chosen_weights: list[float] = []
    for _ in range(count):
        candidates = np.flatnonzero(~in_tree)
        current = int(candidates[np.argmin(best[candidates])])
        if not np.isfinite(best[current]):
            raise ValueError("complete Euclidean graph unexpectedly disconnected")
        in_tree[current] = True
        if parent[current] >= 0:
            left, right = sorted((int(parent[current]), current))
            chosen.append((left, right))
            chosen_weights.append(float(best[current]))
        remaining = np.flatnonzero(~in_tree)
        if len(remaining):
            distances = np.linalg.norm(points[remaining] - points[current], axis=1)
            improve = distances < best[remaining]
            ties = (distances == best[remaining]) & (current < parent[remaining])
            update = improve | ties
            best[remaining[update]] = distances[update]
            parent[remaining[update]] = current
    return np.asarray(chosen, dtype=np.int64), np.asarray(chosen_weights, dtype=np.float64)


def build_exact_euclidean_mst(
    points: np.ndarray,
    *,
    allow_dense_prim_fallback: bool = True,
) -> tuple[nx.Graph, GraphBuildMetadata]:
    """Build an exact Euclidean MST without materializing the complete graph.

    The Euclidean MST is contained in the Delaunay graph. Exact duplicate points
    are joined by deterministic zero-length edges. If Qhull cannot represent a
    degenerate input, a dense-distance *streaming* Prim pass preserves the exact
    objective with O(N) auxiliary memory.
    """
    points = _validated_points(points)
    count = len(points)
    if count <= 1:
        graph = _graph_from_tree(points, np.empty((0, 2), dtype=np.int64), np.empty(0))
        return graph, GraphBuildMetadata(
            implementation="exact_memory_safe_local",
            equivalence_status="verified_exact",
            algorithm="trivial",
            input_nodes=count,
            unique_nodes=count,
            candidate_edges=0,
            mst_edges=0,
            affine_rank=0,
        )

    unique_points, first_indices, inverse = np.unique(
        points,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    order = np.argsort(first_indices, kind="stable")
    unique_points = unique_points[order]
    first_indices = first_indices[order]
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order), dtype=np.int64)
    inverse = remap[inverse]
    projected, rank = _affine_projection(unique_points)
    fallback_reason: str | None = None
    algorithm = "delaunay_kruskal"
    try:
        unique_candidates = _delaunay_candidate_edges(projected, rank)
        unique_tree, _unique_weights = _kruskal(unique_points, unique_candidates)
    except (QhullError, ValueError) as error:
        if not allow_dense_prim_fallback:
            raise
        fallback_reason = f"{type(error).__name__}: {error}"
        algorithm = "dense_streaming_prim_fallback"
        unique_candidates = np.empty((0, 2), dtype=np.int64)
        unique_tree, _unique_weights = _dense_prim_exact(unique_points)

    mapped_unique_tree = first_indices[unique_tree] if len(unique_tree) else np.empty((0, 2), dtype=np.int64)
    duplicate_edges: list[tuple[int, int]] = []
    for unique_id, representative in enumerate(first_indices):
        members = np.flatnonzero(inverse == unique_id)
        for member in members:
            if int(member) != int(representative):
                duplicate_edges.append((int(representative), int(member)))
    pieces: list[np.ndarray] = []
    if len(mapped_unique_tree):
        pieces.append(mapped_unique_tree)
    if duplicate_edges:
        pieces.append(np.asarray(duplicate_edges, dtype=np.int64))
    all_tree_edges = np.concatenate(pieces, axis=0) if pieces else np.empty((0, 2), dtype=np.int64)
    all_tree_edges = _canonical_index_edges(all_tree_edges)
    tree_weights = np.linalg.norm(
        points[all_tree_edges[:, 0]] - points[all_tree_edges[:, 1]], axis=1
    ) if len(all_tree_edges) else np.empty(0, dtype=np.float64)
    graph = _graph_from_tree(points, all_tree_edges, tree_weights)
    candidate_count = len(unique_candidates) + len(duplicate_edges)
    return graph, GraphBuildMetadata(
        implementation="exact_memory_safe_local",
        equivalence_status="verified_exact",
        algorithm=algorithm,
        input_nodes=count,
        unique_nodes=len(unique_points),
        candidate_edges=int(candidate_count),
        mst_edges=len(graph.edges),
        affine_rank=rank,
        fallback_reason=fallback_reason,
    )


def partition_furcations(graph: nx.Graph) -> nx.Graph:
    """Apply ENSTRECT's branch-node duplication step verbatim."""
    graph = graph.copy()
    degrees = np.asarray([value for _node, value in graph.degree()], dtype=np.int64)
    furcations = np.where(2 < degrees)[0]
    for furcation_node in furcations:
        neighbours = list(graph.neighbors(int(furcation_node)))
        for neighbour in neighbours:
            new_node = int(np.max(list(graph.nodes))) + 1
            graph.add_node(new_node, pos=graph.nodes[int(furcation_node)]["pos"])
            graph.add_edge(new_node, neighbour)
            graph.remove_edge(int(furcation_node), neighbour)
        graph.remove_node(int(furcation_node))
    return graph


def canonical_edge_coordinates(graph: nx.Graph, decimals: int = 12) -> np.ndarray:
    """Return edge endpoint coordinates in an order-independent representation."""
    rows: list[np.ndarray] = []
    for left, right in graph.edges:
        a = np.round(np.asarray(graph.nodes[left]["pos"], dtype=np.float64), decimals)
        b = np.round(np.asarray(graph.nodes[right]["pos"], dtype=np.float64), decimals)
        if tuple(b.tolist()) < tuple(a.tolist()):
            a, b = b, a
        rows.append(np.concatenate((a, b)))
    if not rows:
        return np.empty((0, 6), dtype=np.float64)
    array = np.asarray(rows, dtype=np.float64)
    keys = tuple(array[:, index] for index in reversed(range(array.shape[1])))
    return array[np.lexsort(keys)]
