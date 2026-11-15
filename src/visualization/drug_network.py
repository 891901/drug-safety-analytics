"""Drug co-reporting network construction and interactive visualization.

Builds an undirected network from the **project's own drug co-occurrence
results** (real FAERS 2026 Q1/Q2 data) and describes how the drugs in it relate:

* every **drug is a node**;
* every **co-reported drug pair is an edge**;
* the **edge weight is the co-occurrence frequency** (the number of FAERS
  reports in which the two drugs appear together - the ``reports`` column of
  ``cooccurrence_top_pairs.csv``).

Measures computed per node
--------------------------
========================  ====================================================
``degree``                number of distinct drugs the drug was co-reported with
``weighted_degree``       sum of the weights of the incident edges (total
                          co-reports over all partners)
``degree_centrality``     ``degree / (n - 1)``
``betweenness_centrality``share of shortest paths that pass through the node,
                          normalised by NetworkX conventions (undirected,
                          unweighted hop distance)
========================  ====================================================

Betweenness uses **hop distance**, not the co-report weight: the weight is a
similarity (more co-reports = a stronger tie), not a distance, so treating it
as a path length would be meaningless. This choice is stated in the report.

Communities
-----------
Dense sub-groups of drugs are detected with the Louvain modularity algorithm
(seeded for reproducibility) and scored with the modularity of the resulting
partition. Community detection is skipped, with the reason recorded in the
report, when the network is larger than :data:`COMMUNITY_NODE_LIMIT`.

What is read, and why
---------------------
The only input is the co-occurrence module's own output
(``data/analytics/drug_cooccurrence/cooccurrence_top_pairs.csv``). No raw FAERS
file is read, no synthetic edge is invented, and no raw dataset is modified.
Because that table holds the strongest pairs only (it is capped upstream), the
network is a **screened sub-network** of the full co-reporting graph; this
limitation is stated in the report rather than hidden.

!!!!  IMPORTANT INTERPRETATION LIMIT  !!!!
An edge only means two drugs were reported together in the same FAERS report.
It is **NOT** a confirmed drug-drug interaction, **NOT** evidence of a shared
adverse event or indication, and **NOT** evidence of causation. FAERS is a
spontaneous reporting system with no exposure denominator and heavy reporting
bias, so a strong edge can simply reflect co-prescribing, a treated condition,
or reporting habit.

Environment constraints honoured: NetworkX + Plotly only, no pandas, no
PyArrow, no Spark. The input is streamed with the standard library ``csv``
module (500 rows in the current run), so nothing large is held in memory.

Reusable API for the Streamlit dashboard
----------------------------------------
The figure builders return ``plotly.graph_objects.Figure`` objects and never
touch Streamlit, so the dashboard can import and call them directly::

    from src.visualization.drug_network import (
        load_analysis, network_figure, top_connected_figure, relationships_figure,
    )

    analysis = load_analysis()                 # metrics, communities, layout
    network_figure(analysis)                   # interactive drug network
    top_connected_figure(analysis)             # top 20 connected drugs
    relationships_figure(analysis)             # drug relationship analysis

Usage (command line)::

    python -m src.visualization.drug_network
    python -m src.visualization.drug_network --max-nodes 40 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config import ConfigError, get_path, get_visualization_config
from src.utils.logging_setup import get_logger

__all__ = [
    "DrugNetworkError",
    "NETWORK_DISCLAIMER",
    "NETWORK_GRAIN",
    "EdgeRecord",
    "NodeMetric",
    "Community",
    "NetworkAnalysis",
    "load_edge_records",
    "build_graph",
    "compute_node_metrics",
    "detect_communities",
    "analyse",
    "load_analysis",
    "network_figure",
    "top_connected_figure",
    "relationships_figure",
    "write_outputs",
    "render_report",
    "run",
    "main",
]

#: Mandatory caution, echoed in the report and the JSON summary.
NETWORK_DISCLAIMER = (
    "An edge only means the two drugs were reported together in the same FAERS "
    "report. It is NOT a confirmed drug-drug interaction, NOT evidence of a "
    "shared adverse event or indication, and NOT evidence of causation. A "
    "strong edge can simply reflect co-prescribing, the treated condition, or "
    "reporting habit."
)

#: Grain statement reused in the report.
NETWORK_GRAIN = (
    "Node = one standardised drug name (DRUGNAME_STD). Edge = one unordered "
    "co-reported pair {drug_a, drug_b} with drug_a < drug_b. Edge weight = the "
    "number of FAERS reports containing both drugs."
)

#: Columns the reader expects in the co-occurrence pair table.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "drug_a",
    "drug_b",
    "reports",
    "support",
    "confidence_a_to_b",
    "confidence_b_to_a",
    "lift",
    "reports_a",
    "reports_b",
)

#: Output sizing.
TOP_CONNECTED_REPORT = 20
TOP_BETWEENNESS_REPORT = 20
TOP_COMMUNITIES_REPORT = 15
TOP_NODES_JSON = 50
TOP_EDGES_JSON = 20
TOP_COMMUNITY_MEMBERS_JSON = 10

#: Number of edge-weight classes used to draw variable-width edges. Plotly
#: cannot vary the line width inside a single trace, so edges are binned.
EDGE_WIDTH_BINS = 4

#: Betweenness is verified against an independent plain-Python Brandes
#: computation up to this node count (O(V * E)); above it only the bounds and
#: determinism are verified, and the report says so.
BETWEENNESS_RECOMPUTE_LIMIT = 2000

#: Community detection is skipped above this node count (feasibility guard).
COMMUNITY_NODE_LIMIT = 5000

#: Discrete palette for communities.
COMMUNITY_COLORS: Tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
)


class DrugNetworkError(RuntimeError):
    """Raised when the network cannot be built from the co-occurrence results."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class EdgeRecord:
    """One co-reported drug pair read from the co-occurrence output."""

    drug_a: str
    drug_b: str
    reports: int
    support: float
    confidence_a_to_b: float
    confidence_b_to_a: float
    lift: float
    reports_a: int
    reports_b: int
    passes_thresholds: bool = False

    @property
    def key(self) -> Tuple[str, str]:
        """Canonical unordered key (the source already guarantees drug_a < drug_b)."""
        return (self.drug_a, self.drug_b)


@dataclass
class NodeMetric:
    """Centrality measures for one drug in the network."""

    drug: str
    degree: int
    weighted_degree: float
    degree_centrality: float
    betweenness_centrality: float
    community: int = -1


@dataclass
class Community:
    """One detected community of drugs."""

    community_id: int
    size: int
    total_weight: float
    internal_edges: int
    top_drug: str
    drugs: List[str] = field(default_factory=list)


@dataclass
class NetworkAnalysis:
    """Every artefact produced by one network run."""

    generated_at: str = ""
    input_path: str = ""
    input_rows: int = 0
    min_edge_weight: float = 0.0
    edges_below_threshold: int = 0
    self_loops_skipped: int = 0
    duplicate_edges_merged: int = 0
    reversed_edges_fixed: int = 0

    node_count: int = 0
    edge_count: int = 0
    total_weight: float = 0.0
    connected_components: int = 0
    largest_component: int = 0
    density: float = 0.0

    display_node_limit: int = 0
    display_edge_limit: int = 0
    display_nodes: int = 0
    display_edges: int = 0

    layout_name: str = "spring"
    layout_seed: int = 42
    community_count: int = 0
    modularity: float = 0.0
    community_note: str = ""

    nodes: List[NodeMetric] = field(default_factory=list)
    edges: List[EdgeRecord] = field(default_factory=list)
    communities: List[Community] = field(default_factory=list)
    graph: Optional[nx.Graph] = None
    display_graph: Optional[nx.Graph] = None
    display_layout: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    checks: List[Tuple[str, bool, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    output_paths: Dict[str, str] = field(default_factory=dict)

    # -- convenience views ---------------------------------------------------
    def ranked_by_weighted_degree(self) -> List[NodeMetric]:
        """Nodes ordered by total co-reports, then degree, then drug name."""
        return sorted(
            self.nodes,
            key=lambda n: (-n.weighted_degree, -n.degree, n.drug),
        )

    def ranked_by_betweenness(self) -> List[NodeMetric]:
        """Nodes ordered by betweenness centrality, then weighted degree."""
        return sorted(
            self.nodes,
            key=lambda n: (-n.betweenness_centrality, -n.weighted_degree, n.drug),
        )

    def as_summary(self) -> Dict[str, Any]:
        """Bounded JSON payload (never the whole node/edge table)."""
        return {
            "generated_at": self.generated_at,
            "disclaimer": NETWORK_DISCLAIMER,
            "grain": NETWORK_GRAIN,
            "input": {
                "path": self.input_path,
                "rows": self.input_rows,
                "min_edge_weight": self.min_edge_weight,
                "edges_below_threshold": self.edges_below_threshold,
                "self_loops_skipped": self.self_loops_skipped,
                "duplicate_edges_merged": self.duplicate_edges_merged,
                "reversed_edges_fixed": self.reversed_edges_fixed,
            },
            "network": {
                "nodes": self.node_count,
                "edges": self.edge_count,
                "total_weight": self.total_weight,
                "connected_components": self.connected_components,
                "largest_component": self.largest_component,
                "density": self.density,
                "display_nodes": self.display_nodes,
                "display_edges": self.display_edges,
                "display_node_limit": self.display_node_limit,
                "display_edge_limit": self.display_edge_limit,
                "layout": self.layout_name,
                "layout_seed": self.layout_seed,
            },
            "communities": {
                "count": self.community_count,
                "modularity": self.modularity,
                "note": self.community_note,
                "top": [
                    {
                        "community_id": community.community_id,
                        "size": community.size,
                        "internal_edges": community.internal_edges,
                        "total_weight": community.total_weight,
                        "top_drug": community.top_drug,
                        "drugs": community.drugs[:TOP_COMMUNITY_MEMBERS_JSON],
                    }
                    for community in self.communities[:TOP_COMMUNITIES_REPORT]
                ],
            },
            "top_connected_drugs": [
                self._node_dict(node)
                for node in self.ranked_by_weighted_degree()[:TOP_CONNECTED_REPORT]
            ],
            "top_betweenness_drugs": [
                self._node_dict(node)
                for node in self.ranked_by_betweenness()[:TOP_BETWEENNESS_REPORT]
            ],
            "top_edges": [
                self._edge_dict(edge) for edge in self.edges[:TOP_EDGES_JSON]
            ],
            "checks": [
                {"name": name, "passed": ok, "detail": detail}
                for name, ok, detail in self.checks
            ],
            "errors": self.errors,
            "outputs": self.output_paths,
        }

    @staticmethod
    def _node_dict(node: NodeMetric) -> Dict[str, Any]:
        return {
            "drug": node.drug,
            "degree": node.degree,
            "weighted_degree": node.weighted_degree,
            "degree_centrality": node.degree_centrality,
            "betweenness_centrality": node.betweenness_centrality,
            "community": node.community,
        }

    @staticmethod
    def _edge_dict(edge: EdgeRecord) -> Dict[str, Any]:
        return {
            "drug_a": edge.drug_a,
            "drug_b": edge.drug_b,
            "reports": edge.reports,
            "lift": edge.lift,
            "support": edge.support,
            "passes_thresholds": edge.passes_thresholds,
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _clip(text: str, width: int) -> str:
    """Truncate ``text`` to ``width`` characters, adding an ellipsis when cut."""
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def _to_float(value: Any, default: float = 0.0) -> float:
    """Parse a CSV cell defensively; a blank or unparseable cell uses ``default``."""
    try:
        text = "" if value is None else str(value).strip()
        if text == "":
            return default
        result = float(text)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    """Parse an integer cell defensively."""
    parsed = _to_float(value, float(default))
    return int(parsed) if math.isfinite(parsed) else default


def _to_bool(value: Any) -> bool:
    """Parse a boolean cell written by the co-occurrence module."""
    return str(value).strip().lower() in {"true", "1", "yes"}


# ---------------------------------------------------------------------------
# Input reading (streamed, standard library only)
# ---------------------------------------------------------------------------
def load_edge_records(path: Path) -> List[EdgeRecord]:
    """Stream the co-occurrence pair table into :class:`EdgeRecord` objects.

    Raises:
        DrugNetworkError: the file is missing, empty or lacks the required
            columns.
    """
    if not path.exists():
        raise DrugNetworkError(
            f"co-occurrence pair table not found: {path}. Run "
            "'python -m src.analytics.drug_cooccurrence' first."
        )

    records: List[EdgeRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in header]
        if missing:
            raise DrugNetworkError(
                f"{path.name} is missing required column(s): {', '.join(missing)}"
            )
        for row in reader:
            drug_a = str(row.get("drug_a", "")).strip()
            drug_b = str(row.get("drug_b", "")).strip()
            if not drug_a or not drug_b:
                continue
            records.append(
                EdgeRecord(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    reports=_to_int(row.get("reports")),
                    support=_to_float(row.get("support")),
                    confidence_a_to_b=_to_float(row.get("confidence_a_to_b")),
                    confidence_b_to_a=_to_float(row.get("confidence_b_to_a")),
                    lift=_to_float(row.get("lift")),
                    reports_a=_to_int(row.get("reports_a")),
                    reports_b=_to_int(row.get("reports_b")),
                    passes_thresholds=_to_bool(row.get("passes_thresholds")),
                )
            )

    if not records:
        raise DrugNetworkError(f"{path.name} contains no drug pairs.")
    return records


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
@dataclass
class _GraphBuild:
    """A built graph plus the cleanup counters observed while building it."""

    graph: nx.Graph
    edges: List[EdgeRecord]
    edges_below_threshold: int
    self_loops_skipped: int
    duplicate_edges_merged: int
    reversed_edges_fixed: int


def build_graph(
    records: Sequence[EdgeRecord],
    *,
    min_edge_weight: float = 0.0,
) -> _GraphBuild:
    """Build the undirected co-reporting graph from the pair records.

    Edges lighter than ``min_edge_weight`` are dropped, self-loops are skipped
    and a reversed duplicate (``B, A`` when ``A, B`` already exists) is merged
    into the existing edge instead of being added twice.
    """
    graph = nx.Graph()
    kept: List[EdgeRecord] = []
    seen: Dict[Tuple[str, str], int] = {}
    below = self_loops = duplicates = reversed_fixed = 0

    for record in records:
        if record.reports < min_edge_weight:
            below += 1
            continue
        if record.drug_a == record.drug_b:
            self_loops += 1
            continue

        key = (
            (record.drug_a, record.drug_b)
            if record.drug_a < record.drug_b
            else (record.drug_b, record.drug_a)
        )
        if key[0] != record.drug_a:
            reversed_fixed += 1

        if key in seen:
            duplicates += 1
            continue

        seen[key] = record.reports
        kept.append(record)
        graph.add_edge(
            key[0],
            key[1],
            weight=float(record.reports),
            lift=record.lift,
            support=record.support,
            passes_thresholds=record.passes_thresholds,
        )

    return _GraphBuild(
        graph=graph,
        edges=kept,
        edges_below_threshold=below,
        self_loops_skipped=self_loops,
        duplicate_edges_merged=duplicates,
        reversed_edges_fixed=reversed_fixed,
    )


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------
def _brandes_betweenness(adjacency: Mapping[str, Sequence[str]]) -> Dict[str, float]:
    """Independent, plain-Python Brandes betweenness (unweighted, undirected).

    Implemented from scratch so the verification does **not** reuse NetworkX.
    Returns the NetworkX-compatible normalisation: the accumulated pair
    dependency (which counts every unordered pair twice) divided by
    ``(n-1)(n-2)``.
    """
    nodes = list(adjacency)
    count = len(nodes)
    raw: Dict[str, float] = {node: 0.0 for node in nodes}

    for source in nodes:
        stack: List[str] = []
        predecessors: Dict[str, List[str]] = {node: [] for node in nodes}
        sigma: Dict[str, float] = {node: 0.0 for node in nodes}
        distance: Dict[str, float] = {node: -1.0 for node in nodes}
        sigma[source] = 1.0
        distance[source] = 0.0
        queue: deque = deque([source])

        while queue:
            vertex = queue.popleft()
            stack.append(vertex)
            for neighbour in adjacency[vertex]:
                if distance[neighbour] < 0:
                    distance[neighbour] = distance[vertex] + 1.0
                    queue.append(neighbour)
                if distance[neighbour] == distance[vertex] + 1.0:
                    sigma[neighbour] += sigma[vertex]
                    predecessors[neighbour].append(vertex)

        dependency: Dict[str, float] = {node: 0.0 for node in nodes}
        while stack:
            vertex = stack.pop()
            for predecessor in predecessors[vertex]:
                if sigma[vertex] > 0.0:
                    dependency[predecessor] += (sigma[predecessor] / sigma[vertex]) * (
                        1.0 + dependency[vertex]
                    )
            if vertex != source:
                raw[vertex] += dependency[vertex]

    if count <= 2:
        factor = 0.0
    else:
        factor = 1.0 / ((count - 1) * (count - 2))
    return {node: raw[node] * factor for node in nodes}


def compute_node_metrics(graph: nx.Graph) -> List[NodeMetric]:
    """Degree, weighted degree, degree centrality and betweenness per node."""
    degree = dict(graph.degree())
    weighted = dict(graph.degree(weight="weight"))
    centrality = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight=None)

    return [
        NodeMetric(
            drug=node,
            degree=int(degree[node]),
            weighted_degree=float(weighted[node]),
            degree_centrality=float(centrality[node]),
            betweenness_centrality=float(betweenness[node]),
        )
        for node in sorted(graph.nodes())
    ]


# ---------------------------------------------------------------------------
# Communities
# ---------------------------------------------------------------------------
def detect_communities(
    graph: nx.Graph,
    *,
    seed: int = 42,
) -> Tuple[List[List[str]], float, str]:
    """Louvain communities, their modularity and a short provenance note."""
    if graph.number_of_nodes() > COMMUNITY_NODE_LIMIT:
        return (
            [],
            0.0,
            f"skipped: {graph.number_of_nodes():,} nodes exceed the "
            f"{COMMUNITY_NODE_LIMIT:,}-node feasibility limit",
        )
    if graph.number_of_edges() == 0:
        return ([], 0.0, "skipped: the network has no edges")

    try:
        raw = nx.community.louvain_communities(graph, weight="weight", seed=seed)
        note = f"Louvain modularity communities (seeded, random_state={seed})"
    except AttributeError:  # pragma: no cover - older NetworkX
        raw = nx.community.greedy_modularity_communities(graph, weight="weight")
        note = "greedy modularity communities (Louvain unavailable)"

    # Deterministic ordering: biggest first, then by the alphabetically first drug.
    groups = [sorted(group) for group in raw]
    groups.sort(key=lambda group: (-len(group), group[0] if group else ""))
    modularity = float(nx.community.modularity(graph, [set(g) for g in groups], weight="weight"))
    return (groups, modularity, note)


def _describe_communities(
    graph: nx.Graph,
    groups: Sequence[Sequence[str]],
    weighted: Mapping[str, float],
) -> Tuple[List[Community], Dict[str, int]]:
    """Summarise each community; also return the drug -> community-id membership map."""
    described: List[Community] = []
    membership = {drug: index for index, group in enumerate(groups) for drug in group}
    for index, group in enumerate(groups):
        members = set(group)
        internal = [edge for edge in graph.edges(members) if edge[0] in members and edge[1] in members]
        total = sum(float(graph[u][v]["weight"]) for u, v in internal)
        top = max(group, key=lambda drug: (weighted.get(drug, 0.0), drug)) if group else ""
        described.append(
            Community(
                community_id=index,
                size=len(members),
                total_weight=total,
                internal_edges=len(internal),
                top_drug=top,
                drugs=list(group),
            )
        )
    return described, membership


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _select_display(
    graph: nx.Graph,
    weighted: Mapping[str, float],
    *,
    max_nodes: int,
    max_edges: int,
) -> Tuple[nx.Graph, int, int]:
    """Induced sub-graph of the most-connected drugs, capped in nodes and edges."""
    if graph.number_of_nodes() <= max_nodes:
        chosen = list(graph.nodes())
    else:
        chosen = sorted(graph.nodes(), key=lambda node: (-weighted.get(node, 0.0), node))[
            :max_nodes
        ]
    sub = graph.subgraph(chosen).copy()
    reported_edges = sub.number_of_edges()

    if sub.number_of_edges() > max_edges:
        ranked = sorted(
            sub.edges(data=True),
            key=lambda item: (-float(item[2]["weight"]), item[0], item[1]),
        )
        keep = [(u, v) for u, v, _data in ranked[:max_edges]]
        trimmed = nx.Graph()
        trimmed.add_nodes_from(sub.nodes())
        for u, v in keep:
            trimmed.add_edge(u, v, **sub[u][v])
        sub = trimmed

    return sub, sub.number_of_nodes(), reported_edges


def _layout(graph: nx.Graph, *, name: str, seed: int) -> Dict[str, Tuple[float, float]]:
    """Deterministic 2-D layout; weight is treated as tie strength."""
    if graph.number_of_nodes() == 0:
        return {}
    if name == "circular":
        positions = nx.circular_layout(graph)
    elif name == "kamada_kawai":
        positions = nx.kamada_kawai_layout(graph, weight="weight")
    elif name == "shell":
        positions = nx.shell_layout(graph)
    else:
        positions = nx.spring_layout(graph, weight="weight", seed=seed, iterations=200)
    return {str(node): (float(x), float(y)) for node, (x, y) in positions.items()}


def analyse(
    *,
    input_path: Path,
    min_edge_weight: float = 2.0,
    max_display_nodes: int = 60,
    max_display_edges: int = 240,
    layout: str = "spring",
    seed: int = 42,
) -> NetworkAnalysis:
    """Build the network, measure it, and validate every derived quantity."""
    log = get_logger(__name__)
    analysis = NetworkAnalysis(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        input_path=str(input_path),
        min_edge_weight=min_edge_weight,
        display_node_limit=max_display_nodes,
        display_edge_limit=max_display_edges,
        layout_name=layout,
        layout_seed=seed,
    )

    records = load_edge_records(input_path)
    analysis.input_rows = len(records)
    analysis.edges = records
    log.info("read %s co-reported pair(s) from %s", f"{len(records):,}", input_path.name)

    built = build_graph(records, min_edge_weight=min_edge_weight)
    graph = built.graph
    analysis.edges = built.edges
    analysis.edges_below_threshold = built.edges_below_threshold
    analysis.self_loops_skipped = built.self_loops_skipped
    analysis.duplicate_edges_merged = built.duplicate_edges_merged
    analysis.reversed_edges_fixed = built.reversed_edges_fixed

    analysis.node_count = graph.number_of_nodes()
    analysis.edge_count = graph.number_of_edges()
    analysis.total_weight = float(sum(data["weight"] for _u, _v, data in graph.edges(data=True)))
    analysis.density = float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0
    components = list(nx.connected_components(graph))
    analysis.connected_components = len(components)
    analysis.largest_component = max((len(group) for group in components), default=0)

    analysis.nodes = compute_node_metrics(graph)
    log.info(
        "network: %s node(s), %s edge(s), %s component(s)",
        f"{analysis.node_count:,}",
        f"{analysis.edge_count:,}",
        analysis.connected_components,
    )

    groups, modularity, note = detect_communities(graph, seed=seed)
    analysis.modularity = modularity
    analysis.community_note = note
    weighted = {node.drug: node.weighted_degree for node in analysis.nodes}
    analysis.communities, membership = _describe_communities(graph, groups, weighted)
    analysis.community_count = len(analysis.communities)
    for node in analysis.nodes:
        node.community = membership.get(node.drug, -1)

    display_graph, display_nodes, reported_edges = _select_display(
        graph, weighted, max_nodes=max_display_nodes, max_edges=max_display_edges
    )
    analysis.graph = graph
    analysis.display_graph = display_graph
    analysis.display_nodes = display_nodes
    analysis.display_edges = display_graph.number_of_edges()
    analysis.display_layout = _layout(display_graph, name=layout, seed=seed)
    if reported_edges != analysis.display_edges:
        log.info(
            "displayed network trimmed: %s edge(s) -> %s edge(s); %s node(s)",
            f"{reported_edges:,}",
            f"{analysis.display_edges:,}",
            f"{display_nodes:,}",
        )

    analysis.checks = _validate(analysis, records=records, seed=seed, layout=layout)
    return analysis


def load_analysis(
    *,
    input_path: Optional[Path] = None,
    min_edge_weight: float = 2.0,
    max_display_nodes: int = 60,
    max_display_edges: int = 240,
    layout: str = "spring",
    seed: int = 42,
) -> NetworkAnalysis:
    """Dashboard-facing convenience wrapper: analyse using the project defaults."""
    return analyse(
        input_path=input_path or (get_path("analytics") / "drug_cooccurrence" / "cooccurrence_top_pairs.csv"),
        min_edge_weight=min_edge_weight,
        max_display_nodes=max_display_nodes,
        max_display_edges=max_display_edges,
        layout=layout,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(
    analysis: NetworkAnalysis,
    *,
    records: Sequence[EdgeRecord],
    seed: int,
    layout: str,
) -> List[Tuple[str, bool, str]]:
    """Re-derive every headline quantity independently and record the outcome."""
    checks: List[Tuple[str, bool, str]] = []
    graph = analysis.graph
    assert graph is not None  # nosec - set by analyse()

    # 1 - input read -------------------------------------------------------
    checks.append(
        (
            "input pairs read from the co-occurrence results",
            analysis.input_rows == len(records) and analysis.input_rows > 0,
            f"{analysis.input_rows:,} pair row(s) read from {Path(analysis.input_path).name}",
        )
    )

    # 2 - node and edge counts --------------------------------------------
    expected_nodes = {record.drug_a for record in analysis.edges} | {
        record.drug_b for record in analysis.edges
    }
    checks.append(
        (
            "node and edge counts match the retained pair list",
            analysis.node_count == len(expected_nodes) == graph.number_of_nodes()
            and analysis.edge_count == len(analysis.edges) == graph.number_of_edges(),
            f"{analysis.node_count:,} node(s) = {len(expected_nodes):,} distinct drug(s); "
            f"{analysis.edge_count:,} edge(s) = {len(analysis.edges):,} retained pair(s)",
        )
    )

    # 3 - edge weights equal the actual co-occurrence counts ---------------
    mismatches: List[str] = []
    for record in analysis.edges:
        if not graph.has_edge(record.drug_a, record.drug_b):
            mismatches.append(f"{record.drug_a}+{record.drug_b} missing")
            continue
        stored = float(graph[record.drug_a][record.drug_b]["weight"])
        if stored != float(record.reports):
            mismatches.append(f"{record.drug_a}+{record.drug_b} {stored} != {record.reports}")
    total_expected = float(sum(record.reports for record in analysis.edges))
    checks.append(
        (
            "edge weights equal the actual co-occurrence report counts",
            not mismatches and analysis.total_weight == total_expected,
            f"checked {len(analysis.edges):,} edge(s); total weight "
            f"{analysis.total_weight:,.0f} = sum of the 'reports' column "
            f"{total_expected:,.0f}; {len(mismatches)} mismatch(es)",
        )
    )

    # 4 - self-loops --------------------------------------------------------
    self_loops = [node for node in graph.nodes() if graph.has_edge(node, node)]
    checks.append(
        (
            "no self-loop edges",
            not self_loops,
            f"{len(self_loops)} self-loop(s) in the graph; "
            f"{analysis.self_loops_skipped} skipped while reading",
        )
    )

    # 5 - duplicate edges ---------------------------------------------------
    keys = [record.key for record in analysis.edges]
    duplicates = len(keys) - len(set(keys))
    checks.append(
        (
            "no duplicate edges (each unordered pair once)",
            duplicates == 0 and analysis.duplicate_edges_merged == 0,
            f"{len(set(keys)):,} distinct unordered pair(s) over {len(keys):,} edge(s); "
            f"{duplicates} duplicate key(s), {analysis.duplicate_edges_merged} merged",
        )
    )

    # 6 - degree centrality -------------------------------------------------
    adjacency = {node: sorted(graph.neighbors(node)) for node in graph.nodes()}
    count = len(adjacency)
    degree_ok = True
    degree_detail = ""
    for node in analysis.nodes:
        expected_degree = len(adjacency[node.drug])
        expected_centrality = expected_degree / (count - 1) if count > 1 else 0.0
        if node.degree != expected_degree or abs(node.degree_centrality - expected_centrality) > 1e-12:
            degree_ok = False
            degree_detail = (
                f"{node.drug}: degree {node.degree} vs {expected_degree}, "
                f"centrality {node.degree_centrality} vs {expected_centrality}"
            )
            break
    checks.append(
        (
            "degree and degree centrality match an independent recount",
            degree_ok,
            degree_detail or f"recounted all {count:,} node(s) from the adjacency lists",
        )
    )

    # 7 - weighted degree ---------------------------------------------------
    weight_ok = True
    weight_detail = ""
    for node in analysis.nodes:
        expected = float(sum(graph[node.drug][other]["weight"] for other in adjacency[node.drug]))
        if abs(node.weighted_degree - expected) > 1e-9:
            weight_ok = False
            weight_detail = f"{node.drug}: {node.weighted_degree} vs {expected}"
            break
    handshake = math.isclose(
        sum(node.weighted_degree for node in analysis.nodes),
        2.0 * analysis.total_weight,
        rel_tol=1e-9,
    )
    checks.append(
        (
            "weighted degree equals the incident edge weights (handshake holds)",
            weight_ok and handshake,
            weight_detail
            or f"sum of weighted degrees = {2.0 * analysis.total_weight:,.0f} "
            f"= 2 x total weight; every node recounted",
        )
    )

    # 8 - betweenness -------------------------------------------------------
    if count <= BETWEENNESS_RECOMPUTE_LIMIT:
        independent = _brandes_betweenness(adjacency)
        worst = 0.0
        worst_node = ""
        for node in analysis.nodes:
            diff = abs(node.betweenness_centrality - independent[node.drug])
            if diff > worst:
                worst, worst_node = diff, node.drug
        checks.append(
            (
                "betweenness centrality matches an independent Brandes computation",
                worst <= 1e-9,
                f"recomputed all {count:,} node(s) in plain Python; "
                f"max |difference| = {worst:.3e}"
                + (f" at {worst_node}" if worst_node else ""),
            )
        )
    else:
        bounded = all(0.0 <= node.betweenness_centrality <= 1.0 for node in analysis.nodes)
        checks.append(
            (
                "betweenness centrality is within [0, 1]",
                bounded,
                f"{count:,} node(s) exceed the {BETWEENNESS_RECOMPUTE_LIMIT:,}-node "
                "recomputation limit; only the bounds were checked",
            )
        )

    # 9 - communities form a partition --------------------------------------
    if analysis.communities:
        members = [drug for community in analysis.communities for drug in community.drugs]
        partition_ok = (
            len(members) == count
            and len(set(members)) == count
            and set(members) == set(graph.nodes())
            and all(community.size > 0 for community in analysis.communities)
        )
        checks.append(
            (
                "communities partition the nodes exactly once",
                partition_ok,
                f"{analysis.community_count:,} community(ies) covering {len(set(members)):,} "
                f"of {count:,} node(s); modularity = {analysis.modularity:.4f}",
            )
        )
    else:
        checks.append(
            (
                "communities partition the nodes exactly once",
                True,
                f"not applicable - {analysis.community_note}",
            )
        )

    # 10 - displayed network is within the configured limits -----------------
    within = (
        analysis.display_nodes <= analysis.display_node_limit
        and analysis.display_edges <= analysis.display_edge_limit
    )
    checks.append(
        (
            "displayed network is within the configured node and edge limits",
            within,
            f"{analysis.display_nodes:,} node(s) <= {analysis.display_node_limit:,}; "
            f"{analysis.display_edges:,} edge(s) <= {analysis.display_edge_limit:,}",
        )
    )

    # 11 - reproducibility ---------------------------------------------------
    # The third value is the pre-trim edge count (a log aid), so the re-derived
    # graph's own edge count is used for the comparison below.
    repeat_graph, repeat_nodes, _pre_trim_edges = _select_display(
        graph,
        {node.drug: node.weighted_degree for node in analysis.nodes},
        max_nodes=analysis.display_node_limit,
        max_edges=analysis.display_edge_limit,
    )
    repeat_layout = _layout(repeat_graph, name=layout, seed=seed)
    repeat_groups, repeat_modularity, _note = detect_communities(graph, seed=seed)
    layout_drift = 0.0
    for drug in set(analysis.display_layout) | set(repeat_layout):
        first = analysis.display_layout.get(drug)
        second = repeat_layout.get(drug)
        if first is None or second is None:
            layout_drift = math.inf
            break
        layout_drift = max(layout_drift, abs(first[0] - second[0]), abs(first[1] - second[1]))
    same_partition = [sorted(group) for group in repeat_groups] == [
        community.drugs for community in analysis.communities
    ]
    reproducible = (
        layout_drift <= 1e-12
        and same_partition
        and repeat_nodes == analysis.display_nodes
        and repeat_graph.number_of_edges() == analysis.display_edges
        and math.isclose(repeat_modularity, analysis.modularity, rel_tol=1e-12, abs_tol=1e-12)
    )
    checks.append(
        (
            "layout, selection and communities are reproducible with a fixed seed",
            reproducible,
            f"seed={seed}; max layout drift = {layout_drift:.3e}; "
            f"identical communities = {same_partition}; modularity {repeat_modularity:.4f}",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Reusable Plotly figure builders (no Streamlit dependency)
# ---------------------------------------------------------------------------
def _node_sizes(weighted_degrees: Sequence[float]) -> List[float]:
    """Log-scaled marker sizes so a few hubs do not swamp the rest."""
    if not weighted_degrees:
        return []
    lo = math.log10(min(weighted_degrees) + 1.0)
    hi = math.log10(max(weighted_degrees) + 1.0)
    if hi <= lo:
        return [22.0] * len(weighted_degrees)
    return [
        10.0 + 34.0 * (math.log10(value + 1.0) - lo) / (hi - lo)
        for value in weighted_degrees
    ]


def _edge_weight_classes(weights: Sequence[float]) -> Dict[float, int]:
    """Assign each edge weight to one of :data:`EDGE_WIDTH_BINS` quantile classes."""
    if not weights:
        return {}
    ordered = sorted(set(weights))
    if len(ordered) == 1:
        return {ordered[0]: EDGE_WIDTH_BINS - 1}
    cuts = [
        ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]
        for fraction in (0.25, 0.5, 0.75)
    ]
    classes: Dict[float, int] = {}
    for value in ordered:
        if value <= cuts[0]:
            classes[value] = 0
        elif value <= cuts[1]:
            classes[value] = 1
        elif value <= cuts[2]:
            classes[value] = 2
        else:
            classes[value] = 3
    return classes


def network_figure(
    analysis: NetworkAnalysis,
    *,
    theme: str = "plotly_white",
    title: Optional[str] = None,
) -> go.Figure:
    """Interactive drug network: nodes coloured by community, edges by weight.

    The figure is limited to ``analysis.display_graph`` (the most-connected
    drugs) so it stays readable; the metrics themselves come from the full
    network.
    """
    graph = analysis.display_graph
    layout = analysis.display_layout
    members = {drug: node.community for drug, node in
               ((n.drug, n) for n in analysis.nodes)}
    metrics = {node.drug: node for node in analysis.nodes}

    figure = go.Figure()
    if graph is None or graph.number_of_nodes() == 0:
        figure.update_layout(template=theme, title=title or "Drug co-reporting network (empty)")
        return figure

    # --- edges, grouped into width classes --------------------------------
    display_edges = [(u, v, float(data["weight"])) for u, v, data in graph.edges(data=True)]
    classes = _edge_weight_classes([weight for _u, _v, weight in display_edges])
    grouped: Dict[int, List[Tuple[str, str, float]]] = {}
    for u, v, weight in display_edges:
        grouped.setdefault(classes.get(weight, 0), []).append((u, v, weight))

    for index in sorted(grouped):
        xs: List[Optional[float]] = []
        ys: List[Optional[float]] = []
        for u, v, _weight in grouped[index]:
            if u not in layout or v not in layout:
                continue
            xs.extend([layout[u][0], layout[v][0], None])
            ys.extend([layout[u][1], layout[v][1], None])
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=0.6 + 0.9 * index, color="#9aa4b2"),
                hoverinfo="skip",
                showlegend=False,
                name=f"edge weight class {index + 1}",
            )
        )

    # --- nodes, one trace per community (legend + click to toggle) --------
    by_community: Dict[int, List[str]] = {}
    for drug in graph.nodes():
        by_community.setdefault(members.get(drug, -1), []).append(drug)
    sizes = {node.drug: size for node, size in
             zip(analysis.nodes, _node_sizes([n.weighted_degree for n in analysis.nodes]))}

    for order, (community_id, drugs) in enumerate(
        sorted(by_community.items(), key=lambda item: (-len(item[1]), item[0]))
    ):
        drugs = sorted(drugs)
        subset = [metrics[drug] for drug in drugs if drug in metrics]
        figure.add_trace(
            go.Scatter(
                x=[layout[drug][0] for drug in drugs],
                y=[layout[drug][1] for drug in drugs],
                mode="markers+text",
                text=[_clip(drug, 18) for drug in drugs],
                textposition="top center",
                textfont=dict(size=8),
                marker=dict(
                    size=[sizes.get(drug, 14.0) for drug in drugs],
                    color=COMMUNITY_COLORS[order % len(COMMUNITY_COLORS)],
                    line=dict(width=1.0, color="#2b2b2b"),
                    opacity=0.9,
                ),
                name=f"Community {community_id} ({len(drugs)})",
                customdata=[
                    [
                        item.degree,
                        item.weighted_degree,
                        item.degree_centrality,
                        item.betweenness_centrality,
                        item.community,
                        graph.degree(drug),
                    ]
                    for item, drug in zip(subset, drugs)
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "degree: %{customdata[0]}<br>"
                    "weighted degree: %{customdata[1]:,.0f}<br>"
                    "degree centrality: %{customdata[2]:.4f}<br>"
                    "betweenness: %{customdata[3]:.4f}<br>"
                    "community: %{customdata[4]}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        template=theme,
        title=title
        or (
            f"Drug co-reporting network - top {analysis.display_nodes} drugs "
            f"({analysis.display_edges} edges, from {analysis.edge_count:,} screened pairs)"
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(title="Community", orientation="v", font=dict(size=10)),
        hovermode="closest",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return figure


def top_connected_figure(
    analysis: NetworkAnalysis,
    *,
    top_n: int = TOP_CONNECTED_REPORT,
    theme: str = "plotly_white",
) -> go.Figure:
    """Horizontal bars of the ``top_n`` most-connected drugs (two measures)."""
    ranked = analysis.ranked_by_weighted_degree()[:top_n]
    labels = [node.drug for node in ranked][::-1]
    weighted = [node.weighted_degree for node in ranked][::-1]
    degrees = [node.degree for node in ranked][::-1]

    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.12,
        subplot_titles=("Total co-reports (weighted degree)", "Distinct partners (degree)"),
    )
    figure.add_trace(
        go.Bar(
            x=weighted,
            y=labels,
            orientation="h",
            marker=dict(color="#1f77b4"),
            text=[f"{value:,.0f}" for value in weighted],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>total co-reports: %{x:,.0f}<extra></extra>",
            name="Weighted degree",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=degrees,
            y=labels,
            orientation="h",
            marker=dict(color="#ff7f0e"),
            text=[f"{value:,}" for value in degrees],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>distinct partners: %{x:,}<extra></extra>",
            name="Degree",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        template=theme,
        title=f"Top {len(ranked)} most-connected drugs in the co-reporting network",
        showlegend=False,
        margin=dict(l=10, r=40, t=70, b=10),
        height=max(420, 22 * max(len(labels), 1) + 140),
    )
    figure.update_xaxes(title_text="total co-reports", row=1, col=1)
    figure.update_xaxes(title_text="distinct co-reported drugs", row=1, col=2)
    return figure


def relationships_figure(
    analysis: NetworkAnalysis,
    *,
    top_n: int = TOP_CONNECTED_REPORT,
    theme: str = "plotly_white",
) -> go.Figure:
    """Drug relationship analysis: strongest pairs and centrality structure."""
    figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.16,
        subplot_titles=(
            "Strongest co-reported drug pairs",
            "Centrality structure (degree vs betweenness)",
        ),
    )

    strongest = analysis.edges[:top_n]
    pair_labels = [f"{_clip(edge.drug_a, 16)} + {_clip(edge.drug_b, 16)}" for edge in strongest][::-1]
    pair_weights = [edge.reports for edge in strongest][::-1]
    pair_lift = [edge.lift for edge in strongest][::-1]
    figure.add_trace(
        go.Bar(
            x=pair_weights,
            y=pair_labels,
            orientation="h",
            marker=dict(color="#2ca02c"),
            customdata=[[lift] for lift in pair_lift],
            hovertemplate=(
                "<b>%{y}</b><br>reports together: %{x:,}<br>lift: %{customdata[0]:.2f}<extra></extra>"
            ),
            name="Co-reported pairs",
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=[node.degree_centrality for node in analysis.nodes],
            y=[node.betweenness_centrality for node in analysis.nodes],
            mode="markers",
            marker=dict(
                size=_node_sizes([node.weighted_degree for node in analysis.nodes]),
                color=[node.community for node in analysis.nodes],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="community", x=1.02, len=0.8),
                line=dict(width=0.6, color="#2b2b2b"),
                opacity=0.85,
            ),
            customdata=[
                [node.drug, node.degree, node.weighted_degree]
                for node in analysis.nodes
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "degree centrality: %{x:.4f}<br>"
                "betweenness: %{y:.4f}<br>"
                "degree: %{customdata[1]}<br>"
                "weighted degree: %{customdata[2]:,.0f}<extra></extra>"
            ),
            name="Drugs",
        ),
        row=1,
        col=2,
    )

    figure.update_layout(
        template=theme,
        title=(
            f"Drug relationship analysis - {analysis.node_count:,} drugs, "
            f"{analysis.edge_count:,} co-reported pairs"
        ),
        showlegend=False,
        height=max(520, 22 * max(len(pair_labels), 1) + 160),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    figure.update_xaxes(title_text="reports containing both drugs", row=1, col=1)
    figure.update_xaxes(title_text="degree centrality", row=1, col=2)
    figure.update_yaxes(title_text="betweenness centrality", row=1, col=2)
    return figure


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_html(figure: go.Figure, path: Path) -> None:
    """Write a self-contained interactive HTML page (Plotly only, no pandas)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs=True, full_html=True)


def write_outputs(
    analysis: NetworkAnalysis,
    *,
    output_dir: Path,
    theme: str = "plotly_white",
    top_n: int = TOP_CONNECTED_REPORT,
) -> Dict[str, Path]:
    """Write the three figures and the node/edge/community tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    _write_html(network_figure(analysis, theme=theme), output_dir / "drug_network.html")
    written["drug_network.html"] = output_dir / "drug_network.html"
    _write_html(
        top_connected_figure(analysis, top_n=top_n, theme=theme),
        output_dir / "top_connected_drugs.html",
    )
    written["top_connected_drugs.html"] = output_dir / "top_connected_drugs.html"
    _write_html(
        relationships_figure(analysis, top_n=top_n, theme=theme),
        output_dir / "drug_relationships.html",
    )
    written["drug_relationships.html"] = output_dir / "drug_relationships.html"

    _write_csv(
        output_dir / "drug_network_nodes.csv",
        [
            "drug",
            "degree",
            "weighted_degree",
            "degree_centrality",
            "betweenness_centrality",
            "community",
            "in_displayed_network",
        ],
        (
            [
                node.drug,
                node.degree,
                round(node.weighted_degree, 6),
                round(node.degree_centrality, 10),
                round(node.betweenness_centrality, 10),
                node.community,
                node.drug in analysis.display_layout,
            ]
            for node in analysis.ranked_by_weighted_degree()
        ),
    )
    written["drug_network_nodes.csv"] = output_dir / "drug_network_nodes.csv"

    _write_csv(
        output_dir / "drug_network_edges.csv",
        [
            "drug_a",
            "drug_b",
            "reports",
            "weight",
            "support",
            "confidence_a_to_b",
            "confidence_b_to_a",
            "lift",
            "passes_thresholds",
            "in_displayed_network",
        ],
        (
            [
                edge.drug_a,
                edge.drug_b,
                edge.reports,
                edge.reports,
                round(edge.support, 10),
                round(edge.confidence_a_to_b, 10),
                round(edge.confidence_b_to_a, 10),
                round(edge.lift, 6),
                edge.passes_thresholds,
                analysis.display_graph is not None
                and analysis.display_graph.has_edge(edge.drug_a, edge.drug_b),
            ]
            for edge in analysis.edges
        ),
    )
    written["drug_network_edges.csv"] = output_dir / "drug_network_edges.csv"

    _write_csv(
        output_dir / "drug_network_communities.csv",
        ["community_id", "size", "internal_edges", "total_weight", "top_drug", "drugs"],
        (
            [
                community.community_id,
                community.size,
                community.internal_edges,
                round(community.total_weight, 6),
                community.top_drug,
                " | ".join(community.drugs),
            ]
            for community in analysis.communities
        ),
    )
    written["drug_network_communities.csv"] = output_dir / "drug_network_communities.csv"

    analysis.output_paths = {name: str(path) for name, path in written.items()}
    summary_path = output_dir / "drug_network_summary.json"
    summary_path.write_text(
        json.dumps(analysis.as_summary(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written["drug_network_summary.json"] = summary_path
    analysis.output_paths["drug_network_summary.json"] = str(summary_path)
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_report(analysis: NetworkAnalysis) -> str:
    """Render the network report as plain text."""
    lines: List[str] = [
        "=" * 96,
        "FAERS DRUG CO-REPORTING NETWORK REPORT",
        f"Generated : {analysis.generated_at}",
        "Engine    : NetworkX (graph metrics) + Plotly (interactive figures)",
        "Input     : the project's own co-occurrence results (real FAERS 2026 Q1/Q2 data)",
        "=" * 96,
        "",
        "0. SCOPE AND METHOD",
        "-" * 96,
        f"  {NETWORK_GRAIN}",
        "  Centrality measures per drug:",
        "    degree                = number of distinct co-reported drugs",
        "    weighted degree       = sum of the incident edge weights (total co-reports)",
        "    degree centrality     = degree / (n - 1)",
        "    betweenness centrality= share of shortest paths passing through the drug,",
        "                            normalised (undirected, unweighted hop distance)",
        "  Betweenness uses hop distance, not the co-report weight: the weight is a",
        "  similarity (more co-reports = a stronger tie), not a path length.",
        f"  Communities: {analysis.community_note or 'not run'};",
        f"  partition quality (modularity) = {analysis.modularity:.4f}.",
        "",
        "  !! " + NETWORK_DISCLAIMER,
        "",
        "  LIMITATION: the input pair table holds the strongest pairs only (it is",
        "  capped upstream), so this is a screened sub-network of the full",
        "  co-reporting graph, not the complete drug network.",
        "",
        "1. INPUT DATA",
        "-" * 96,
        f"  Pair table                     : {analysis.input_path}",
        f"  Pair rows read                 : {analysis.input_rows:>14,}",
        f"  Minimum edge weight (config)   : {analysis.min_edge_weight:>14,.0f}",
        f"  Edges dropped below threshold  : {analysis.edges_below_threshold:>14,}",
        f"  Self-loops skipped             : {analysis.self_loops_skipped:>14,}",
        f"  Reversed edges re-keyed        : {analysis.reversed_edges_fixed:>14,}",
        f"  Duplicate edges merged         : {analysis.duplicate_edges_merged:>14,}",
        "",
        "2. NETWORK SUMMARY (full network built from the pair table)",
        "-" * 96,
        f"  Nodes (distinct drugs)         : {analysis.node_count:>14,}",
        f"  Edges (co-reported pairs)      : {analysis.edge_count:>14,}",
        f"  Total edge weight (co-reports) : {analysis.total_weight:>14,.0f}",
        f"  Connected components           : {analysis.connected_components:>14,}",
        f"  Largest component              : {analysis.largest_component:>14,}",
        f"  Density                        : {analysis.density:>14.6f}",
        "",
        f"3. TOP {TOP_CONNECTED_REPORT} MOST-CONNECTED DRUGS (by weighted degree)",
        "-" * 96,
        f"  {'#':>3}  {'DRUG':<28}{'DEGREE':>8}{'WEIGHTED DEG':>14}{'DEG CENTRAL':>13}{'BETWEENNESS':>13}",
    ]
    for index, node in enumerate(analysis.ranked_by_weighted_degree()[:TOP_CONNECTED_REPORT], 1):
        lines.append(
            f"  {index:>3}  {_clip(node.drug, 26):<28}{node.degree:>8,}"
            f"{node.weighted_degree:>14,.0f}{node.degree_centrality:>13.4f}"
            f"{node.betweenness_centrality:>13.4f}"
        )

    lines.append("")
    lines.append(f"4. CENTRALITY LEADERS (top {TOP_BETWEENNESS_REPORT} by betweenness centrality)")
    lines.append("-" * 96)
    lines.append(
        f"  {'#':>3}  {'DRUG':<28}{'BETWEENNESS':>13}{'DEGREE':>8}{'WEIGHTED DEG':>14}{'COMMUNITY':>11}"
    )
    for index, node in enumerate(analysis.ranked_by_betweenness()[:TOP_BETWEENNESS_REPORT], 1):
        lines.append(
            f"  {index:>3}  {_clip(node.drug, 26):<28}{node.betweenness_centrality:>13.4f}"
            f"{node.degree:>8,}{node.weighted_degree:>14,.0f}{node.community:>11}"
        )
    lines.append(
        "  Betweenness highlights drugs that bridge otherwise separate groups; a high"
    )
    lines.append(
        "  value is a property of this reporting network, not evidence of any interaction."
    )

    lines.append("")
    lines.append("5. COMMUNITIES")
    lines.append("-" * 96)
    if analysis.communities:
        lines.append(f"  {analysis.community_count:,} community(ies); modularity = {analysis.modularity:.4f}")
        lines.append(f"  {'ID':>4}  {'SIZE':>6}{'INT EDGES':>11}{'WEIGHT':>14}  TOP DRUG")
        for community in analysis.communities[:TOP_COMMUNITIES_REPORT]:
            lines.append(
                f"  {community.community_id:>4}  {community.size:>6,}{community.internal_edges:>11,}"
                f"{community.total_weight:>14,.0f}  {_clip(community.top_drug, 34)}"
            )
        lines.append("")
        lines.append("  Largest community members (capped at 20 shown):")
        for community in analysis.communities[:3]:
            members = " | ".join(community.drugs[:20])
            lines.append(f"    [{community.community_id}] {_clip(members, 88)}")
    else:
        lines.append(f"  no communities detected ({analysis.community_note}).")

    lines.append("")
    lines.append("6. DISPLAYED NETWORK (the interactive figure)")
    lines.append("-" * 96)
    lines.append(f"  Layout                                 : {analysis.layout_name} (seed {analysis.layout_seed})")
    lines.append(f"  Node limit (config)                    : {analysis.display_node_limit:>14,}")
    lines.append(f"  Edge limit (config)                    : {analysis.display_edge_limit:>14,}")
    lines.append(f"  Nodes shown                            : {analysis.display_nodes:>14,}")
    lines.append(f"  Edges shown                            : {analysis.display_edges:>14,}")
    lines.append(
        "  The figure keeps the most-connected drugs so it stays readable; every"
    )
    lines.append(
        "  metric in sections 2-5 is computed on the FULL network, not on this subset."
    )

    lines.append("")
    lines.append("7. VALIDATION CHECKS")
    lines.append("-" * 96)
    for name, ok, detail in analysis.checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        lines.append(f"         {detail}")

    lines.append("")
    lines.append("8. ISSUES AND ERRORS")
    lines.append("-" * 96)
    if analysis.errors:
        for error in analysis.errors:
            lines.append(f"  {error}")
    else:
        lines.append("  none")

    lines.append("")
    lines.append("9. OUTPUTS")
    lines.append("-" * 96)
    for name, path in sorted(analysis.output_paths.items()):
        lines.append(f"  {name:<30} {path}")

    failed = [name for name, ok, _detail in analysis.checks if not ok]
    lines.append("")
    lines.append("10. SUMMARY")
    lines.append("-" * 96)
    lines.append(
        f"  {analysis.node_count:,} drug node(s), {analysis.edge_count:,} co-reported "
        f"edge(s) (total weight {analysis.total_weight:,.0f}), "
        f"{analysis.community_count:,} community(ies) at modularity {analysis.modularity:.4f}; "
        f"figure shows {analysis.display_nodes:,} node(s) / {analysis.display_edges:,} edge(s)."
    )
    if analysis.errors or failed:
        lines.append(
            f"  Status: INCOMPLETE - {len(analysis.errors)} error(s), "
            f"{len(failed)} failed check(s)."
        )
    else:
        lines.append("  Status: COMPLETE - network built, measured and validated; all checks passed.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def resolve_params(overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Merge the ``visualization`` config with non-null CLI overrides."""
    section = get_visualization_config()
    network = section.get("network") or {}
    params: Dict[str, Any] = {
        "theme": str(section.get("theme", "plotly_white")),
        "top_n": int(section.get("top_n", TOP_CONNECTED_REPORT) or TOP_CONNECTED_REPORT),
        "min_edge_weight": float(network.get("min_edge_weight", 0.0) or 0.0),
        "layout": str(network.get("layout", "spring") or "spring"),
        "max_display_nodes": int(network.get("max_display_nodes", 60) or 60),
        "max_display_edges": int(network.get("max_display_edges", 240) or 240),
        "seed": 42,
    }
    for key, value in (overrides or {}).items():
        if value is not None and key in params:
            params[key] = type(params[key])(value) if not isinstance(params[key], str) else str(value)
        elif value is not None:
            params[key] = value
    return params


def run(
    *,
    input_path: Path,
    output_dir: Optional[Path] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> NetworkAnalysis:
    """Analyse the network, persist the outputs (when a directory is given)."""
    log = get_logger(__name__)
    resolved = resolve_params(params)
    analysis = analyse(
        input_path=input_path,
        min_edge_weight=resolved["min_edge_weight"],
        max_display_nodes=resolved["max_display_nodes"],
        max_display_edges=resolved["max_display_edges"],
        layout=resolved["layout"],
        seed=resolved["seed"],
    )
    if output_dir is not None:
        written = write_outputs(
            analysis,
            output_dir=output_dir,
            theme=resolved["theme"],
            top_n=resolved["top_n"],
        )
        for name, path in written.items():
            log.info("wrote %s -> %s", name, path)
    return analysis


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Drug co-reporting network + interactive Plotly figures (read-only)."
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Pair table (default data/analytics/drug_cooccurrence/cooccurrence_top_pairs.csv)."
        ),
    )
    parser.add_argument(
        "--outputs",
        default=None,
        help="Figure/result directory (default data/analytics/network; '' disables it).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Report path (default reports/drug_network_report.txt; '' disables it).",
    )
    parser.add_argument("--max-nodes", type=int, default=None, help="Displayed node cap.")
    parser.add_argument("--max-edges", type=int, default=None, help="Displayed edge cap.")
    parser.add_argument("--min-edge-weight", type=float, default=None, help="Drop lighter edges.")
    parser.add_argument("--layout", default=None, help="spring | circular | kamada_kawai | shell")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the layout/communities.")
    parser.add_argument("--top-n", type=int, default=None, help="Size of the 'top connected' charts.")
    args = parser.parse_args(argv)

    log = get_logger(__name__)
    try:
        params = resolve_params(
            {
                "min_edge_weight": args.min_edge_weight,
                "max_display_nodes": args.max_nodes,
                "max_display_edges": args.max_edges,
                "layout": args.layout,
                "seed": args.seed,
                "top_n": args.top_n,
            }
        )
    except ConfigError as exc:
        log.error("configuration error: %s", exc)
        return 1

    input_path = (
        Path(args.input)
        if args.input
        else get_path("analytics") / "drug_cooccurrence" / "cooccurrence_top_pairs.csv"
    )
    if args.outputs is None:
        output_dir: Optional[Path] = get_path("analytics") / "network"
    else:
        output_dir = Path(args.outputs) if args.outputs else None

    try:
        analysis = run(input_path=input_path, output_dir=output_dir, params=params)
    except DrugNetworkError as exc:
        log.error("network construction failed: %s", exc)
        return 2
    except MemoryError as exc:
        log.error("network construction ran out of memory: %s", exc)
        return 3

    report = render_report(analysis)
    print(report)

    report_path = args.report
    if report_path is None:
        report_path = str(get_path("reports") / "drug_network_report.txt")
    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report + "\n", encoding="utf-8")
        log.info("Report written to: %s", target)

    if analysis.errors or any(not ok for _n, ok, _d in analysis.checks):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
