"""Section 5 - Drug Network.

Interactive drug relationship network plus centrality and community summaries,
built from the network CSVs.  Positions are recomputed with a seeded spring
layout (cached); the pre-rendered interactive HTML is offered as a fallback and
a download.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st

from dashboard.components import charts as ch
from dashboard.components import data as ds
from dashboard.components import ui

_NODE_COLUMNS = [
    "drug",
    "degree",
    "weighted_degree",
    "degree_centrality",
    "betweenness_centrality",
    "community",
    "in_displayed_network",
]
_EDGE_COLUMNS = [
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
]


@st.cache_data(show_spinner=False)
def _layout(
    nodes: Tuple[Tuple[str, int], ...],
    edges: Tuple[Tuple[str, str], ...],
) -> Dict[str, Tuple[float, float]]:
    """Seeded spring layout for the displayed network (cached)."""
    import networkx as nx  # local import; networkx is an optional dependency

    graph = nx.Graph()
    for drug, _degree in nodes:
        graph.add_node(drug)
    for a, b in edges:
        graph.add_edge(a, b)
    positions = nx.spring_layout(graph, seed=42, k=0.35, iterations=80)
    return {node: (float(xy[0]), float(xy[1])) for node, xy in positions.items()}


def _node_color(community: int, palette: List[str]) -> str:
    return palette[community % len(palette)]


def render() -> None:
    ui.page_header(
        "Drug Network",
        "Co-reporting network: nodes are standardised drug names, edges are "
        "reports containing both drugs. An edge is not a confirmed interaction.",
    )
    ui.disclaimer()

    if not ds.is_available("network_nodes"):
        ui.missing_output("Network nodes", "network_nodes")
        return

    nodes = ds.get_rows("network_nodes", columns=_NODE_COLUMNS)
    edges = ds.get_rows("network_edges", columns=_EDGE_COLUMNS)
    if not nodes:
        ui.empty_notice("The network node output is present but empty.")
        return

    summary = ds.get_json("network_summary") or {}
    net = summary.get("network", {}) if summary else {}
    if net:
        ui.metric_row(
            [
                {"label": "Nodes (drugs)", "value": ds.fmt_int(net.get("nodes"))},
                {"label": "Edges (pairs)", "value": ds.fmt_int(net.get("edges"))},
                {
                    "label": "Connected components",
                    "value": ds.fmt_int(net.get("connected_components")),
                },
                {
                    "label": "Largest component",
                    "value": ds.fmt_int(net.get("largest_component")),
                },
            ]
        )

    # ------------------------------------------------------- interactive graph
    st.markdown("#### Interactive drug relationship network")
    displayed_nodes = [n for n in nodes if ds.as_bool(n.get("in_displayed_network"))]
    displayed_edges = [e for e in edges if ds.as_bool(e.get("in_displayed_network"))]
    if not displayed_nodes:
        displayed_nodes = nodes[:60]
    if not displayed_edges:
        displayed_edges = edges[:240]

    rendered = False
    try:
        node_key = tuple((n.get("drug", ""), ds.as_int(n.get("degree"))) for n in displayed_nodes)
        edge_key = tuple((e.get("drug_a", ""), e.get("drug_b", "")) for e in displayed_edges)
        pos = _layout(node_key, edge_key)
        degrees = [ds.as_int(n.get("degree")) for n in displayed_nodes]
        max_degree = max(degrees) if degrees else 1
        graph_nodes = []
        for n in displayed_nodes:
            drug = n.get("drug", "")
            if drug not in pos:
                continue
            degree = ds.as_int(n.get("degree"))
            community = ds.as_int(n.get("community"))
            graph_nodes.append(
                {
                    "x": pos[drug][0],
                    "y": pos[drug][1],
                    "label": drug,
                    "size": 10 + 24 * (degree / max_degree),
                    "color": _node_color(community, ch.PALETTE),
                    "hover": (
                        f"{drug}<br>degree {degree}<br>weighted degree "
                        f"{ds.as_int(n.get('weighted_degree')):,}<br>community {community}"
                    ),
                }
            )
        graph_edges = []
        for e in displayed_edges:
            a, b = e.get("drug_a", ""), e.get("drug_b", "")
            if a in pos and b in pos:
                graph_edges.append(
                    {"x0": pos[a][0], "y0": pos[a][1], "x1": pos[b][0], "y1": pos[b][1]}
                )
        if graph_nodes:
            st.plotly_chart(
                ch.network_graph(
                    graph_nodes,
                    graph_edges,
                    title=(
                        f"Displayed network: {len(graph_nodes)} drugs, "
                        f"{len(graph_edges)} links"
                    ),
                    height=640,
                ),
                width="stretch",
            )
            st.caption(
                "Node size = degree; colour = Louvain community. Positions come "
                "from a seeded spring layout and are for readability only - "
                "distance has no clinical meaning."
            )
            rendered = True
    except Exception as exc:  # noqa: BLE001 - fall back to the pre-rendered figure
        st.caption(f"Interactive layout unavailable ({exc}); showing the pre-rendered figure.")

    if not rendered:
        _embed_pre_rendered()

    st.divider()

    # ---------------------------------------------------------- centrality
    st.markdown("#### Highly connected drugs and centrality")
    ordered = sorted(nodes, key=lambda r: ds.as_int(r.get("degree")), reverse=True)
    top_deg = ordered[:20]
    top_bet = sorted(
        nodes, key=lambda r: ds.as_float(r.get("betweenness_centrality"), 0.0), reverse=True
    )[:20]

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            ch.horizontal_bar(
                [r.get("drug", "") for r in top_deg],
                [ds.as_int(r.get("degree")) for r in top_deg],
                title="Most connected drugs (degree)",
                x_title="Distinct co-reported partners",
                color=ch.PALETTE[0],
                hover_suffix="partners",
            ),
            width="stretch",
        )
    with col2:
        st.plotly_chart(
            ch.horizontal_bar(
                [r.get("drug", "") for r in top_bet],
                [ds.as_float(r.get("betweenness_centrality"), 0.0) for r in top_bet],
                title="Highest betweenness centrality",
                x_title="Betweenness centrality",
                color=ch.PALETTE[5],
                hover_suffix="centrality",
            ),
            width="stretch",
        )

    ui.render_table(
        ordered,
        columns=_NODE_COLUMNS,
        height=340,
        caption="All nodes with degree, weighted degree, centrality and community",
    )
    ui.download_rows(
        ordered, "network_nodes.csv", label="Download network nodes", key="dl_net_nodes"
    )
    if edges:
        ui.download_rows(
            edges, "network_edges.csv", label="Download network edges", key="dl_net_edges"
        )

    st.divider()
    st.markdown("#### Communities")
    if not ds.is_available("network_communities"):
        ui.missing_output("Network communities", "network_communities")
    else:
        communities = ds.get_rows(
            "network_communities",
            columns=["community_id", "size", "internal_edges", "total_weight", "top_drug", "drugs"],
        )
        if not communities:
            ui.empty_notice("No community rows available.")
        else:
            for c in communities:
                members = c.get("drugs", "").split("|")
                preview = ", ".join(m.strip() for m in members[:12] if m.strip())
                st.markdown(
                    f"**Community {c.get('community_id')}** · size {c.get('size')} · "
                    f"{c.get('internal_edges')} internal edges · top drug "
                    f"{c.get('top_drug')}"
                )
                st.caption(preview + (" ..." if len(members) > 12 else ""))
            ui.download_rows(communities, "network_communities.csv", key="dl_net_comm")
    if net:
        ui.stat_caption(
            f"Modularity {ds.fmt_num(summary.get('communities', {}).get('modularity'), 4)} · "
            f"density {ds.fmt_num(net.get('density'), 4)} · layout "
            f"{net.get('layout')} (seed {net.get('layout_seed')})."
        )

    with st.expander("Pre-rendered interactive network figure"):
        _embed_pre_rendered()
        ui.download_file("network_html", label="Download network figure (HTML)")


def _embed_pre_rendered() -> None:
    """Embed the pipeline's pre-rendered interactive HTML, if present."""
    if not ds.is_available("network_html"):
        ui.missing_output("Network figure", "network_html")
        return
    path = ds.source_path("network_html")
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
        if hasattr(st, "iframe"):  # Streamlit >= 1.67 (st.components.v1.html deprecated)
            st.iframe(html, height=680)
        else:
            st.components.v1.html(html, height=680, scrolling=True)
    except Exception:  # noqa: BLE001
        st.caption("Could not embed the pre-rendered network figure.")
