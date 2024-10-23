import networkx as nx
import os
import json
import streamlit as st
from typing import List, Dict, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px


def load_2d_graph(data_path: str, timestamp: int) -> nx.Graph:
    timestamp_file = os.path.join(data_path, f"t_{timestamp}.json")

    try:
        with open(timestamp_file, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        G = nx.Graph()
        G.add_nodes_from((n, d) for n, d in graph_data["nodes"].items())
        G.add_edges_from(graph_data["edges"])

        return G
    except FileNotFoundError:
        st.error(f"Timestamp file not found: {timestamp_file}")
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON in {timestamp_file}: {str(e)}")
    except KeyError as e:
        st.error(f"Missing key in graph data: {str(e)}")

    return None


def load_3d_graphs(
    data_path: str, start_timestamp: int, end_timestamp: int
) -> List[nx.Graph]:
    graphs = []
    for timestamp in range(start_timestamp, end_timestamp + 1):
        G = load_2d_graph(data_path, timestamp)
        if G is not None:
            graphs.append(G)
    return graphs


def get_node_info(G: nx.Graph, node_id: str) -> Dict[str, Any]:
    return G.nodes[node_id]


def get_neighbors(G: nx.Graph, node_id: str) -> List[str]:
    return list(G.neighbors(node_id))


def get_shortest_path(G: nx.Graph, source: str, target: str) -> List[str]:
    try:
        return nx.shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return []


def get_node_degree(G: nx.Graph, node_id: str) -> int:
    return G.degree[node_id]


def get_connected_components(G: nx.Graph) -> List[List[str]]:
    return list(nx.connected_components(G))


def get_centrality_measures(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    return {
        "degree": nx.degree_centrality(G),
        "closeness": nx.closeness_centrality(G),
        "betweenness": nx.betweenness_centrality(G),
    }


def get_subgraph_by_type(G: nx.Graph, node_type: str) -> nx.Graph:
    nodes = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
    return G.subgraph(nodes)


def get_failed_parts(G: nx.Graph) -> List[str]:
    return [
        n
        for n, d in G.nodes(data=True)
        if d.get("type") == "Part Instance" and d.get("status") == "Failed"
    ]


def get_product_composition(G: nx.Graph, product_id: str) -> Dict[str, int]:
    composition = {}
    for neighbor in G.neighbors(product_id):
        node_data = G.nodes[neighbor]
        if node_data.get("type") == "Part" or node_data.get("type") == "Part Instance":
            part_type = node_data.get("original_id", node_data.get("id"))
            composition[part_type] = composition.get(part_type, 0) + 1
    return composition


def get_supplier_products(G: nx.Graph, supplier_id: str) -> List[str]:
    return [n for n in G.neighbors(supplier_id) if G.nodes[n].get("type") == "Product"]


def get_warehouse_inventory(G: nx.Graph, warehouse_id: str) -> Dict[str, int]:
    inventory = {}
    for neighbor in G.neighbors(warehouse_id):
        node_data = G.nodes[neighbor]
        if node_data.get("type") == "Product Instance":
            product_type = node_data.get("original_id")
            inventory[product_type] = inventory.get(product_type, 0) + 1
    return inventory


def get_node_types(G: nx.Graph) -> List[str]:
    return list(set(data["type"] for _, data in G.nodes(data=True)))


def get_nodes_by_type(G: nx.Graph, node_type: str) -> List[str]:
    return [node for node, data in G.nodes(data=True) if data.get("type") == node_type]


def get_node_color(node_type: str) -> str:
    color_map = {
        "Business Unit": "#1f77b4",
        "Product": "#ff7f0e",
        "Product Instance": "#2ca02c",
        "Part": "#d62728",
        "Part Instance": "#9467bd",
        "Supplier": "#8c564b",
        "Warehouse": "#e377c2",
        "Purchase Order": "#7f7f7f",
    }
    return color_map.get(
        node_type, "#17becf"
    )  # Default color if type is not in the map


def visualize_graph(
    G: nx.Graph, highlight_nodes: List[str] = None, highlight_edges: List[tuple] = None
):
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_type = G.nodes[node].get("type", "Unknown")
        node_colors.append(get_node_color(node_type))
        node_text.append(
            f"Node: {node}<br>Type: {node_type}<br># of connections: {G.degree[node]}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(size=10, color=node_colors, line_width=2),
    )

    node_trace.text = node_text

    # Highlight specific nodes and edges if provided
    highlight_node_trace = None
    highlight_edge_trace = None
    if highlight_nodes:
        h_node_x = [pos[node][0] for node in highlight_nodes]
        h_node_y = [pos[node][1] for node in highlight_nodes]
        highlight_node_trace = go.Scatter(
            x=h_node_x,
            y=h_node_y,
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            hoverinfo="text",
            text=[f"Highlighted Node: {node}" for node in highlight_nodes],
        )

    if highlight_edges:
        h_edge_x = []
        h_edge_y = []
        for edge in highlight_edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            h_edge_x.extend([x0, x1, None])
            h_edge_y.extend([y0, y1, None])
        highlight_edge_trace = go.Scatter(
            x=h_edge_x,
            y=h_edge_y,
            line=dict(width=2, color="red"),
            hoverinfo="none",
            mode="lines",
        )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace])
    if highlight_node_trace:
        fig.add_trace(highlight_node_trace)
    if highlight_edge_trace:
        fig.add_trace(highlight_edge_trace)

    fig.update_layout(
        title="Network Graph",
        titlefont_size=16,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig


def visualize_shortest_path(G: nx.Graph, path: List[str]):
    highlight_edges = list(zip(path[:-1], path[1:]))
    return visualize_graph(G, highlight_nodes=path, highlight_edges=highlight_edges)


def visualize_neighbors(G: nx.Graph, node: str):
    neighbors = list(G.neighbors(node))
    highlight_edges = [(node, neighbor) for neighbor in neighbors]
    return visualize_graph(
        G, highlight_nodes=[node] + neighbors, highlight_edges=highlight_edges
    )


def visualize_product_composition(composition: Dict[str, int]):
    fig = px.bar(
        x=list(composition.keys()),
        y=list(composition.values()),
        labels={"x": "Part Types", "y": "Quantity"},
        title="Product Composition",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def visualize_warehouse_inventory(inventory: Dict[str, int]):
    fig = px.bar(
        x=list(inventory.keys()),
        y=list(inventory.values()),
        labels={"x": "Product Types", "y": "Quantity"},
        title="Warehouse Inventory",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def visualize_centrality_measures(centrality: Dict[str, Dict[str, float]]):
    fig = go.Figure()
    for measure, values in centrality.items():
        fig.add_trace(go.Histogram(x=list(values.values()), name=measure))

    fig.update_layout(
        title="Centrality Measures",
        xaxis_title="Centrality Value",
        yaxis_title="Frequency",
        barmode="overlay",
    )
    fig.update_traces(opacity=0.75)
    return fig


def get_node_evolution(
    graphs: List[nx.Graph], node_id: str
) -> Dict[int, Dict[str, Any]]:
    evolution = {}
    for i, G in enumerate(graphs):
        if node_id in G.nodes:
            evolution[i] = G.nodes[node_id]
    return evolution


def get_edge_evolution(
    graphs: List[nx.Graph], source: str, target: str
) -> Dict[int, bool]:
    evolution = {}
    for i, G in enumerate(graphs):
        evolution[i] = G.has_edge(source, target)
    return evolution


def visualize_node_evolution(evolution: Dict[int, Dict[str, Any]]):
    fig = go.Figure()
    for attribute in evolution[list(evolution.keys())[0]].keys():
        if attribute != "type":
            values = [
                data[attribute] for data in evolution.values() if attribute in data
            ]
            fig.add_trace(
                go.Scatter(
                    x=list(evolution.keys()),
                    y=values,
                    mode="lines+markers",
                    name=attribute,
                )
            )

    fig.update_layout(
        title="Node Attribute Evolution",
        xaxis_title="Timestamp",
        yaxis_title="Attribute Value",
    )
    return fig


def visualize_edge_evolution(evolution: Dict[int, bool]):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(evolution.keys()),
            y=[int(v) for v in evolution.values()],
            mode="lines+markers",
            name="Edge Existence",
        )
    )
    fig.update_layout(
        title="Edge Evolution",
        xaxis_title="Timestamp",
        yaxis_title="Edge Exists (1) or Not (0)",
    )
    return fig


# Streamlit UI
def main():
    st.title("Supply Chain Graph Query Tool")

    data_path = st.text_input("Enter the data path:", value="data/run1")

    # Toggle between 2D and 3D
    graph_mode = st.radio("Select graph mode:", ("2D", "3D"))

    if graph_mode == "2D":
        timestamp = st.number_input(
            "Enter the timestamp:", min_value=1, value=1, step=1
        )

        if st.button("Load Graph"):
            G = load_2d_graph(data_path, timestamp)
            if G is not None:
                st.session_state.graph = G
                st.session_state.graph_mode = "2D"
                st.success(
                    f"2D Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
                )
            else:
                st.error("Failed to load the graph.")

    else:  # 3D mode
        start_timestamp = st.number_input(
            "Enter the start timestamp:", min_value=1, value=1, step=1
        )
        end_timestamp = st.number_input(
            "Enter the end timestamp:",
            min_value=start_timestamp,
            value=start_timestamp + 2,
            step=1,
        )

        if st.button("Load Graphs"):
            graphs = load_3d_graphs(data_path, start_timestamp, end_timestamp)
            if graphs:
                st.session_state.graphs = graphs
                st.session_state.graph_mode = "3D"
                st.success(f"3D Graphs loaded with {len(graphs)} timestamps")
            else:
                st.error("Failed to load the graphs.")

    if "graph_mode" in st.session_state:
        st.header("Query Options")

        if st.session_state.graph_mode == "2D":
            G = st.session_state.graph
            query_type = st.selectbox(
                "Select a query type:",
                [
                    "Node Info",
                    "Neighbors",
                    "Shortest Path",
                    "Node Degree",
                    "Connected Components",
                    "Centrality Measures",
                    "Subgraph by Type",
                    "Failed Parts",
                    "Product Composition",
                    "Supplier Products",
                    "Warehouse Inventory",
                ],
            )

            if query_type in ["Node Info", "Neighbors", "Node Degree"]:
                node_type = st.selectbox("Select node type:", get_node_types(G))
                node_id = st.selectbox(
                    "Select node ID:", get_nodes_by_type(G, node_type)
                )

                if query_type == "Node Info":
                    if st.button("Get Node Info"):
                        info = get_node_info(G, node_id)
                        st.json(info)

                elif query_type == "Neighbors":
                    if st.button("Get Neighbors"):
                        neighbors = get_neighbors(G, node_id)
                        st.write(neighbors)
                        fig = visualize_neighbors(G, node_id)
                        st.plotly_chart(fig)

                elif query_type == "Node Degree":
                    if st.button("Get Node Degree"):
                        degree = get_node_degree(G, node_id)
                        st.write(f"Degree: {degree}")

            elif query_type == "Shortest Path":
                source_type = st.selectbox(
                    "Select source node type:", get_node_types(G), key="source_type"
                )
                source = st.selectbox(
                    "Select source node ID:",
                    get_nodes_by_type(G, source_type),
                    key="source",
                )
                target_type = st.selectbox(
                    "Select target node type:", get_node_types(G), key="target_type"
                )
                target = st.selectbox(
                    "Select target node ID:",
                    get_nodes_by_type(G, target_type),
                    key="target",
                )
                if st.button("Find Shortest Path"):
                    path = get_shortest_path(G, source, target)
                    st.write(path)
                    if path:
                        fig = visualize_shortest_path(G, path)
                        st.plotly_chart(fig)
                    else:
                        st.write("No path found between the selected nodes.")

            elif query_type == "Connected Components":
                if st.button("Get Connected Components"):
                    components = get_connected_components(G)
                    st.write(f"Number of connected components: {len(components)}")
                    st.write("Components:", components)

            elif query_type == "Centrality Measures":
                if st.button("Calculate Centrality Measures"):
                    centrality = get_centrality_measures(G)
                    st.write(centrality)
                    fig = visualize_centrality_measures(centrality)
                    st.plotly_chart(fig)

            elif query_type == "Subgraph by Type":
                node_type = st.selectbox("Select node type:", get_node_types(G))
                if st.button("Get Subgraph"):
                    subgraph = get_subgraph_by_type(G, node_type)
                    st.write(
                        f"Subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges"
                    )

            elif query_type == "Failed Parts":
                if st.button("Get Failed Parts"):
                    failed_parts = get_failed_parts(G)
                    st.write(failed_parts)

            elif query_type == "Product Composition":
                product_type = st.selectbox(
                    "Select product type:", ["Product", "Product Instance"]
                )
                product_id = st.selectbox(
                    "Select product ID:", get_nodes_by_type(G, product_type)
                )
                if st.button("Get Product Composition"):
                    composition = get_product_composition(G, product_id)
                    st.write(composition)
                    if composition:
                        fig = visualize_product_composition(composition)
                        st.plotly_chart(fig)
                    else:
                        st.write("No composition data available for this product.")

            elif query_type == "Supplier Products":
                supplier_id = st.selectbox(
                    "Select supplier ID:", get_nodes_by_type(G, "Supplier")
                )
                if st.button("Get Supplier Products"):
                    products = get_supplier_products(G, supplier_id)
                    st.write(products)

            elif query_type == "Warehouse Inventory":
                warehouse_id = st.selectbox(
                    "Select warehouse ID:", get_nodes_by_type(G, "Warehouse")
                )
                if st.button("Get Warehouse Inventory"):
                    inventory = get_warehouse_inventory(G, warehouse_id)
                    st.write(inventory)
                    if inventory:
                        fig = visualize_warehouse_inventory(inventory)
                        st.plotly_chart(fig)
                    else:
                        st.write("No inventory data available for this warehouse.")

        else:  # 3D mode
            graphs = st.session_state.graphs
            query_type = st.selectbox(
                "Select a query type:", ["Node Evolution", "Edge Evolution"]
            )

            if query_type == "Node Evolution":
                node_type = st.selectbox("Select node type:", get_node_types(graphs[0]))
                node_id = st.selectbox(
                    "Select node ID:", get_nodes_by_type(graphs[0], node_type)
                )
                if st.button("Get Node Evolution"):
                    evolution = get_node_evolution(graphs, node_id)
                    st.write(evolution)
                    fig = visualize_node_evolution(evolution)
                    st.plotly_chart(fig)

            elif query_type == "Edge Evolution":
                source_type = st.selectbox(
                    "Select source node type:",
                    get_node_types(graphs[0]),
                    key="source_type",
                )
                source = st.selectbox(
                    "Select source node ID:",
                    get_nodes_by_type(graphs[0], source_type),
                    key="source",
                )
                target_type = st.selectbox(
                    "Select target node type:",
                    get_node_types(graphs[0]),
                    key="target_type",
                )
                target = st.selectbox(
                    "Select target node ID:",
                    get_nodes_by_type(graphs[0], target_type),
                    key="target",
                )
                if st.button("Get Edge Evolution"):
                    evolution = get_edge_evolution(graphs, source, target)
                    st.write(evolution)
                    fig = visualize_edge_evolution(evolution)
                    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
