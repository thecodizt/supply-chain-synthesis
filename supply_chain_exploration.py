import streamlit as st
import os
import json
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from streamlit import session_state as state
import plotly.express as px


def load_graph(simulation_path, timestamp):
    G = nx.Graph()
    timestamp_path = os.path.join(simulation_path, f"t_{timestamp}")

    for root, dirs, files in os.walk(timestamp_path):
        if "node_data.json" in files:
            try:
                with open(
                    os.path.join(root, "node_data.json"), "r", encoding="utf-8"
                ) as f:
                    node_data = json.load(f)
                    G.add_node(node_data["id"], **node_data["data"])
            except json.JSONDecodeError as e:
                st.error(
                    f"Error decoding JSON in {os.path.join(root, 'node_data.json')}: {str(e)}"
                )
            except KeyError as e:
                st.error(f"Missing key in node data: {str(e)}")
            except UnicodeDecodeError as e:
                st.error(
                    f"Encoding error in {os.path.join(root, 'node_data.json')}: {str(e)}"
                )

        if "edge_data.json" in files:
            try:
                with open(
                    os.path.join(root, "edge_data.json"), "r", encoding="utf-8"
                ) as f:
                    edge_data = json.load(f)
                    for edge in edge_data:
                        G.add_edge(edge["source"], edge["target"])
            except json.JSONDecodeError as e:
                st.error(
                    f"Error decoding JSON in {os.path.join(root, 'edge_data.json')}: {str(e)}"
                )
            except KeyError as e:
                st.error(f"Missing key in edge data: {str(e)}")
            except UnicodeDecodeError as e:
                st.error(
                    f"Encoding error in {os.path.join(root, 'edge_data.json')}: {str(e)}"
                )

    return G


def visualize_graph(G):
    pos = nx.spring_layout(G, k=0.5, iterations=50)

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
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_types = [data.get("type", "Unknown") for node, data in G.nodes(data=True)]
    unique_types = list(set(node_types))
    color_map = {
        t: f"rgb({hash(t) % 256}, {(hash(t) >> 8) % 256}, {(hash(t) >> 16) % 256})"
        for t in unique_types
    }
    node_colors = [color_map[t] for t in node_types]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(showscale=False, color=node_colors, size=10, line_width=2),
    )

    node_text = [
        f"ID: {node}<br>Type: {data.get('type', 'Unknown')}"
        for node, data in G.nodes(data=True)
    ]
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Supply Chain Graph",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def load_schema(simulation_path):
    schema_path = os.path.join(simulation_path, "metadata", "schema.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            return json.load(f)
    else:
        st.error(f"Schema file not found: {schema_path}")
        return None


def load_data(simulation_path):
    data_path = os.path.join(simulation_path, "metadata", "data.json")
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        st.error(f"Data file not found: {data_path}")
        return None


def visualize_schema(schema):
    G = nx.DiGraph()
    edge_labels = {}

    for entity, relations in schema.items():
        G.add_node(entity)
        for relation, targets in relations.items():
            if isinstance(targets, list):
                for target in targets:
                    G.add_edge(entity, target)
                    edge_labels[(entity, target)] = relation
            elif isinstance(targets, (int, str)):
                G.add_node(f"{entity}_{relation}")
                G.add_edge(entity, f"{entity}_{relation}")
                edge_labels[(entity, f"{entity}_{relation}")] = relation

    pos = nx.spring_layout(G)

    edge_x, edge_y = [], []
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

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            colorscale="YlGnBu",
            size=10,
            line_width=2,
        ),
        text=[node for node in G.nodes()],
        textposition="top center",
    )

    annotations = []
    for edge, label in edge_labels.items():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                xref="x",
                yref="y",
                text=label,
                showarrow=False,
                font=dict(size=8),
            )
        )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Schema Visualization",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def visualize_data(data):
    # Prepare data for treemap
    treemap_data = []
    for entity_type, entities in data.items():
        for entity_id, entity_data in entities.items():
            treemap_data.append(
                {
                    "id": f"{entity_type}|{entity_id}",
                    "parent": entity_type,
                    "name": entity_id,
                    "value": 1,
                }
            )
        treemap_data.append(
            {
                "id": entity_type,
                "parent": "",
                "name": entity_type,
                "value": len(entities),
            }
        )

    # Create treemap
    fig = px.treemap(
        treemap_data,
        ids="id",
        names="name",
        parents="parent",
        values="value",
        title="Data Overview - Hierarchical Structure",
    )

    fig.update_traces(
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    )

    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

    st.plotly_chart(fig, use_container_width=True)

    # Add entity counts
    st.subheader("Entity Counts")
    entity_counts = {
        entity_type: len(entities) for entity_type, entities in data.items()
    }
    st.write(entity_counts)


def visualize_data_graph(data):
    G = nx.Graph()
    edge_labels = {}

    # Create nodes and edges
    for entity_type, entities in data.items():
        for entity_id, entity_data in entities.items():
            G.add_node(entity_id, type=entity_type)
            for relation, related_entities in entity_data.items():
                if isinstance(related_entities, dict):
                    for related_id, quantity in related_entities.items():
                        G.add_edge(entity_id, related_id)
                        edge_labels[(entity_id, related_id)] = f"{relation}: {quantity}"
                elif isinstance(related_entities, list):
                    for related_id in related_entities:
                        G.add_edge(entity_id, related_id)
                        edge_labels[(entity_id, related_id)] = relation

    # Create hierarchical layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Adjust y-coordinates based on node type to create layers
    node_types = list(set(data.keys()))
    type_to_layer = {t: i for i, t in enumerate(node_types)}
    max_layer = len(node_types) - 1

    for node, coords in pos.items():
        node_type = G.nodes[node]["type"]
        layer = type_to_layer[node_type]
        pos[node] = (
            coords[0],
            (layer / max_layer) * 0.9 + 0.05,
        )  # Scale to 0.05-0.95 range

    edge_x, edge_y = [], []
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

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_types = [G.nodes[node]["type"] for node in G.nodes()]
    unique_types = list(set(node_types))
    color_map = {
        t: f"rgb({hash(t) % 256}, {(hash(t) >> 8) % 256}, {(hash(t) >> 16) % 256})"
        for t in unique_types
    }
    node_colors = [color_map[G.nodes[node]["type"]] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            color=node_colors,
            size=10,
            line_width=2,
        ),
    )

    node_text = [f"ID: {node}<br>Type: {G.nodes[node]['type']}" for node in G.nodes()]
    node_trace.text = node_text

    annotations = [
        dict(
            x=(pos[edge[0]][0] + pos[edge[1]][0]) / 2,
            y=(pos[edge[0]][1] + pos[edge[1]][1]) / 2,
            xref="x",
            yref="y",
            text=label,
            showarrow=False,
            font=dict(size=8),
        )
        for edge, label in edge_labels.items()
    ]

    # Add annotations for node types (layers)
    for node_type, layer in type_to_layer.items():
        annotations.append(
            dict(
                x=-0.05,
                y=(layer / max_layer) * 0.9 + 0.05,
                xref="paper",
                yref="y",
                text=node_type,
                showarrow=False,
                font=dict(size=12),
                textangle=-90,
            )
        )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Data Graph Visualization (Hierarchical)",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=80, r=20, t=40),
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Supply Chain Graph Explorer")

    # Initialize session state variables
    if "selected_simulation" not in state:
        state.selected_simulation = None
    if "selected_timestamp" not in state:
        state.selected_timestamp = None
    if "graph" not in state:
        state.graph = None

    # Get list of simulations
    data_path = "data"
    simulations = [
        d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
    ]

    # Select simulation
    selected_simulation = st.selectbox(
        "Select a simulation", simulations, key="simulation_select"
    )

    if selected_simulation != state.selected_simulation:
        state.selected_simulation = selected_simulation
        state.selected_timestamp = None
        state.graph = None

    if state.selected_simulation:
        simulation_path = os.path.join(data_path, state.selected_simulation)

        # Visualize schema
        st.header("Schema Visualization")
        schema = load_schema(simulation_path)
        if schema:
            visualize_schema(schema)

        # Visualize data overview
        st.header("Data Overview")
        data = load_data(simulation_path)
        if data:
            visualize_data(data)
            st.header("Data Graph Visualization")
            visualize_data_graph(data)

        # Get list of timestamps
        timestamps = [
            d.split("_")[1] for d in os.listdir(simulation_path) if d.startswith("t_")
        ]
        timestamps.sort(key=int)

        if not timestamps:
            st.error("No timestamps found for the selected simulation.")
            return

        # Select timestamp
        selected_timestamp = st.selectbox(
            "Select a timestamp", timestamps, key="timestamp_select"
        )

        if selected_timestamp != state.selected_timestamp:
            state.selected_timestamp = selected_timestamp
            state.graph = None

        if state.selected_timestamp:
            # Check if the timestamp directory exists
            timestamp_path = os.path.join(
                simulation_path, f"t_{state.selected_timestamp}"
            )
            if not os.path.exists(timestamp_path):
                st.error(f"Timestamp directory not found: {timestamp_path}")
                return

            # Load graph
            if state.graph is None:
                state.graph = load_graph(simulation_path, state.selected_timestamp)

            # Display basic graph info
            st.write(f"Number of nodes: {state.graph.number_of_nodes()}")
            st.write(f"Number of edges: {state.graph.number_of_edges()}")

            # Visualize graph
            st.subheader("Graph Visualization")
            visualize_graph(state.graph)

            # Node type distribution
            st.subheader("Node Type Distribution")
            node_types = [
                data.get("type", "Unknown") for _, data in state.graph.nodes(data=True)
            ]
            type_counts = pd.Series(node_types).value_counts()
            st.bar_chart(type_counts)

            # Node explorer
            st.subheader("Node Explorer")
            selected_node = st.selectbox("Select a node", list(state.graph.nodes()))
            st.json(state.graph.nodes[selected_node])

            # Edge explorer
            st.subheader("Edge Explorer")
            edges = list(state.graph.edges())
            if edges:
                selected_edge = st.selectbox(
                    "Select an edge", [f"{u} - {v}" for u, v in edges]
                )
                u, v = selected_edge.split(" - ")
                st.write(f"Edge data: {state.graph.get_edge_data(u, v)}")
            else:
                st.write("No edges in the graph.")


if __name__ == "__main__":
    main()
