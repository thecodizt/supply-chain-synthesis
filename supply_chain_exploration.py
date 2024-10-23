import streamlit as st
import os
import json
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from streamlit import session_state as state
import plotly.express as px
import matplotlib.pyplot as plt


def load_graph(simulation_path, timestamp):
    timestamp_file = os.path.join(simulation_path, f"t_{timestamp}.json")

    try:
        with open(timestamp_file, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        G = nx.Graph()
        G.add_nodes_from((n, d) for n, d in graph_data["nodes"].items())
        G.add_edges_from(graph_data["edges"])

        return G
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON in {timestamp_file}: {str(e)}")
    except KeyError as e:
        st.error(f"Missing key in graph data: {str(e)}")
    except UnicodeDecodeError as e:
        st.error(f"Encoding error in {timestamp_file}: {str(e)}")

    return None


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


def display_node_properties(G):
    st.subheader("Node Properties")
    node_types = set(nx.get_node_attributes(G, "type").values())
    selected_type = st.selectbox("Select node type", list(node_types))

    nodes_of_type = [n for n, d in G.nodes(data=True) if d.get("type") == selected_type]
    if nodes_of_type:
        sample_node = G.nodes[nodes_of_type[0]]
        properties = list(sample_node.keys())
        st.write(f"Properties for {selected_type}:")
        st.write(", ".join(properties))
    else:
        st.write(f"No nodes of type {selected_type} found.")


def compare_graphs(G1, G2):
    added_nodes = set(G2.nodes()) - set(G1.nodes())
    removed_nodes = set(G1.nodes()) - set(G2.nodes())
    common_nodes = set(G1.nodes()) & set(G2.nodes())

    updated_nodes = []
    for node in common_nodes:
        if G1.nodes[node] != G2.nodes[node]:
            updated_nodes.append(node)

    added_edges = set(G2.edges()) - set(G1.edges())
    removed_edges = set(G1.edges()) - set(G2.edges())

    return {
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "updated_nodes": updated_nodes,
        "added_edges": added_edges,
        "removed_edges": removed_edges,
    }


def visualize_graph_changes(G1, G2, changes):
    pos = nx.spring_layout(nx.compose(G1, G2))

    def create_edge_trace(edges, color, dash):
        edge_x, edge_y = [], []
        for edge in edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        return go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color=color, dash=dash),
            hoverinfo="none",
            mode="lines",
        )

    # Create separate traces for different edge types
    unchanged_edges = set(G2.edges()) - set(changes["added_edges"])
    edge_trace_unchanged = create_edge_trace(unchanged_edges, "#888", "dot")
    edge_trace_added = create_edge_trace(changes["added_edges"], "green", "solid")
    edge_trace_removed = create_edge_trace(changes["removed_edges"], "red", "dash")

    node_x, node_y = [], []
    node_colors = []
    node_sizes = []
    node_text = []
    for node in G2.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if node in changes["added_nodes"]:
            node_colors.append("green")
            node_sizes.append(15)
            node_text.append(f"Added: {node}")
        elif node in changes["updated_nodes"]:
            node_colors.append("yellow")
            node_sizes.append(12)
            node_text.append(f"Updated: {node}")
        else:
            node_colors.append("blue")
            node_sizes.append(10)
            node_text.append(f"Unchanged: {node}")

    for node in changes["removed_nodes"]:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append("red")
        node_sizes.append(15)
        node_text.append(f"Removed: {node}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(color=node_colors, size=node_sizes, line_width=2),
        text=node_text,
    )

    fig = go.Figure(
        data=[edge_trace_unchanged, edge_trace_added, edge_trace_removed, node_trace],
        layout=go.Layout(
            title="Graph Changes Visualization",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    # Update traces for legend
    fig.data[0].name = "Unchanged Edge"
    fig.data[1].name = "Added Edge"
    fig.data[2].name = "Removed Edge"

    st.plotly_chart(fig, use_container_width=True)


def visualize_node_property_changes(G1, G2, updated_nodes):
    changes = []
    for node in updated_nodes:
        old_props = G1.nodes[node]
        new_props = G2.nodes[node]
        node_changes = {"node": node}
        for key in set(old_props.keys()) | set(new_props.keys()):
            if key not in old_props:
                node_changes[key] = ("Added", new_props[key])
            elif key not in new_props:
                node_changes[key] = ("Removed", old_props[key])
            elif old_props[key] != new_props[key]:
                node_changes[key] = ("Changed", f"{old_props[key]} -> {new_props[key]}")
        changes.append(node_changes)

    df = pd.DataFrame(changes)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns), fill_color="paleturquoise", align="left"
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    fig.update_layout(title="Node Property Changes")
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
        timestamps = []
        for filename in os.listdir(simulation_path):
            if filename.startswith("t_") and filename.endswith(".json"):
                try:
                    timestamp = int(filename.split("_")[1].split(".")[0])
                    timestamps.append(timestamp)
                except ValueError:
                    continue
        timestamps.sort()

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
            # Load graph
            if state.graph is None:
                state.graph = load_graph(simulation_path, state.selected_timestamp)

            if state.graph:
                # Display basic graph info
                st.write(f"Number of nodes: {state.graph.number_of_nodes()}")
                st.write(f"Number of edges: {state.graph.number_of_edges()}")

                # Visualize graph
                st.subheader("Graph Visualization")
                visualize_graph(state.graph)

                # Node type distribution
                st.subheader("Node Type Distribution")
                node_types = [
                    data.get("type", "Unknown")
                    for _, data in state.graph.nodes(data=True)
                ]
                type_counts = pd.Series(node_types).value_counts()
                st.bar_chart(type_counts)

                # Display node properties
                display_node_properties(state.graph)

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

                # Add a section for visualizing changes
                st.header("Graph Changes Visualization")
                timestamps = sorted(timestamps)
                current_index = timestamps.index(state.selected_timestamp)

                if current_index > 0:
                    previous_timestamp = timestamps[current_index - 1]
                    previous_graph = load_graph(simulation_path, previous_timestamp)
                    current_graph = state.graph

                    if previous_graph and current_graph:
                        changes = compare_graphs(previous_graph, current_graph)

                        st.subheader(
                            f"Changes from t_{previous_timestamp} to t_{state.selected_timestamp}"
                        )
                        st.write(f"Added nodes: {len(changes['added_nodes'])}")
                        st.write(f"Removed nodes: {len(changes['removed_nodes'])}")
                        st.write(f"Updated nodes: {len(changes['updated_nodes'])}")
                        st.write(f"Added edges: {len(changes['added_edges'])}")
                        st.write(f"Removed edges: {len(changes['removed_edges'])}")

                        visualize_graph_changes(previous_graph, current_graph, changes)

                        st.subheader("Node Property Changes")
                        visualize_node_property_changes(
                            previous_graph, current_graph, changes["updated_nodes"]
                        )


if __name__ == "__main__":
    main()
