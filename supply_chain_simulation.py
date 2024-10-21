import networkx as nx
import random
import uuid
import json
import os
from collections import Counter
import errno
import hashlib
import sys


class SupplyChainSimulation:
    def __init__(self, config):
        self.config = config
        self.graph = nx.Graph()
        self.timestamp = 0
        self.data_path = os.path.abspath(config.get("data_path", "supply_chain_data"))
        self.metadata_path = os.path.join(self.data_path, "metadata")

        # Ensure base directories exist
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)

        self.schema = self.load_or_create_schema()
        self.data = self.load_or_create_data()
        self.initialize_graph()

    def load_or_create_schema(self):
        schema_path = os.path.join(self.metadata_path, "schema.json")
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                return json.load(f)
        else:
            schema = self.create_schema()
            os.makedirs(self.metadata_path, exist_ok=True)
            with open(schema_path, "w") as f:
                json.dump(schema, f, indent=2)
            return schema

    def create_schema(self):
        return {
            "Business Unit": {"products": ["Product"]},
            "Product": {
                "parts": ["Part"],
                "suppliers": ["Supplier"],
                "warehouses": ["Warehouse"],
            },
            "Part": {
                "subparts": ["Part"],
                "max_depth": 5,  # Maximum depth of part hierarchy
                "max_subparts": 3,  # Maximum number of subparts for each part
            },
            "Supplier": {},
            "Warehouse": {},
        }

    def load_or_create_data(self):
        data_path = os.path.join(self.metadata_path, "data.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                return json.load(f)
        else:
            data = self.create_data()
            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)
            return data

    def create_data(self):
        data = {
            "Business Units": {},
            "Products": {},
            "Parts": {},
            "Suppliers": {},
            "Warehouses": {},
        }

        # Create Business Units
        for i in range(self.config["num_business_units"]):
            bu_id = f"BU_{i+1}"
            data["Business Units"][bu_id] = {
                "name": f"Business Unit {i+1}",
                "products": [],
            }

        # Create Parts with hierarchy based on schema
        self.create_parts_hierarchy(data)

        # Create Products
        for i in range(self.config["num_products"]):
            product_id = f"P_{i+1}"
            business_unit = random.choice(list(data["Business Units"].keys()))
            data["Products"][product_id] = {
                "name": f"Product {i+1}",
                "expected_life": random.randint(5000, 20000),
                "parts": self.assign_parts_to_product(data["Parts"]),
                "business_unit": business_unit,
            }
            data["Business Units"][business_unit]["products"].append(product_id)

        # Create Suppliers
        for i in range(self.config["num_suppliers"]):
            supplier_id = f"S_{i+1}"
            data["Suppliers"][supplier_id] = {
                "name": f"Supplier {i+1}",
                "location": f"Location {i+1}",
                "products": random.sample(
                    list(data["Products"].keys()), random.randint(1, 5)
                ),
            }

        # Create Warehouses
        for i in range(self.config["num_warehouses"]):
            warehouse_id = f"W_{i+1}"
            data["Warehouses"][warehouse_id] = {
                "name": f"Warehouse {i+1}",
                "location": f"Location {i+1}",
                "products": random.sample(
                    list(data["Products"].keys()), random.randint(5, 20)
                ),
            }

        return data

    def create_parts_hierarchy(self, data):
        max_depth = self.schema["Part"]["max_depth"]
        max_subparts = self.schema["Part"]["max_subparts"]

        def create_part(depth=0):
            part_id = f"PT_{len(data['Parts']) + 1}"
            data["Parts"][part_id] = {
                "name": f"Part Type {len(data['Parts']) + 1}",
                "expected_life": random.randint(1000, 10000),
                "subparts": {},
            }

            if (
                depth < max_depth and random.random() < 0.5
            ):  # 50% chance of having subparts
                num_subparts = random.randint(1, max_subparts)
                available_subparts = [p for p in data["Parts"] if p != part_id]
                if available_subparts:
                    subparts = random.sample(
                        available_subparts, min(num_subparts, len(available_subparts))
                    )
                    for subpart in subparts:
                        data["Parts"][part_id]["subparts"][subpart] = random.randint(
                            1, 5
                        )

            return part_id

        for _ in range(self.config["num_part_types"]):
            create_part()

    def assign_parts_to_product(self, parts):
        num_parts = random.randint(1, 5)
        selected_parts = random.sample(list(parts.keys()), min(num_parts, len(parts)))
        return {part: random.randint(1, 10) for part in selected_parts}

    def initialize_graph(self):
        # Business Units and Products
        for bu_id, bu_data in self.data["Business Units"].items():
            self.add_node(bu_id, "Business Unit", bu_data)
            for product_id in bu_data["products"]:
                self.add_node(product_id, "Product", self.data["Products"][product_id])
                self.graph.add_edge(bu_id, product_id)

        # Parts
        for part_id, part_data in self.data["Parts"].items():
            self.add_node(part_id, "Part", part_data)
            for subpart_id, quantity in part_data.get("subparts", {}).items():
                self.add_node(subpart_id, "Part", self.data["Parts"][subpart_id])
                self.graph.add_edge(part_id, subpart_id, quantity=quantity)

        # Products and their parts
        for product_id, product_data in self.data["Products"].items():
            for part_id, quantity in product_data.get("parts", {}).items():
                self.graph.add_edge(product_id, part_id, quantity=quantity)

        # Suppliers
        for supplier_id, supplier_data in self.data["Suppliers"].items():
            self.add_node(supplier_id, "Supplier", supplier_data)
            for product_id in supplier_data["products"]:
                self.graph.add_edge(supplier_id, product_id)

        # Warehouses
        for warehouse_id, warehouse_data in self.data["Warehouses"].items():
            self.add_node(warehouse_id, "Warehouse", warehouse_data)
            for product_id in warehouse_data["products"]:
                self.graph.add_edge(warehouse_id, product_id)

        # Verify all nodes have a 'type' attribute
        for node, data in self.graph.nodes(data=True):
            if "type" not in data:
                print(f"Warning: Node {node} initialized without a 'type' attribute.")
                # Attempt to infer the type based on the node ID prefix
                if node.startswith("BU_"):
                    self.graph.nodes[node]["type"] = "Business Unit"
                elif node.startswith("P_"):
                    self.graph.nodes[node]["type"] = "Product"
                elif node.startswith("PT_"):
                    self.graph.nodes[node]["type"] = "Part"
                elif node.startswith("S_"):
                    self.graph.nodes[node]["type"] = "Supplier"
                elif node.startswith("W_"):
                    self.graph.nodes[node]["type"] = "Warehouse"
                else:
                    print(f"Error: Unable to infer type for node {node}")

    def add_node(self, node_id, node_type, node_data):
        if node_id not in self.graph:
            node_data = (
                node_data.copy()
            )  # Create a copy to avoid modifying the original data
            node_data["type"] = node_type  # Add node type as an attribute
            self.graph.add_node(node_id, **node_data)
        return node_id

    def generate_purchase_order(self):
        suppliers = [
            n for n, d in self.graph.nodes(data=True) if d.get("type") == "Supplier"
        ]
        if not suppliers:
            print(
                "Warning: No suppliers available. Skipping purchase order generation."
            )
            return None

        supplier = random.choice(suppliers)
        supplier_data = self.graph.nodes[supplier]
        if "products" not in supplier_data:
            print(
                f"Warning: Supplier {supplier} has no products. Skipping purchase order generation."
            )
            return None

        if not supplier_data["products"]:
            print(
                f"Warning: Supplier {supplier} has an empty products list. Skipping purchase order generation."
            )
            return None

        product = random.choice(supplier_data["products"])
        units = random.randint(1, 10)  # Random number of units for the product
        po_id = f"PO_{uuid.uuid4()}"
        self.add_node(
            po_id,
            "Purchase Order",
            {"supplier": supplier, "product": product, "units": units},
        )
        self.graph.add_edge(po_id, supplier)
        self.graph.add_edge(po_id, product)
        return po_id

    def process_purchase_order(self, po_id):
        po_data = self.graph.nodes[po_id]
        supplier = po_data["supplier"]
        product = po_data["product"]
        units = po_data["units"]

        if product not in self.data["Products"]:
            print(
                f"Warning: Product {product} not found in data. Skipping purchase order processing."
            )
            return

        for _ in range(units):
            self.create_product_instance(product, supplier)

    def create_product_instance(self, product_id, supplier):
        new_product_id = f"{product_id}_{uuid.uuid4()}"
        product_data = self.data["Products"][product_id].copy()
        product_data["original_id"] = product_id
        self.add_node(new_product_id, "Product Instance", product_data)
        self.graph.add_edge(new_product_id, supplier)

        for part_id, quantity in product_data["parts"].items():
            for _ in range(quantity):
                new_part_id = self.create_part_instance(part_id)
                self.graph.add_edge(new_product_id, new_part_id)

        # Find a warehouse to store the product
        warehouses = [
            w
            for w in self.data["Warehouses"]
            if product_id in self.data["Warehouses"][w]["products"]
        ]
        if warehouses:
            warehouse = random.choice(warehouses)
            self.graph.add_edge(new_product_id, warehouse)
        else:
            print(f"Warning: No warehouses available for product {new_product_id}")

    def create_part_instance(self, part_id):
        new_part_id = f"{part_id}_{uuid.uuid4()}"
        part_data = self.data["Parts"][part_id].copy()
        part_data["original_id"] = part_id
        self.add_node(new_part_id, "Part Instance", part_data)

        for subpart_id, quantity in part_data.get("subparts", {}).items():
            for _ in range(quantity):
                new_subpart_id = self.create_part_instance(subpart_id)
                self.graph.add_edge(new_part_id, new_subpart_id)

        return new_part_id

    def simulate_timestamp(self):
        self.timestamp += 1
        num_pos = random.randint(1, self.config["max_pos_per_timestamp"])

        for _ in range(num_pos):
            po_id = self.generate_purchase_order()
            if po_id:
                self.process_purchase_order(po_id)

        self.age_parts()
        self.save_graph_state()

    def age_parts(self):
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "Part Instance":
                part_id = data.get("original_id")
                if part_id and part_id in self.data["Parts"]:
                    expected_life = self.data["Parts"][part_id]["expected_life"]
                    age = data.get("age", 0) + 1
                    self.graph.nodes[node]["age"] = age
                    if age > expected_life:
                        self.graph.nodes[node]["status"] = "Failed"
                else:
                    print(
                        f"Warning: Part type {part_id} not found in data or missing original_id. Skipping aging for node {node}."
                    )
            elif "type" not in data:
                print(f"Warning: Node {node} has no 'type' attribute. Skipping aging.")

    def save_graph_state(self):
        timestamp_path = os.path.join(self.data_path, f"t_{self.timestamp}")
        timestamp_path = self.create_directory(timestamp_path)
        print(f"Saving graph state for timestamp {self.timestamp}")

        node_counts = Counter()
        edge_counts = Counter()

        root_nodes = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("type") in ["Business Unit", "Purchase Order"]
        ]
        for root in root_nodes:
            subgraph = self.graph.subgraph(nx.descendants(self.graph, root) | {root})
            self.save_subgraph(subgraph, timestamp_path, node_counts, edge_counts)

        print(f"Timestamp {self.timestamp} summary:")
        for node_type, count in node_counts.items():
            print(f"  Saved {count} {node_type} nodes")
        for node_type, count in edge_counts.items():
            print(f"  Saved {count} edges for {node_type} nodes")
        print(f"Finished saving graph state for timestamp {self.timestamp}")

        # Print warning for nodes without 'type' attribute
        nodes_without_type = [
            n for n, d in self.graph.nodes(data=True) if "type" not in d
        ]
        if nodes_without_type:
            print(
                f"Warning: The following nodes have no 'type' attribute: {nodes_without_type}"
            )

    def save_subgraph(self, subgraph, current_path, node_counts, edge_counts):
        original_dir = os.getcwd()
        try:
            current_path = os.path.abspath(current_path)
            os.chdir(current_path)
            for node in subgraph.nodes():
                node_data = subgraph.nodes[node]
                node_type = node_data.get("type", "Unknown")
                safe_name = self.safe_filename(node, node_type)

                try:
                    node_dir = self.create_directory(
                        os.path.join(current_path, safe_name)
                    )
                    os.chdir(node_dir)

                    # Save node data
                    with open("node_data.json", "w") as f:
                        json.dump({"id": node, "data": node_data}, f)
                    node_counts[node_type] += 1

                    # Save edge data
                    edges = list(subgraph.edges(node))
                    if edges:
                        edge_data = [
                            {"source": node, "target": target}
                            for target in subgraph.neighbors(node)
                        ]
                        with open("edge_data.json", "w") as f:
                            json.dump(edge_data, f)
                        edge_counts[node_type] += len(edges)

                    # Recursively save the subgraph of this node's neighbors
                    neighbor_subgraph = subgraph.subgraph(
                        list(subgraph.neighbors(node))
                    )
                    self.save_subgraph(
                        neighbor_subgraph, node_dir, node_counts, edge_counts
                    )

                except OSError as e:
                    if e.errno == errno.ENAMETOOLONG:
                        print(
                            f"Warning: Path too long for node {node}. Skipping this node and its subgraph."
                        )
                    else:
                        raise
                finally:
                    os.chdir(current_path)
        finally:
            os.chdir(original_dir)

    def create_directory(self, path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            if e.errno == errno.ENAMETOOLONG:
                # If the path is too long, create a shortened version
                parts = path.split(os.sep)
                shortened_parts = [
                    p[:20] for p in parts
                ]  # Truncate each part to 20 characters
                short_path = os.sep.join(shortened_parts)
                os.makedirs(short_path, exist_ok=True)
                return short_path
            else:
                raise
        return path

    def safe_filename(self, name, node_type):
        # Prefix the filename with the node type
        prefix = f"{node_type}_"
        # Replace any characters that are invalid in filenames
        safe_name = "".join(c for c in name if c.isalnum() or c in ["-", "_"]).rstrip()
        # Truncate the name if it's too long
        max_length = 255 - len(prefix) - 1  # Account for prefix and potential separator
        if len(safe_name) > max_length:
            safe_name = safe_name[:max_length]
        return prefix + safe_name

    def safe_save_file(self, directory, filename, content):
        full_path = os.path.join(directory, filename)
        with open(full_path, "w") as f:
            f.write(content)

    def run_simulation(self):
        for _ in range(self.config["num_timestamps"]):
            self.simulate_timestamp()

    def save_node_data(self, node_id, node_data, path):
        filename = hashlib.md5(node_id.encode()).hexdigest() + ".json"
        with open(os.path.join(path, filename), "w") as f:
            json.dump({"id": node_id, "data": node_data}, f)

    def save_edge_data(self, edges, path):
        filename = "edges.json"
        with open(os.path.join(path, filename), "a") as f:
            for edge in edges:
                json.dump({"source": edge[0], "target": edge[1]}, f)
                f.write("\n")


# Increase the recursion limit
sys.setrecursionlimit(10000)

# Example usage
config = {
    "num_timestamps": 100,
    "max_pos_per_timestamp": 5,
    "num_business_units": 10,
    "num_products": 50,
    "num_suppliers": 20,
    "num_warehouses": 30,
    "num_part_types": 50,
    "data_path": "data/run1",
}

simulation = SupplyChainSimulation(config)
simulation.run_simulation()
