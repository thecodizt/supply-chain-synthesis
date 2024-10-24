import networkx as nx
import random
import uuid
import json
import os
from collections import Counter
import errno
import hashlib
import sys
import shutil
from datetime import datetime
import time
from collections import defaultdict
import matplotlib.pyplot as plt


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

        self.counters = {
            "Business Unit": 0,
            "Product": 0,
            "Part": 0,
            "Supplier": 0,
            "Warehouse": 0,
            "Purchase Order": 0,
            "Product Instance": 0,
            "Part Instance": 0,
        }

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
        for _ in range(self.config["num_business_units"]):
            bu_id = self.generate_id("Business Unit")
            data["Business Units"][bu_id] = {
                "name": f"Business Unit {self.counters['Business Unit']}",
                "products": [],
                "timestamp": self.timestamp,
            }

        # Create Parts with hierarchy based on schema
        self.create_parts_hierarchy(data)

        # Create Products
        for _ in range(self.config["num_products"]):
            product_id = self.generate_id("Product")
            business_unit = random.choice(list(data["Business Units"].keys()))
            data["Products"][product_id] = {
                "name": f"Product {self.counters['Product']}",
                "expected_life": random.randint(5000, 20000),
                "parts": self.assign_parts_to_product(data["Parts"]),
                "business_unit": business_unit,
                "timestamp": self.timestamp,
            }
            data["Business Units"][business_unit]["products"].append(product_id)

        # Create Suppliers
        for _ in range(self.config["num_suppliers"]):
            supplier_id = self.generate_id("Supplier")
            num_products = max(1, int(len(data["Products"]) * random.uniform(0.1, 0.3)))
            data["Suppliers"][supplier_id] = {
                "name": f"Supplier {self.counters['Supplier']}",
                "location": f"Location {self.counters['Supplier']}",
                "products": random.sample(list(data["Products"].keys()), num_products),
                "timestamp": self.timestamp,
            }

        # Create Warehouses
        for _ in range(self.config["num_warehouses"]):
            warehouse_id = self.generate_id("Warehouse")
            num_products = max(1, int(len(data["Products"]) * random.uniform(0.3, 0.6)))
            data["Warehouses"][warehouse_id] = {
                "name": f"Warehouse {self.counters['Warehouse']}",
                "location": f"Location {self.counters['Warehouse']}",
                "products": random.sample(list(data["Products"].keys()), num_products),
                "timestamp": self.timestamp,
            }

        return data

    def create_parts_hierarchy(self, data):
        max_depth = self.schema["Part"]["max_depth"]
        max_subparts = self.schema["Part"]["max_subparts"]

        def create_part(depth=0):
            part_id = self.generate_id("Part")
            data["Parts"][part_id] = {
                "name": f"Part Type {self.counters['Part']}",
                "expected_life": random.randint(1000, 10000),
                "subparts": {},
                "timestamp": self.timestamp,
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
        num_parts = random.randint(1, min(5, len(parts)))
        selected_parts = random.sample(list(parts.keys()), num_parts)
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
        now = datetime.now().isoformat()
        node_data["type"] = node_type
        node_data["created_at"] = now
        node_data["updated_at"] = now
        self.graph.add_node(node_id, **node_data)

    def update_node(self, node_id, updates):
        if node_id in self.graph:
            self.graph.nodes[node_id].update(updates)
            self.graph.nodes[node_id]["updated_at"] = datetime.now().isoformat()

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
        po_id = self.generate_id("Purchase Order")
        self.add_node(
            po_id,
            "Purchase Order",
            {
                "supplier": supplier,
                "product": product,
                "units": units,
                "timestamp": self.timestamp,
            },
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
        new_product_id = self.generate_id("Product Instance")
        product_data = self.data["Products"][product_id].copy()
        product_data["original_id"] = product_id
        product_data["timestamp"] = self.timestamp
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
        new_part_id = self.generate_id("Part Instance")
        part_data = self.data["Parts"][part_id].copy()
        part_data["original_id"] = part_id
        part_data["timestamp"] = self.timestamp
        self.add_node(new_part_id, "Part Instance", part_data)

        for subpart_id, quantity in part_data.get("subparts", {}).items():
            for _ in range(quantity):
                new_subpart_id = self.create_part_instance(subpart_id)
                self.graph.add_edge(new_part_id, new_subpart_id)

        return new_part_id

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
        timestamp_file = os.path.join(self.data_path, f"t_{self.timestamp}.json")
        graph_data = {
            "nodes": dict(self.graph.nodes(data=True)),
            "edges": list(self.graph.edges()),
        }
        with open(timestamp_file, "w") as f:
            json.dump(graph_data, f)

    def load_graph_state(self, timestamp):
        timestamp_file = os.path.join(self.data_path, f"t_{timestamp}.json")
        if os.path.exists(timestamp_file):
            with open(timestamp_file, "r") as f:
                graph_data = json.load(f)

            G = nx.Graph()
            G.add_nodes_from((n, d) for n, d in graph_data["nodes"].items())
            G.add_edges_from(graph_data["edges"])
            return G
        else:
            raise FileNotFoundError(f"No data found for timestamp {timestamp}")

    def simulate_addition(self):
        start_time = time.time()
        self.timestamp += 1
        num_pos = random.randint(1, self.config["max_pos_per_timestamp"])

        for _ in range(num_pos):
            po_id = self.generate_purchase_order()
            if po_id:
                self.process_purchase_order(po_id)

        self.age_parts()
        self.save_graph_state()
        end_time = time.time()
        self.log_timestamp_info(start_time, end_time)

    def simulate_updation(self):
        start_time = time.time()
        self.timestamp += 1
        num_updates = random.randint(1, self.config["max_updates_per_timestamp"])

        for _ in range(num_updates):
            node = random.choice(list(self.graph.nodes()))
            updates = self.generate_random_updates(self.graph.nodes[node])
            self.update_node(node, updates)

        self.age_parts()
        self.save_graph_state()
        end_time = time.time()
        self.log_timestamp_info(start_time, end_time)

    def simulate_deletion(self):
        start_time = time.time()
        self.timestamp += 1
        num_deletions = random.randint(1, self.config["max_deletions_per_timestamp"])

        for _ in range(num_deletions):
            if self.graph.number_of_nodes() > 0:
                node = random.choice(list(self.graph.nodes()))
                self.graph.remove_node(node)

        self.age_parts()
        self.save_graph_state()
        end_time = time.time()
        self.log_timestamp_info(start_time, end_time)

    def simulate_schema_update(self):
        start_time = time.time()
        self.timestamp += 1
        node_types = set(nx.get_node_attributes(self.graph, "type").values())

        for node_type in node_types:
            if random.random() < 0.3:  # 30% chance to update schema for each node type
                if random.random() < 0.5:  # 50% chance to add a property
                    new_property = f"new_property_{self.timestamp}"
                    default_value = random.choice([0, 1, True, False, "New Value"])
                    for node, data in self.graph.nodes(data=True):
                        if data.get("type") == node_type:
                            self.graph.nodes[node][new_property] = default_value
                else:  # 50% chance to remove a property
                    non_essential_props = [
                        prop
                        for prop in self.graph.nodes[list(self.graph.nodes())[0]].keys()
                        if prop not in ["type", "id", "created_at", "updated_at"]
                    ]
                    if non_essential_props:
                        prop_to_remove = random.choice(non_essential_props)
                        for node, data in self.graph.nodes(data=True):
                            if data.get("type") == node_type and prop_to_remove in data:
                                del self.graph.nodes[node][prop_to_remove]

        self.save_graph_state()
        end_time = time.time()
        self.log_timestamp_info(start_time, end_time)

    def log_timestamp_info(self, start_time, end_time):
        timestamp_file = os.path.join(self.data_path, f"t_{self.timestamp}.json")
        file_size = os.path.getsize(timestamp_file) / 1024  # Convert to KB
        generation_time = end_time - start_time

        node_type_stats = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            node_type_stats[data.get("type", "Unknown")] += 1

        print(f"Timestamp {self.timestamp}:")
        print(f"  File size: {file_size:.2f} KB")
        print(f"  Generation time: {generation_time:.2f} seconds")
        print("  Node type stats:")
        for node_type, count in node_type_stats.items():
            print(f"    {node_type}: {count}")
        print()

    def generate_random_updates(self, node_data):
        updates = {}
        if "age" in node_data:
            updates["age"] = node_data["age"] + random.randint(1, 10)
        if (
            "status" in node_data and random.random() < 0.1
        ):  # 10% chance to change status
            updates["status"] = random.choice(["Active", "Inactive", "Maintenance"])
        return updates

    def run_simulation(self):
        # Initial synthesis
        for _ in range(self.config["initial_timestamps"]):
            self.simulate_addition()

        # Updates, deletions, and schema updates
        for _ in range(self.config["update_delete_timestamps"]):
            action = random.choices(
                ["update", "delete", "schema_update"], weights=[0.6, 0.3, 0.1]
            )[0]
            if action == "update":
                self.simulate_updation()
            elif action == "delete":
                self.simulate_deletion()
            else:
                self.simulate_schema_update()

        print("Simulation completed.")

    def save_node_data(self, node_id, node_data, path):
        filename = hashlib.md5(node_id.encode()).hexdigest() + ".json"
        with open(os.path.join(path, filename), "w") as f:
            json.dump(
                {"id": node_id, "data": node_data}, f, ensure_ascii=False, indent=2
            )

    def save_edge_data(self, edges, path):
        filename = "edges.json"
        with open(os.path.join(path, filename), "a") as f:
            for edge in edges:
                json.dump({"source": edge[0], "target": edge[1]}, f)
                f.write("\n")

    def generate_id(self, node_type):
        if node_type not in self.counters:
            self.counters[node_type] = 0
        self.counters[node_type] += 1
        return f"{node_type.replace(' ', '_')}_{self.counters[node_type]:06d}"

    def plot_node_edge_evolution(self):
        plt.figure(figsize=(12, 6))

        # Plot node evolution
        ax1 = plt.subplot(121)
        for node_type in self.node_counts[0].keys():
            counts = [
                timestamp_count[node_type] for timestamp_count in self.node_counts
            ]
            ax1.plot(range(len(self.node_counts)), counts, label=node_type)

        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Number of Nodes")
        ax1.set_title("Node Evolution")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Plot edge evolution
        ax2 = plt.subplot(122)
        for edge_type in self.edge_counts[0].keys():
            counts = [
                timestamp_count[edge_type] for timestamp_count in self.edge_counts
            ]
            ax2.plot(range(len(self.edge_counts)), counts, label=edge_type)

        ax2.set_xlabel("Timestamp")
        ax2.set_ylabel("Number of Edges")
        ax2.set_title("Edge Evolution")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config["data_path"], "node_edge_evolution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


# Increase the recursion limit
sys.setrecursionlimit(10000)

# Example usage
config = {
    "initial_timestamps": 10,
    "update_delete_timestamps": 2,
    "max_pos_per_timestamp": 5,
    "max_updates_per_timestamp": 10,
    "max_deletions_per_timestamp": 3,
    "num_business_units": 10,
    "num_products": 50,
    "num_suppliers": 20,
    "num_warehouses": 30,
    "num_part_types": 100,
    "data_path": "data/run1",
}

simulation = SupplyChainSimulation(config)
simulation.run_simulation()
