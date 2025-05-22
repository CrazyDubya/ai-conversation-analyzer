import csv
import time
from collections import defaultdict
from typing import Dict

import community
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from pyvis.network import Network


def load_function_counts(file_path: str) -> Dict[str, Dict[str, int]]:
    """Load function counts from CSV file."""
    function_counts = defaultdict(lambda: defaultdict(int))
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for function, conversation, count in reader:
            function_counts[conversation][function] = int(count)
    return function_counts


def build_function_graph(function_counts: Dict[str, Dict[str, int]]) -> nx.Graph:
    """Build a graph of function co-occurrences."""
    G = nx.Graph()
    for functions in function_counts.values():
        for f1 in functions:
            for f2 in functions:
                if f1 != f2:
                    if G.has_edge(f1, f2):
                        G[f1][f2]['weight'] += 1
                    else:
                        G.add_edge(f1, f2, weight=1)
    return G


def analyze_function_communities(G: nx.Graph) -> Dict[str, int]:
    """Detect communities in the function graph."""
    return community.best_partition(G)


def create_heatmap(G: nx.Graph, output_file: str):
    """Create a heatmap of function co-occurrences."""
    adj_matrix = nx.to_pandas_adjacency(G)
    plt.figure(figsize=(20, 16))
    sns.heatmap(adj_matrix, cmap="YlGnBu", annot=False)
    plt.title("Function Co-occurrence Heatmap")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def create_interactive_graph(G: nx.Graph, communities: Dict[str, int], output_file: str):
    """Create an interactive graph visualization."""
    net = Network(notebook=True, height="750px", width="100%")
    net.from_nx(G)
    for node in net.nodes:
        node["group"] = communities[node["id"]]
        node["size"] = G.degree(node["id"]) * 3
    net.show_buttons(filter_=['physics'])
    net.save_graph(output_file)


def analyze_function_usage(function_counts: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """Analyze function usage across conversations."""
    function_usage = defaultdict(int)
    for functions in function_counts.values():
        for function in functions:
            function_usage[function] += 1

    df = pd.DataFrame.from_dict(function_usage, orient='index', columns=['Count'])
    df = df.sort_values('Count', ascending=False)
    return df


def print_top_functions(df: pd.DataFrame, n: int = 10):
    """Print the top N most used functions."""
    print(f"\nTop {n} Most Used Functions:")
    print(df.head(n).to_string())


def main():
    input_file = 'function_counts.csv'
    heatmap_file = 'function_co_occurrence_heatmap.png'
    interactive_graph_file = 'function_graph.html'

    start_time = time.time()
    function_counts = load_function_counts(input_file)
    print(f"Loaded data from {len(function_counts)} conversations")

    G = build_function_graph(function_counts)
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    communities = analyze_function_communities(G)
    print(f"Detected {len(set(communities.values()))} communities")

    create_heatmap(G, heatmap_file)
    print(f"Created heatmap: {heatmap_file}")

    create_interactive_graph(G, communities, interactive_graph_file)
    print(f"Created interactive graph: {interactive_graph_file}")

    usage_df = analyze_function_usage(function_counts)
    print_top_functions(usage_df)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
