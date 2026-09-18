#!/usr/bin/env python3
"""
NetworkX benchmark template for scirs2-graph comparison

This script provides standardized benchmarking for NetworkX algorithms
to enable fair comparison with scirs2-graph implementations.
"""

import networkx as nx
import time
import psutil
import os
import sys
import json
from typing import Dict, Any, Optional

class NetworkXBenchmark:
    """Standardized NetworkX benchmark runner"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def create_graph(self, graph_type: str, size: int, **kwargs) -> nx.Graph:
        """Create graph of specified type and size"""
        if graph_type == "erdos_renyi":
            p = kwargs.get("p", 0.01)
            return nx.erdos_renyi_graph(size, p)
        elif graph_type == "barabasi_albert":
            m = kwargs.get("m", 3)
            return nx.barabasi_albert_graph(size, m)
        elif graph_type == "watts_strogatz":
            k = kwargs.get("k", 6)
            p = kwargs.get("p", 0.3)
            return nx.watts_strogatz_graph(size, k, p)
        elif graph_type == "complete":
            return nx.complete_graph(size)
        elif graph_type == "path":
            return nx.path_graph(size)
        elif graph_type == "cycle":
            return nx.cycle_graph(size)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
    
    def run_algorithm(self, algorithm: str, graph: nx.Graph, **kwargs) -> Any:
        """Run specified algorithm on graph"""
        if algorithm == "bfs":
            source = kwargs.get("source", 0)
            return list(nx.bfs_tree(graph, source))
        elif algorithm == "dfs":
            source = kwargs.get("source", 0)
            return list(nx.dfs_tree(graph, source))
        elif algorithm == "shortest_path":
            source = kwargs.get("source", 0)
            target = kwargs.get("target", min(10, len(graph) - 1))
            if target in graph:
                return nx.shortest_path(graph, source, target)
            return None
        elif algorithm == "dijkstra":
            source = kwargs.get("source", 0)
            target = kwargs.get("target", min(10, len(graph) - 1))
            if target in graph:
                return nx.dijkstra_path(graph, source, target)
            return None
        elif algorithm == "betweenness_centrality":
            return nx.betweenness_centrality(graph)
        elif algorithm == "closeness_centrality":
            return nx.closeness_centrality(graph)
        elif algorithm == "eigenvector_centrality":
            try:
                return nx.eigenvector_centrality(graph, max_iter=1000)
            except:
                return None
        elif algorithm == "pagerank":
            return nx.pagerank(graph)
        elif algorithm == "connected_components":
            return list(nx.connected_components(graph))
        elif algorithm == "strongly_connected_components":
            if isinstance(graph, nx.DiGraph):
                return list(nx.strongly_connected_components(graph))
            else:
                # Convert to directed graph for testing
                digraph = graph.to_directed()
                return list(nx.strongly_connected_components(digraph))
        elif algorithm == "minimum_spanning_tree":
            # Add weights if not present
            if not any('weight' in graph[u][v] for u, v in graph.edges()):
                for u, v in graph.edges():
                    graph[u][v]['weight'] = 1.0
            return nx.minimum_spanning_tree(graph)
        elif algorithm == "louvain_communities":
            try:
                import networkx.algorithms.community as community
                return community.greedy_modularity_communities(graph)
            except ImportError:
                # Fallback to simpler community detection
                return list(nx.connected_components(graph))
        elif algorithm == "clustering_coefficient":
            return nx.clustering(graph)
        elif algorithm == "triangles":
            return nx.triangles(graph)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def benchmark(self, graph_type: str, size: int, algorithm: str, **kwargs) -> Dict[str, Any]:
        """Run complete benchmark and return results"""
        results = {
            "graph_type": graph_type,
            "graph_size": size,
            "algorithm": algorithm,
            "success": False,
            "error": None,
            "execution_time_ms": 0.0,
            "memory_usage_mb": 0.0,
            "graph_nodes": 0,
            "graph_edges": 0,
        }
        
        try:
            # Create graph
            graph_start = time.time()
            graph = self.create_graph(graph_type, size, **kwargs)
            graph_creation_time = time.time() - graph_start
            
            results["graph_nodes"] = len(graph)
            results["graph_edges"] = len(graph.edges())
            results["graph_creation_time_ms"] = graph_creation_time * 1000
            
            # Measure memory before algorithm
            mem_before = self.get_memory_usage()
            
            # Run algorithm
            start_time = time.time()
            result = self.run_algorithm(algorithm, graph, **kwargs)
            end_time = time.time()
            
            # Measure memory after algorithm
            mem_after = self.get_memory_usage()
            
            results["execution_time_ms"] = (end_time - start_time) * 1000
            results["memory_usage_mb"] = mem_after - mem_before
            results["total_memory_mb"] = mem_after - self.initial_memory
            results["success"] = True
            
            # Add algorithm-specific metrics
            if algorithm in ["betweenness_centrality", "closeness_centrality", "eigenvector_centrality", "pagerank"]:
                if result and isinstance(result, dict):
                    results["result_size"] = len(result)
                    if result:
                        results["avg_centrality"] = sum(result.values()) / len(result)
            elif algorithm in ["connected_components", "strongly_connected_components", "louvain_communities"]:
                if result:
                    results["num_components"] = len(list(result))
            
        except Exception as e:
            results["error"] = str(e)
            results["success"] = False
        
        return results

def main():
    """Main benchmark runner"""
    if len(sys.argv) < 4:
        print("Usage: networkx_benchmark.py <graph_type> <size> <algorithm> [options]")
        print("Graph types: erdos_renyi, barabasi_albert, watts_strogatz, complete, path, cycle")
        print("Algorithms: bfs, dfs, shortest_path, dijkstra, betweenness_centrality, pagerank, etc.")
        sys.exit(1)
    
    graph_type = sys.argv[1]
    size = int(sys.argv[2])
    algorithm = sys.argv[3]
    
    # Parse additional options
    kwargs = {}
    for i in range(4, len(sys.argv), 2):
        if i + 1 < len(sys.argv):
            key = sys.argv[i].lstrip('-')
            value = sys.argv[i + 1]
            try:
                # Try to parse as number
                if '.' in value:
                    kwargs[key] = float(value)
                else:
                    kwargs[key] = int(value)
            except ValueError:
                kwargs[key] = value
    
    benchmark = NetworkXBenchmark()
    results = benchmark.benchmark(graph_type, size, algorithm, **kwargs)
    
    # Output results in both human-readable and machine-readable formats
    print(f"NetworkX Benchmark Results:")
    print(f"Graph: {graph_type} ({size} nodes)")
    print(f"Algorithm: {algorithm}")
    print(f"Success: {results['success']}")
    
    if results['success']:
        print(f"Execution Time: {results['execution_time_ms']:.3f} ms")
        print(f"Memory Usage: {results['memory_usage_mb']:.3f} MB")
        print(f"Graph Creation: {results.get('graph_creation_time_ms', 0):.3f} ms")
        print(f"Actual Nodes: {results['graph_nodes']}")
        print(f"Actual Edges: {results['graph_edges']}")
    else:
        print(f"Error: {results['error']}")
    
    # Machine-readable output
    print("\n--- JSON RESULTS ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()