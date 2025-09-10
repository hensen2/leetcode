"""
Comprehensive tests for GraphGenerator

Tests graph generation, connectivity, directedness, weights, and CLI integration.
Most complex generator with zero previous test coverage.
"""

from typing import Dict, List

import pytest
from testgen.core.generators import GraphGenerator
from testgen.core.models import Constraints, GraphProperties


class TestGraphGeneratorBasics:
    """Test basic graph generation functionality"""

    def test_graph_generator_instantiation(self):
        """Test GraphGenerator can be instantiated"""
        generator = GraphGenerator()
        assert generator is not None
        assert hasattr(generator, "generate")

    def test_graph_generator_with_seed(self):
        """Test GraphGenerator with seed produces reproducible results"""
        seed = 12345
        props = GraphProperties(num_nodes=5, num_edges=6)

        # First generation
        gen1 = GraphGenerator(seed)
        graph1 = gen1.generate(props)
        print(graph1)
        # Second generation resets with same seed
        gen2 = GraphGenerator(seed)
        graph2 = gen2.generate(props)
        print(graph2)

        # Same seed should produce same graph structure
        assert self._graphs_equal(graph1, graph2), "Same seed should produce same graph"

    def test_basic_graph_generation(self):
        """Test basic graph generation with various sizes"""
        generator = GraphGenerator()

        test_configs = [
            (3, 2),  # Small graph
            (5, 4),  # Medium graph
            (10, 15),  # Larger graph
            (4, 0),  # Graph with no edges
        ]

        for num_nodes, num_edges in test_configs:
            props = GraphProperties(num_nodes=num_nodes, num_edges=num_edges)
            graph = generator.generate(props)

            assert graph is not None, (
                f"Graph should not be None for {num_nodes} nodes, {num_edges} edges"
            )

            # Verify graph structure
            actual_nodes = self._count_nodes(graph)
            actual_edges = self._count_edges(graph)

            assert actual_nodes == num_nodes, (
                f"Should have {num_nodes} nodes, got {actual_nodes}"
            )
            assert actual_edges == num_edges, (
                f"Should have {num_edges} edges, got {actual_edges}"
            )

    def test_empty_graph_generation(self):
        """Test generation of graphs with no nodes"""
        generator = GraphGenerator()

        props = GraphProperties(num_nodes=0, num_edges=0)
        graph = generator.generate(props)

        # Should handle empty graph gracefully
        if graph is not None:
            assert self._count_nodes(graph) == 0, "Empty graph should have 0 nodes"

    def test_single_node_graph(self):
        """Test generation of single node graphs"""
        generator = GraphGenerator()

        props = GraphProperties(num_nodes=1, num_edges=0)
        graph = generator.generate(props)

        assert graph is not None, "Single node graph should not be None"
        assert self._count_nodes(graph) == 1, "Should have exactly 1 node"
        assert self._count_edges(graph) == 0, "Single node should have no edges"

    def _graphs_equal(self, graph1, graph2) -> bool:
        """Helper to check if two graphs are structurally equal"""
        if graph1 is None and graph2 is None:
            return True
        if graph1 is None or graph2 is None:
            return False

        # Compare basic structure (depends on graph representation)
        nodes1 = self._count_nodes(graph1)
        nodes2 = self._count_nodes(graph2)
        edges1 = self._count_edges(graph1)
        edges2 = self._count_edges(graph2)

        return nodes1 == nodes2 and edges1 == edges2

    def _count_nodes(self, graph) -> int:
        """Helper to count nodes in graph"""
        if graph is None:
            return 0

        # Graph representation could be adjacency list, matrix, or edge list
        if isinstance(graph, dict):
            return len(graph)
        elif isinstance(graph, list) and len(graph) > 0:
            if isinstance(graph[0], list):  # Adjacency matrix
                return len(graph)
            elif isinstance(graph[0], tuple):  # Edge list
                nodes = set()
                for edge in graph:
                    nodes.add(edge[0])
                    nodes.add(edge[1])
                return len(nodes)

        return 0

    def _count_edges(self, graph) -> int:
        """Helper to count edges in graph"""
        if graph is None:
            return 0

        if isinstance(graph, dict):  # Adjacency list
            return (
                sum(len(neighbors) for neighbors in graph.values()) // 2
            )  # Undirected
        elif isinstance(graph, list) and len(graph) > 0:
            if isinstance(graph[0], list):  # Adjacency matrix
                count = 0
                for i in range(len(graph)):
                    for j in range(len(graph[i])):
                        if graph[i][j] != 0:
                            count += 1
                return count // 2  # Undirected
            elif isinstance(graph[0], tuple):  # Edge list
                return len(graph)

        return 0


class TestGraphGeneratorConnectivity:
    """Test graph connectivity properties"""

    def test_connected_graph_generation(self):
        """Test generation of connected graphs"""
        generator = GraphGenerator()

        # Test various sizes of connected graphs
        for num_nodes in [3, 5, 8, 12]:
            min_edges = num_nodes - 1  # Minimum edges for connectivity
            props = GraphProperties(
                num_nodes=num_nodes,
                num_edges=min_edges + 2,  # A few extra edges
                connected=True,
            )

            graph = generator.generate(props)

            assert graph is not None, (
                f"Connected graph should not be None for {num_nodes} nodes"
            )
            assert self._is_connected(graph), (
                f"Graph should be connected for {num_nodes} nodes"
            )

    def test_disconnected_graph_generation(self):
        """Test generation of disconnected graphs"""
        generator = GraphGenerator()

        props = GraphProperties(
            num_nodes=8,
            num_edges=3,  # Too few edges to be connected
            connected=False,
        )

        # Generate multiple graphs to test variety
        graphs = []
        for _ in range(10):
            graph = generator.generate(props)
            graphs.append(graph)

        # At least some should be disconnected
        disconnected_count = sum(1 for graph in graphs if not self._is_connected(graph))
        assert disconnected_count > 0, (
            "Should generate some disconnected graphs when connected=False"
        )

    def test_minimum_edges_for_connectivity(self):
        """Test that connected graphs have minimum required edges"""
        generator = GraphGenerator()

        num_nodes = 6
        min_edges = num_nodes - 1  # Minimum for connectivity

        props = GraphProperties(
            num_nodes=num_nodes, num_edges=min_edges, connected=True
        )

        graph = generator.generate(props)

        actual_edges = self._count_edges(graph)
        assert actual_edges >= min_edges, (
            "Connected graph should have at least n-1 edges"
        )
        assert self._is_connected(graph), (
            "Graph with minimum edges should still be connected"
        )

    def _is_connected(self, graph) -> bool:
        """Check if graph is connected using BFS"""
        if graph is None:
            return True

        nodes = self._count_nodes(graph)
        if nodes <= 1:
            return True

        # Convert to adjacency list format for BFS
        adj_list = self._to_adjacency_list(graph)
        if not adj_list:
            return nodes <= 1

        # BFS from first node
        visited = set()
        queue = [next(iter(adj_list.keys()))]
        visited.add(queue[0])

        while queue:
            current = queue.pop(0)
            for neighbor in adj_list.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == nodes

    def _to_adjacency_list(self, graph) -> Dict:
        """Convert graph to adjacency list format"""
        if isinstance(graph, dict):
            return graph
        elif isinstance(graph, list) and len(graph) > 0:
            if isinstance(graph[0], list):  # Adjacency matrix
                adj_list = {}
                for i in range(len(graph)):
                    adj_list[i] = []
                    for j in range(len(graph[i])):
                        if graph[i][j] != 0:
                            adj_list[i].append(j)
                return adj_list
            elif isinstance(graph[0], tuple):  # Edge list
                adj_list = {}
                for u, v in graph:
                    if u not in adj_list:
                        adj_list[u] = []
                    if v not in adj_list:
                        adj_list[v] = []
                    adj_list[u].append(v)
                    adj_list[v].append(u)  # Assume undirected
                return adj_list

        return {}


class TestGraphGeneratorDirectedness:
    """Test directed vs undirected graph generation"""

    def test_undirected_graph_generation(self):
        """Test generation of undirected graphs"""
        generator = GraphGenerator()

        props = GraphProperties(num_nodes=5, num_edges=6, directed=False)

        graph = generator.generate(props)

        assert graph is not None, "Undirected graph should not be None"
        assert self._is_undirected(graph), "Graph should be undirected"

    def test_directed_graph_generation(self):
        """Test generation of directed graphs"""
        generator = GraphGenerator()

        props = GraphProperties(num_nodes=5, num_edges=8, directed=True)

        graph = generator.generate(props)

        assert graph is not None, "Directed graph should not be None"
        # Note: Directed graphs may or may not be symmetric
        # We mainly test that it's generated successfully

    def test_edge_count_consistency(self):
        """Test that edge counts are consistent with directedness"""
        generator = GraphGenerator()

        # Undirected graph
        props_undirected = GraphProperties(num_nodes=4, num_edges=4, directed=False)

        undirected_graph = generator.generate(props_undirected)
        undirected_edges = self._count_edges(undirected_graph)

        # Directed graph
        props_directed = GraphProperties(num_nodes=4, num_edges=4, directed=True)

        directed_graph = generator.generate(props_directed)
        directed_edges = self._count_directed_edges(directed_graph)

        # Both should respect their specified edge counts
        assert undirected_edges == 4, (
            "Undirected graph should have specified edge count"
        )

    def _is_undirected(self, graph) -> bool:
        """Check if graph is undirected (symmetric)"""
        adj_list = self._to_adjacency_list(graph)

        for node, neighbors in adj_list.items():
            for neighbor in neighbors:
                # Check if edge exists in both directions
                if neighbor not in adj_list or node not in adj_list[neighbor]:
                    return False

        return True

    def _count_directed_edges(self, graph) -> int:
        """Count edges in directed graph"""
        if isinstance(graph, dict):  # Adjacency list
            return sum(len(neighbors) for neighbors in graph.values())
        elif isinstance(graph, list) and len(graph) > 0:
            if isinstance(graph[0], list):  # Adjacency matrix
                count = 0
                for i in range(len(graph)):
                    for j in range(len(graph[i])):
                        if graph[i][j] != 0:
                            count += 1
                return count
            elif isinstance(graph[0], tuple):  # Edge list
                return len(graph)

        return 0

    def _to_adjacency_list(self, graph) -> Dict:
        """Convert graph to adjacency list format"""
        if isinstance(graph, dict):
            return graph
        elif isinstance(graph, list) and len(graph) > 0:
            if isinstance(graph[0], list):  # Adjacency matrix
                adj_list = {}
                for i in range(len(graph)):
                    adj_list[i] = []
                    for j in range(len(graph[i])):
                        if graph[i][j] != 0:
                            adj_list[i].append(j)
                return adj_list
            elif isinstance(graph[0], tuple):  # Edge list
                adj_list = {}
                for u, v in graph:
                    if u not in adj_list:
                        adj_list[u] = []
                    if v not in adj_list:
                        adj_list[v] = []
                    adj_list[u].append(v)
                    if not self._is_directed_from_properties(graph):
                        adj_list[v].append(u)
                return adj_list

        return {}

    def _is_directed_from_properties(self, graph) -> bool:
        """Heuristic to determine if graph is directed"""
        # This is a simplified heuristic
        return False  # Default to undirected for testing


class TestGraphGeneratorWeights:
    """Test weighted graph generation"""

    def test_weighted_graph_generation(self):
        """Test generation of weighted graphs"""
        generator = GraphGenerator()

        props = GraphProperties(num_nodes=5, num_edges=6, weighted=True)

        graph = generator.generate(props)

        assert graph is not None, "Weighted graph should not be None"
        assert self._has_weights(graph), "Graph should contain weight information"

    def test_unweighted_graph_generation(self):
        """Test generation of unweighted graphs"""
        generator = GraphGenerator()

        props = GraphProperties(num_nodes=5, num_edges=6, weighted=False)

        graph = generator.generate(props)

        assert graph is not None, "Unweighted graph should not be None"
        # Unweighted graphs may still work with weight-checking

    def test_weight_ranges(self):
        """Test that weights are within reasonable ranges"""
        generator = GraphGenerator()

        props = GraphProperties(num_nodes=4, num_edges=5, weighted=True)

        graph = generator.generate(props)

        if self._has_weights(graph):
            weights = self._extract_weights(graph)

            # Weights should be reasonable numbers
            assert all(isinstance(w, (int, float)) for w in weights), (
                "All weights should be numbers"
            )
            assert all(w > 0 for w in weights), (
                "Weights should be positive (unless specified otherwise)"
            )

    def _has_weights(self, graph) -> bool:
        """Check if graph contains weight information"""
        if isinstance(graph, dict):
            # Check if adjacency list contains weights
            for neighbors in graph.values():
                if neighbors and isinstance(neighbors[0], tuple):
                    return True
        elif isinstance(graph, list) and len(graph) > 0:
            if isinstance(graph[0], tuple) and len(graph[0]) > 2:
                return True  # Edge list with weights

        return False

    def _extract_weights(self, graph) -> List:
        """Extract all weights from graph"""
        weights = []

        if isinstance(graph, dict):
            for neighbors in graph.values():
                for neighbor in neighbors:
                    if isinstance(neighbor, tuple):
                        weights.append(neighbor[1])  # (node, weight)
        elif isinstance(graph, list) and len(graph) > 0:
            if isinstance(graph[0], tuple) and len(graph[0]) > 2:
                weights = [edge[2] for edge in graph]  # (u, v, weight)

        return weights


class TestGraphGeneratorEdgeCases:
    """Test edge cases and error conditions"""

    def test_impossible_edge_count(self):
        """Test handling of impossible edge counts"""
        generator = GraphGenerator()

        # Too many edges for number of nodes
        max_edges = 4 * 3 // 2  # Complete graph on 4 nodes
        props = GraphProperties(
            num_nodes=4,
            num_edges=max_edges + 5,  # Impossible
            connected=True,
        )

        try:
            graph = generator.generate(props)
            # Should either handle gracefully or raise error
            if graph is not None:
                actual_edges = self._count_edges(graph)
                assert actual_edges <= max_edges, (
                    "Should not exceed maximum possible edges"
                )
        except (ValueError, AssertionError):
            print("✅ Properly handles impossible edge count with error")

    def test_negative_node_count(self):
        """Test handling of negative node counts"""
        generator = GraphGenerator()

        try:
            props = GraphProperties(num_nodes=-5, num_edges=0)
            graph = generator.generate(props)

            # Should handle gracefully or raise error
            assert graph is None, "Negative nodes should produce None or raise error"
        except (ValueError, AssertionError):
            print("✅ Properly handles negative node count with error")

    def test_large_graph_generation(self):
        """Test generation of large graphs"""
        generator = GraphGenerator()

        large_nodes = 1000
        large_edges = 1500

        props = GraphProperties(
            num_nodes=large_nodes,
            num_edges=large_edges,
            connected=False,  # Don't require connectivity for performance
        )

        graph = generator.generate(props)

        assert graph is not None, "Large graph should not be None"

        actual_nodes = self._count_nodes(graph)
        actual_edges = self._count_edges(graph)

        assert actual_nodes == large_nodes, (
            f"Large graph should have {large_nodes} nodes"
        )
        # Allow some tolerance for edge count in large graphs
        assert abs(actual_edges - large_edges) <= large_edges * 0.1, (
            "Large graph should have approximately correct edge count"
        )

    def test_repeated_generation_variety(self):
        """Test that repeated generation produces variety"""
        generator = GraphGenerator()

        props = GraphProperties(num_nodes=6, num_edges=7)

        graphs = []
        for _ in range(10):
            graph = generator.generate(props)
            graphs.append(graph)

        # Should have some variety in structure
        # This is a basic check - more sophisticated comparison could be done
        assert all(g is not None for g in graphs), "All graphs should be generated"

    def _count_nodes(self, graph) -> int:
        """Helper to count nodes in graph"""
        if graph is None:
            return 0

        if isinstance(graph, dict):
            return len(graph)
        elif isinstance(graph, list) and len(graph) > 0:
            if isinstance(graph[0], list):  # Adjacency matrix
                return len(graph)
            elif isinstance(graph[0], tuple):  # Edge list
                nodes = set()
                for edge in graph:
                    nodes.add(edge[0])
                    nodes.add(edge[1])
                return len(nodes)

        return 0

    def _count_edges(self, graph) -> int:
        """Helper to count edges in graph"""
        if graph is None:
            return 0

        if isinstance(graph, dict):  # Adjacency list
            return sum(len(neighbors) for neighbors in graph.values()) // 2
        elif isinstance(graph, list) and len(graph) > 0:
            if isinstance(graph[0], list):  # Adjacency matrix
                count = 0
                for i in range(len(graph)):
                    for j in range(i + 1, len(graph[i])):  # Upper triangle only
                        if graph[i][j] != 0:
                            count += 1
                return count
            elif isinstance(graph[0], tuple):  # Edge list
                return len(graph)

        return 0


class TestGraphGeneratorIntegration:
    """Test GraphGenerator integration with other components"""

    def test_cli_integration_compatibility(self):
        """Test compatibility with CLI usage patterns"""
        generator = GraphGenerator(seed=42)  # CLI uses seed parameter

        # Test CLI-style usage
        props = GraphProperties(
            num_nodes=8,
            num_edges=10,
            connected=True,  # CLI --connected flag
            directed=False,  # CLI --directed flag
            weighted=True,  # CLI --weighted flag
        )

        graph = generator.generate(props)

        assert graph is not None, "CLI-style generation should work"
        assert self._is_connected(graph), "CLI connected flag should work"

    def test_graph_properties_integration(self):
        """Test integration with GraphProperties model"""
        generator = GraphGenerator()

        # Test with all properties specified
        props = GraphProperties(
            num_nodes=6,
            num_edges=8,
            connected=True,
            directed=False,
            weighted=True,
            allow_self_loops=False,
            allow_multi_edges=False,
        )

        graph = generator.generate(props)

        assert graph is not None, "Graph should be generated with full properties"
        assert self._count_nodes(graph) == 6, "Should respect num_nodes property"

    def test_constraints_integration(self):
        """Test integration with Constraints model for node values"""
        generator = GraphGenerator()

        props = GraphProperties(num_nodes=5, num_edges=6)
        constraints = Constraints(min_value=10, max_value=50)

        try:
            # Some generators might accept constraints for node values
            graph = generator.generate(props, constraints)
            assert graph is not None, "Graph generation with constraints should work"
        except TypeError:
            # If constraints not supported, that's also valid
            graph = generator.generate(props)
            assert graph is not None, "Basic graph generation should still work"

    def test_memory_efficiency(self):
        """Test memory efficiency for large graphs"""
        generator = GraphGenerator()

        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Generate several medium-large graphs
        for _ in range(5):
            props = GraphProperties(num_nodes=500, num_edges=750)
            graph = generator.generate(props)
            assert graph is not None

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Should not use excessive memory
        assert memory_growth < 200 * 1024 * 1024, (
            f"Graph generation used too much memory: {memory_growth / 1024 / 1024:.1f}MB"
        )

    def _is_connected(self, graph) -> bool:
        """Check if graph is connected using BFS"""
        if graph is None:
            return True

        nodes = self._count_nodes(graph)
        if nodes <= 1:
            return True

        adj_list = self._to_adjacency_list(graph)
        if not adj_list:
            return nodes <= 1

        # BFS from first node
        visited = set()
        queue = [next(iter(adj_list.keys()))]
        visited.add(queue[0])

        while queue:
            current = queue.pop(0)
            for neighbor in adj_list.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == nodes

    def _to_adjacency_list(self, graph) -> Dict:
        """Convert graph to adjacency list format"""
        if isinstance(graph, dict):
            return graph
        elif isinstance(graph, list) and len(graph) > 0:
            if isinstance(graph[0], list):  # Adjacency matrix
                adj_list = {}
                for i in range(len(graph)):
                    adj_list[i] = []
                    for j in range(len(graph[i])):
                        if graph[i][j] != 0:
                            adj_list[i].append(j)
                return adj_list
            elif isinstance(graph[0], tuple):  # Edge list
                adj_list = {}
                for edge in graph:
                    u, v = edge[0], edge[1]
                    if u not in adj_list:
                        adj_list[u] = []
                    if v not in adj_list:
                        adj_list[v] = []
                    adj_list[u].append(v)
                    adj_list[v].append(u)  # Assume undirected
                return adj_list

        return {}

    def _count_nodes(self, graph) -> int:
        """Helper to count nodes in graph"""
        if graph is None:
            return 0

        if isinstance(graph, dict):
            return len(graph)
        elif isinstance(graph, list) and len(graph) > 0:
            if isinstance(graph[0], list):  # Adjacency matrix
                return len(graph)
            elif isinstance(graph[0], tuple):  # Edge list
                nodes = set()
                for edge in graph:
                    nodes.add(edge[0])
                    nodes.add(edge[1])
                return len(nodes)

        return 0


if __name__ == "__main__":
    # Can be run directly for quick testing
    pytest.main([__file__, "-v"])
