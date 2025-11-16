"""
Computational graph builder from traced operations.
Tracks dependencies and data flow between operations.
"""

from typing import List, Dict, Set, Optional, Any
from ..tracers.tracer import Operation
import numpy as np


class GraphNode:
    """Node in the computational graph."""
    
    def __init__(self, operation: Operation):
        self.operation = operation
        self.inputs: List['GraphNode'] = []
        self.outputs: List['GraphNode'] = []
        self.visited = False
        
    def __repr__(self):
        return f"GraphNode(op={self.operation.op_name}, id={self.operation.op_id})"


class ComputationalGraph:
    """Represents the computational graph of operations."""
    
    def __init__(self, operations: List[Operation]):
        self.operations = operations
        self.nodes: List[GraphNode] = []
        self.array_to_node: Dict[int, GraphNode] = {}  # id(array) -> node that created it
        
    def build(self):
        """Build the computational graph from operations."""
        # Create nodes for each operation
        for op in self.operations:
            node = GraphNode(op)
            self.nodes.append(node)
            
            # Track array creation
            if isinstance(op.result, np.ndarray):
                self.array_to_node[id(op.result)] = node
        
        # Build edges (dependencies)
        for i, node in enumerate(self.nodes):
            op = node.operation
            
            # Find dependencies: arrays used as inputs
            for arg in op.args:
                if isinstance(arg, np.ndarray):
                    # Find which node created this array
                    # Check previous nodes in reverse order
                    for prev_node in reversed(self.nodes[:i]):
                        if (isinstance(prev_node.operation.result, np.ndarray) and
                            id(prev_node.operation.result) == id(arg)):
                            node.inputs.append(prev_node)
                            prev_node.outputs.append(node)
                            break
                    else:
                        # Could be an input array
                        pass
        
        # Find input nodes (nodes with no dependencies)
        self.input_nodes = [node for node in self.nodes if len(node.inputs) == 0]
        
        # Find output nodes (nodes with no outputs)
        self.output_nodes = [node for node in self.nodes if len(node.outputs) == 0]
        
    def topological_sort(self) -> List[GraphNode]:
        """Return nodes in topological order."""
        result = []
        visited = set()
        
        def visit(node: GraphNode):
            if node in visited:
                return
            visited.add(node)
            for input_node in node.inputs:
                visit(input_node)
            result.append(node)
        
        for node in self.nodes:
            visit(node)
        
        return result
    
    def get_output_variables(self) -> List[str]:
        """Get names of output variables (if available)."""
        # This is a simplified version - in practice, you'd need
        # to track variable names during execution
        return [f"output_{i}" for i, node in enumerate(self.output_nodes)]


def build_computational_graph(operations: List[Operation]) -> ComputationalGraph:
    """
    Build a computational graph from traced operations.
    
    Args:
        operations: List of captured operations
        
    Returns:
        ComputationalGraph instance
    """
    graph = ComputationalGraph(operations)
    graph.build()
    return graph

