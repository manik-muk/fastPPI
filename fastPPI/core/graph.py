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
        # Try to import pandas for type checking
        try:
            import pandas as pd
            PANDAS_AVAILABLE = True
        except ImportError:
            PANDAS_AVAILABLE = False
            pd = None
        
        # Try to import pandas tracer for PandasOperation
        try:
            from ..tracers.pandas_tracer import PandasOperation
        except ImportError:
            PandasOperation = None
        
        # Create nodes for each operation
        for op in self.operations:
            node = GraphNode(op)
            self.nodes.append(node)
            
            # Track array creation (NumPy arrays)
            if isinstance(op.result, np.ndarray):
                self.array_to_node[id(op.result)] = node
            
            # Track pandas objects (DataFrame/Series)
            if PANDAS_AVAILABLE and pd is not None:
                try:
                    if isinstance(op.result, (pd.DataFrame, pd.Series)):
                        self.array_to_node[id(op.result)] = node
                except (TypeError, AttributeError):
                    pass
        
        # Build edges (dependencies)
        for i, node in enumerate(self.nodes):
            op = node.operation
            
            # Find dependencies: arrays/DataFrames/Series used as inputs
            for arg in op.args:
                # Check for NumPy arrays
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
                
                # Check for pandas objects (DataFrame/Series)
                elif PANDAS_AVAILABLE and pd is not None:
                    try:
                        if isinstance(arg, (pd.DataFrame, pd.Series)):
                            # Find which node created this pandas object
                            for prev_node in reversed(self.nodes[:i]):
                                try:
                                    if isinstance(prev_node.operation.result, (pd.DataFrame, pd.Series)) and \
                                       id(prev_node.operation.result) == id(arg):
                                        node.inputs.append(prev_node)
                                        prev_node.outputs.append(node)
                                        break
                                except (TypeError, AttributeError):
                                    pass
                            else:
                                # Could be an input DataFrame/Series
                                pass
                    except (TypeError, AttributeError):
                        pass
        
        # Find input nodes (nodes with no dependencies)
        self.input_nodes = [node for node in self.nodes if len(node.inputs) == 0]
        
        # Find output nodes (nodes with no outputs)
        self.output_nodes = [node for node in self.nodes if len(node.outputs) == 0]
    
    def eliminate_dead_code(self):
        """
        Remove operations that are not used (dead code elimination).
        A node is live if:
        1. It's an output node (final result), OR
        2. It's used as input by a live node
        
        This performs a reverse traversal from output nodes to mark all live nodes.
        Conservative approach: keep all output nodes and everything they depend on.
        """
        if not self.nodes:
            return 0
        
        if not self.output_nodes:
            # No outputs, but be conservative - don't remove everything
            # Only remove nodes that are truly isolated (no inputs, no outputs)
            original_count = len(self.nodes)
            self.nodes = [node for node in self.nodes if len(node.inputs) > 0 or len(node.outputs) > 0]
            removed_count = original_count - len(self.nodes)
            return removed_count
        
        # Mark all nodes as dead initially
        live_nodes = set()
        
        # Start from output nodes and traverse backwards
        def mark_live(node: GraphNode):
            """Recursively mark node and all its dependencies as live."""
            if node in live_nodes:
                return  # Already marked
            live_nodes.add(node)
            # Mark all input nodes (dependencies) as live
            for input_node in node.inputs:
                mark_live(input_node)
        
        # Mark all output nodes and their dependencies as live
        for output_node in self.output_nodes:
            mark_live(output_node)
        
        # Filter to only live nodes
        original_count = len(self.nodes)
        self.nodes = [node for node in self.nodes if node in live_nodes]
        removed_count = original_count - len(self.nodes)
        
        # Rebuild edges for remaining nodes (remove dead edges)
        for node in self.nodes:
            node.inputs = [inp for inp in node.inputs if inp in live_nodes]
            node.outputs = [out for out in node.outputs if out in live_nodes]
        
        # Recalculate input and output nodes
        self.input_nodes = [node for node in self.nodes if len(node.inputs) == 0]
        self.output_nodes = [node for node in self.nodes if len(node.outputs) == 0]
        
        return removed_count
        
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

