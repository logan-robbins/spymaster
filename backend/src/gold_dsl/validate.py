"""Gold DSL validator: structural and semantic checks on a GoldDslSpec.

Validates the DAG for correctness before any execution:
    - Reference integrity (all source/left/right resolve)
    - Acyclicity via Kahn's algorithm
    - Silver column existence
    - Output uniqueness and presence
    - Parameter constraints (ewm alpha, zscore window_bins)
"""
from __future__ import annotations

from collections import defaultdict, deque

from ..qmachina.stage_schema import SILVER_COLS
from .schema import (
    ArithmeticExpr,
    GoldDslSpec,
    NormExpr,
    OutputNode,
    SilverRef,
    SpatialNeighborhood,
    TemporalWindow,
)


def _get_source_refs(node: object) -> list[str]:
    """Extract all upstream node name references from a DSL node.

    Args:
        node: A DSL node instance.

    Returns:
        List of referenced node names (may be empty for SilverRef).
    """
    if isinstance(node, SilverRef):
        return []
    if isinstance(node, (TemporalWindow, SpatialNeighborhood, NormExpr, OutputNode)):
        return [node.source]
    if isinstance(node, ArithmeticExpr):
        return [node.left, node.right]
    return []


def validate_dsl(spec: GoldDslSpec) -> list[str]:
    """Validate a GoldDslSpec for structural and semantic correctness.

    Performs the following checks in order:
        1. All source/left/right references point to existing node names.
        2. No cycles in the DAG (Kahn's topological sort).
        3. SilverRef fields exist in SILVER_COLS.
        4. At least one OutputNode is present.
        5. OutputNode names are unique.
        6. ewm alpha is provided and in (0, 1) when agg="ewm".
        7. zscore window_bins is provided when method="zscore".
        8. Type compatibility: arithmetic/norm sources must resolve to
           numeric nodes (not OutputNodes used as intermediates).

    Args:
        spec: The GoldDslSpec to validate.

    Returns:
        List of error strings. An empty list indicates a valid spec.
    """
    errors: list[str] = []
    node_names = set(spec.nodes.keys())

    # 1. Reference integrity
    for name, node in spec.nodes.items():
        for ref in _get_source_refs(node):
            if ref not in node_names:
                errors.append(
                    f"Node '{name}' references unknown node '{ref}'"
                )

    # 2. Cycle detection via Kahn's algorithm
    in_degree: dict[str, int] = {name: 0 for name in node_names}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for name, node in spec.nodes.items():
        for ref in _get_source_refs(node):
            if ref in node_names:
                adjacency[ref].append(name)
                in_degree[name] += 1

    queue: deque[str] = deque(
        name for name, degree in in_degree.items() if degree == 0
    )
    sorted_count = 0
    while queue:
        current = queue.popleft()
        sorted_count += 1
        for downstream in adjacency[current]:
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                queue.append(downstream)

    if sorted_count != len(node_names):
        cycle_members = sorted(
            name for name, degree in in_degree.items() if degree > 0
        )
        errors.append(
            f"Cycle detected involving nodes: {cycle_members}"
        )

    # 3. SilverRef field existence
    for name, node in spec.nodes.items():
        if isinstance(node, SilverRef):
            if node.field not in SILVER_COLS:
                errors.append(
                    f"Node '{name}': silver field '{node.field}' does not "
                    f"exist in SILVER_COLS"
                )

    # 4. At least one OutputNode
    output_nodes = [
        (name, node)
        for name, node in spec.nodes.items()
        if isinstance(node, OutputNode)
    ]
    if not output_nodes:
        errors.append("Spec must contain at least one OutputNode")

    # 5. Output names uniqueness
    output_names = [node.name for _, node in output_nodes]
    seen_names: set[str] = set()
    for oname in output_names:
        if oname in seen_names:
            errors.append(f"Duplicate output name: '{oname}'")
        seen_names.add(oname)

    # 6. ewm alpha validation
    for name, node in spec.nodes.items():
        if isinstance(node, TemporalWindow) and node.agg == "ewm":
            if node.alpha is None:
                errors.append(
                    f"Node '{name}': ewm aggregation requires alpha parameter"
                )
            elif not (0.0 < node.alpha < 1.0):
                errors.append(
                    f"Node '{name}': ewm alpha must be in (0, 1), "
                    f"got {node.alpha}"
                )

    # 7. zscore window_bins validation
    for name, node in spec.nodes.items():
        if isinstance(node, NormExpr) and node.method == "zscore":
            if node.window_bins is None:
                errors.append(
                    f"Node '{name}': zscore normalization requires "
                    f"window_bins parameter"
                )

    # 8. Type compatibility: arithmetic and norm sources must not be OutputNodes
    for name, node in spec.nodes.items():
        if isinstance(node, ArithmeticExpr):
            for ref_name, side in [(node.left, "left"), (node.right, "right")]:
                if ref_name in node_names:
                    ref_node = spec.nodes[ref_name]
                    if isinstance(ref_node, OutputNode):
                        errors.append(
                            f"Node '{name}': {side} operand '{ref_name}' is an "
                            f"OutputNode, which cannot be used as an arithmetic input"
                        )
        if isinstance(node, NormExpr):
            if node.source in node_names:
                ref_node = spec.nodes[node.source]
                if isinstance(ref_node, OutputNode):
                    errors.append(
                        f"Node '{name}': source '{node.source}' is an "
                        f"OutputNode, which cannot be used as a norm input"
                    )

    return errors
