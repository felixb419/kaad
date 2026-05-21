# Operators

Operators are the primary mechanism for constructing computation graphs. Each operator creates a new node in the graph representing a differentiable tensor operation.
These functions operate on Node objects and return new nodes that are automatically inserted into the computation graph.

All operators are graph-building operations, not immediate evaluations. Each call records the operation in the graph structure, which is later evaluated during the forward pass and differentiated during the backward pass.

The rules for broadcasting inputs are outlined here [broadcasting_rules.md](broadcasting.md).

The following operators are currently supported:

## Input operator

- @ref kaad::Graph::input

## Binary operators

- @ref kaad::Graph::add
- @ref kaad::Graph::sub
- @ref kaad::Graph::mul
- @ref kaad::Graph::div
- @ref kaad::Graph::pow
- @ref kaad::Graph::min
- @ref kaad::Graph::max
- @ref kaad::Graph::dot
- @ref kaad::Graph::matmul
- @ref kaad::Graph::outer

## Unary operators

- @ref kaad::Graph::negative
- @ref kaad::Graph::square
- @ref kaad::Graph::sqrt
- @ref kaad::Graph::log
- @ref kaad::Graph::exp
- @ref kaad::Graph::abs
- @ref kaad::Graph::slice
- @ref kaad::Graph::sum
- @ref kaad::Graph::mean
- @ref kaad::Graph::transpose
