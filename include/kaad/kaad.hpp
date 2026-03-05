#pragma once

#include "graph/computation_graph.hpp"
#include "operators/operators.hpp"
#include "scalar.hpp"
#include "tensor/tensor.hpp"

/**
 * @mainpage KAAD Documentation
 *
 * # KAAD: Kinda Alright Auto Differentiation
 *
 * KAAD is a lightweight C++ library for multi-dimensional tensor operations and
 * automatic differentiation using computation graphs. It enables forward
 * evaluation of tensor expressions and computation of gradients via
 * backward-mode autodiff.
 *
 * ## Features
 * - Support for tensors with arbitrary dimensions
 * - Efficient forward evaluation of computation graphs
 * - Backward-mode automatic differentiation
 *
 * ## Getting Started
 * 1. Include the header file: `"kaad.hpp"`.
 * 2. Create a `kaad::Computation_graph` object.
 * 3. Use the `add_input_node` member function to add leaf nodes (which hold
 * value and gradient tensors).
 * 4. Apply 'operators' to add operation nodes to the graph.
 *
 * Once the computation graph is built:
 * 1. Use `reset()` to reset all values of non-leaf nodes to 0 and reset all
 * gradients to 0.
 * 2. Call `eval()` on a `CompGraph` object to evaluate specific nodes.
 * 3. Use `getGrad()` to compute gradients of one or more tensors.
 *
 * This structure allows you to easily compose and differentiate complex
 * tensor-based computations.
 */
