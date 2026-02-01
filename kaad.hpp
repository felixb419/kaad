#pragma once

#include "computation_graph/computation_graph.hpp"
#include "computation_graph/nodes/nodes.hpp"
#include "computation_graph/operators/operators.hpp"
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
 * 2. Create a `kaad::CompGraph<T>` object.
 * 3. Use the `append` member function to add leaf nodes (which hold value and
 * gradient tensors).
 * 4. Apply unary or binary operations to add operation nodes to the graph.
 *
 * Once the computation graph is built:
 * - Call `eval()` on a `CompGraph` object to evaluate specific nodes.
 * - Use `getGrad()` to compute gradients of one or more tensors.
 * - Use `reset()` to reset all values of non-leaf nodes to 0 and reset all
 * gradients to 0.
 *
 * This structure allows you to easily compose and differentiate complex
 * tensor-based computations.
 */
