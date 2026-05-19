#pragma once

#include "kaad/tensor/internal/tensor.hpp"

#include <kaad/graph/graph.hpp>
#include <kaad/operators/operators.hpp>
#include <kaad/scalar.hpp>

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
 * 2. Create a `kaad::Graph` object.
 * 4. Apply 'operators' to add operation nodes to the graph.
 *
 * Once the computation graph is built:
 * 1. Use init() function to allocate tensor memory and initialize node
 * parameters.
 * 1. Use `reset()` to reset all values of non-leaf nodes to 0 and reset all
 * gradients to 0.
 * 2. Call `eval()` on a `CompGraph` object to evaluate specific nodes.
 * 3. Use `get_grad()` to compute gradients of one or more tensors.
 *
 * This structure allows you to easily compose and differentiate complex
 * tensor-based computations.
 */
