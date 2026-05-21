# Memory Model / Tensor Layout

## Overview

Tensors in this library follow a **row-major logical ordering** as the default indexing convention. However, row-major ordering does not imply contiguity in memory for all tensors.

## Contiguity and input guarantees

A contiguous row-major memory layout is **only guaranteed for tensors originating from input nodes**. These tensors are physically stored in a dense, contiguous block of memory consistent with row-major layout.

All other tensors produced by operations in the computational graph are not guaranteed to be contiguous.

Contiguity can be queried explicitly using:

- is_contiguous()

This check reflects the actual memory layout and stride configuration of the tensor at runtime.

## View-based system

All tensors produced by the computational graph are represented as views.

A tensor returned from any graph operation is a TensorView, meaning:

- It does not own its underlying memory
- It does not allocate or copy data by default
- It represents a transformation over one or more underlying storage buffers via stride metadata

As a consequence, operations such as broadcasting, transposition, slicing, and reshaping are implemented through modified views, not materialized copies.

## Immutability and memory ownership

All TensorView objects are **logically immutable**:

- Their shape and stride metadata cannot be modified after creation
- They do not expose mutating operations on their underlying storage
- They do not share ownership of mutable memory regions

Instead, computation produces new views derived from existing tensors, while memory ownership remains isolated in input nodes (or explicitly materialized tensors, if supported by the library).

## Stride-based layout

Tensor memory layout is fully defined by its stride .

Strides determine how multi-dimensional indexing maps to linear memory offsets, and allow the same underlying storage to be interpreted in multiple ways without copying.

Because views may modify strides without modifying storage, most tensors in the graph should be treated as **non-contiguous by default**, even if their logical shape suggests a dense layout.
