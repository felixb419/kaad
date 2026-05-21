# Terminology

## shape

The shape of a tensor is an ordered list (or array) of integers describing the extent of each axis.

Formally, a tensor with shape S has rank len(S), and for each axis i, S[i] is the extent of axis i.

For example:

- [] represents a scalar (rank 0)
- [5] represents a vector of length 5
- [2, 3, 4] represents a rank-3 tensor with three axes

The shape fully determines the logical indexing structure of the tensor, but not its memory layout.

## strides

The strides of a tensor describe how to map multi-dimensional indices into a linear memory offset.

Each stride value corresponds to an axis and indicates how many elements in memory must be skipped when incrementing that axis by one.

Formally, for a tensor with rank r, strides is an array T of length r, where T[i] is the offset in memory (in number of elements, not bytes) between consecutive elements along axis i.

Given a multi-dimensional index (i0, i1, ..., ir-1), the linear index is computed as:

```text
offset = i0 * T[0] + i1 * T[1] + ... + i(r-1) * T[r-1]
```

In a contiguous row-major layout:

- the last axis has stride 1
- each preceding axis has stride equal to the product of extents of all later axes

For example, a tensor with shape [2, 3, 4] typically has strides:

- [12, 4, 1]

## axis

An axis is a single dimension of a tensor's shape. Each axis corresponds to one index position in the tensor’s multi-dimensional indexing scheme.

In this library, axes are ordered according to the tensor’s shape array, where axis i corresponds to shape[i].

For example, for a tensor with shape [2, 3, 4], there are three axes:

- axis 0 has extent 2
- axis 1 has extent 3
- axis 2 has extent 4

## rank

The rank of a tensor is the number of axes it has.

Equivalently, it is the length of the tensor’s shape array.

For example:

- scalar: rank 0 (shape [])
- vector: rank 1 (shape [n])
- matrix: rank 2 (shape [m, n])

## extent

The extent of a tensor along a given axis is the size (number of elements) along that axis.

Formally, for a tensor with shape S, the extent of axis i is S[i].

For example, for shape [2, 3, 4]:

- extent(axis 0) = 2
- extent(axis 1) = 3
- extent(axis 2) = 4
