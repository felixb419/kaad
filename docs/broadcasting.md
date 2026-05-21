# Broadcasting Rules {#broadcasting}

KAAD uses `compatible broadcasting semantics identical to NumPy` for all tensor operations that support broadcasting.

This means that whenever two tensors are combined elementwise (e.g. addition, subtraction, multiplication, division, and batch axes in higher-order operations), their shapes are aligned according to the same rules used by NumPy.

No implicit data copying occurs during broadcasting; instead, broadcasting is implemented through stride manipulation.

## Broadcasting Rules

Two shapes are considered broadcast-compatible if, when compared element-by-element from the trailing (rightmost) axis:

**1.** The axes are equal, or
**2.** One of the axes is has extent 1, or
**3.** One of the tensors does not have a axis at that position (it is treated as 1)

If neither condition holds, the shapes are incompatible and the operation is invalid.

## Broadcasting Procedure

When two tensors are broadcast together:

**1.** The shape with fewer axes is left-padded with 1s.

**2.** Each axis pair is compared from right to left.

**3.** The resulting broadcasted shape is computed as:

- the maximum of each axis pair

## Result Shape Rule

For each axis i:

```text
result_shape[i] = max(shape_a[i], shape_b[i])
```

(where missing axes are treated as `1`)

## Examples

### Scalar with tensor

```text
A: []
B: [3, 4]

Result: [3, 4]
```

A scalar is treated as shape `[1, 1]...` conceptually and broadcast to match the tensor.

### Vector with matrix

```text
A: [4]
B: [3, 4]

Result: [3, 4]
```

The vector is broadcast across the first axis.

### Compatible trailing axes

```text
A: [2, 3, 1]
B: [2, 1, 4]

Result: [2, 3, 4]
```

Broadcasting happens independently per axis:

- 2 vs 2 → 2
- 3 vs 1 → 3
- 1 vs 4 → 4

### Incompatible shapes

```text
A: [2, 3]
B: [2, 4]
```

Fails because:

- last axis: 3 vs 4 (neither is 1)

### Higher-rank broadcasting

```
A: [5, 1, 3]
B: [1, 4, 1]

Result: [5, 4, 3]
```

## Important Implementation Note

KAAD implements broadcasting using `stride manipulation rather than memory duplication`.

In broadcasted axes:

- the stride may be set to 0
- the same memory is reused across repeated logical indices

This ensures broadcasting is zero-copy and compatible with the view-based tensor system.

## Relationship to KAAD operations

Broadcasting is used in:

- elementwise binary operators (`add`, `mul`, etc.)
- gradients during backward propagation
- batch axes in higher-rank operations (e.g. matrix multiplication)
