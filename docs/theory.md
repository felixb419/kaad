# Reverse Mode Autodiff (Backpropagation)

Reverse mode automatic differentiation (autodiff) computes derivatives efficiently by traversing a computation **backward** from the output to the inputs.

Its goal is to compute gradients like:

\f[
\frac{\partial y}{\partial x_i}
\f]

where $y$ is a scalar output (such as a loss function), and $x_i$ are the input variables or model parameters.

---

## Core Idea

Instead of computing derivatives separately for each input, reverse mode applies the **chain rule backward** through the computation graph.

Given a function:

\f[
y = f(x_1, x_2, \dots, x_n)
\f]

we evaluate it in two phases:

### 1. Forward Pass

Compute all intermediate values normally and store them.

Example:

\f[
a = x_1 x_2
\f]

\f[
y = \sin(a)
\f]

This builds a computation graph and caches values needed later.

---

### 2. Backward Pass

Starting from:

\f[
\frac{\partial y}{\partial y} = 1
\f]

propagate gradients backward using the chain rule:

\f[
\frac{\partial y}{\partial a}
=
\frac{\partial y}{\partial y}
\cdot
\frac{dy}{da}
=
\cos(a)
\f]

Then distribute to inputs:

\f[
\frac{\partial y}{\partial x_1}
=
\frac{\partial y}{\partial a}
\cdot
\frac{\partial a}{\partial x_1}
=
\cos(a)\,x_2
\f]

\f[
\frac{\partial y}{\partial x_2}
=
\frac{\partial y}{\partial a}
\cdot
\frac{\partial a}{\partial x_2}
=
\cos(a)\,x_1
\f]

Each node receives a gradient from downstream and passes contributions upstream.

---

## Computational Graph View

A program can be represented as a graph:

```text
x₁ ----\
        (*) → a → sin → y
x₂ ----/
```
