# Reverse Mode Autodiff

We compute gradients using reverse accumulation.

$$
\frac{\partial y}{\partial x_i}
$$

## Intuition

Reverse mode is efficient when:
- many inputs
- single output