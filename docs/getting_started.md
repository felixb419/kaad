# Getting Started {#gettingStarted}

## 1. Include the library

```c++
#include <kaad/kaad.hpp>
```

## 2. Create a computation graph

The `Graph` object records all tensor operations and owns all nodes and associated memory.

```c++
kaad::Graph graph;
```

## 3. Create input nodes

Input nodes represent externally provided tensors. Their shapes must be specified at creation time.

```c++
kaad::Node input_a = graph.input(kaad::Shape{3, 2});
kaad::Node input_b = graph.input(kaad::Shape{});
kaad::Node input_c = graph.input(kaad::Shape{2, 3, 1});
```

## 4. Build the computation graph

Operations are recorded implicitly through operator functions on the graph.

```c++
kaad::Node sqrt_a = graph.sqrt(input_a);
kaad::Node a_plus_b = graph.add(sqrt_a, input_b);
kaad::Node result = graph.mul(a_plus_b, input_c);
```

## 5. Initialize tensor storage

Before evaluation, the graph must allocate all required tensor memory and initialize all nodes.

```c++
graph.init();
```

After this step, input nodes expose mutable memory for data insertion.

## 6. Provide input data

Input tensors can be filled using raw memory access.

```c++
std::fill_n(input_a.data_mut(), input_a.size(), 10);
std::fill_n(input_b.data_mut(), input_b.size(), 30);
std::fill_n(input_c.data_mut(), input_c.size(), 50);
```

## 7. Reset the graph

Before each evaluation or gradient computation, the graph must be reset to clear internal state.

```c++
graph.reset();
```

## 8. Forward evaluation

Evaluate one or more output nodes to compute forward values.

```c++
std::vector<TensorView> vals = graph.evaluate(std::array{res});
TensorView res_val = vals[0];
```

Alternatively, values can be accessed directly from nodes after evaluation:

```c++
graph.evaluate(std::array{res});
TensorView res_val = res.value();
```

## 9. Backward pass (gradients)

Compute gradients of selected inputs with respect to an output node.

```c++
std::vector<TensorView> grads = graph.get_gradient(res, std::array{input_a, input_b, input_c});
a_grad = grads[0];
b_grad = grads[1];
c_grad = grads[2];
```

Gradients are also accessible directly from input nodes after backpropagation:

```c++
graph.get_gradient(res, std::array{input_a, input_b, input_c});
a_grad = input_a.grad();
b_grad = input_b.grad();
c_grad = input_c.grad();
```

## 10. Reuse the graph

To run another evaluation with new input values:

```c++
graph.reset();
```

Then update input data and repeat from step 6.
