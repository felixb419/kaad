# Graph

The computation graph is the centerpiece of the library. It represents the full sequence of tensor operations required to compute a result and its corresponding gradients during the backward pass.

Nodes are added to the graph implicitly through tensor operations (operator overloading and functional APIs). The supported operations are documented in [operators.md](operators.md).

## Node storage and structure

The graph maintains a linear container (typically a `std::vector`) of pointers to `INode`, which serves as the base interface for all graph nodes.

Each node in the graph is either:

- an `InputNode`, representing a user-provided tensor
- an `OperatorNode`, representing a computation applied to one or more input nodes

The `INode` interface provides a uniform abstraction over these node types, allowing the graph to treat all operations uniformly during both forward and backward passes.

## Memory ownership

The graph is the `sole owner of all node memory`, including all tensors stored within those nodes.

This means:

- All tensors associated with nodes are owned by the graph
- Tensor lifetime is strictly bound to the lifetime of the graph
- No node or tensor is responsible for freeing memory outside the graph
- Destroying the graph invalidates all associated tensors and views

This ownership model ensures deterministic memory management and avoids accidental lifetime extension of intermediate computation results.

## Design implications

Because the graph owns all node and tensor memory:

- Tensor views returned from operations are only valid as long as the graph exists
- Intermediate results are stored implicitly inside nodes rather than externally materialized
- The backward pass operates directly over the owned node structure without additional allocation of computational state

This design prioritizes performance and locality at the cost of requiring strict lifetime discipline around graph objects.
