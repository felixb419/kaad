# TODO

## FIX

## REFACTOR

- [?] [graph] put reset, evaluate and getGradient into single function

## FEAT

- [?] [graph] Add system for inference

## PERF

- [graph] make stride function transpose axes so biggest is highest

- [graph] implement system for memory aliasing
  - [operations] transpose should alias memory instead of copying

- [graph] Add a way to move values to input nodes without copying

- [tensor] make Tensor variant with compile time shape

## DOC

## CHORE

## BUILD

## LATER

- [operations] use `std::mdspan` as soon as available
