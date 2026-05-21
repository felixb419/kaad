# Maximum rank

The maximum supported tensor rank in KAAD is controlled by the compile-time constant:

`KAAD_MAX_RANK`

If not explicitly defined, `KAAD_MAX_RANK` defaults to **20**.

This constant determines the maximum number of axes a tensor may have. For example, a tensor with shape `[2, 3, 4]` has rank 3 and therefore requires `KAAD_MAX_RANK >= 3`.

There is no fixed upper bound imposed by the library itself. `KAAD_MAX_RANK` may be increased as needed, subject only to practical memory and compilation constraints.

Increasing `KAAD_MAX_RANK` increases both the size of tensor metadata and the size of internal runtime dispatch tables, since shape and stride storage are allocated according to this limit.

Larger values may therefore increase memory usage and compile times. For most applications, the default value of **20** is sufficient.
