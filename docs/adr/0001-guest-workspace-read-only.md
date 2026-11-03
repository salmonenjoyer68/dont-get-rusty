# Guest filesystem is read-only; durable writes go through host tools

The Guest runs model-generated code with a preopened workspace. A read-write preopen would let that code overwrite files without the host's confirmation, bypass the 50 MB quota (`wasmtime-wasi` 50.0.1's `preopened_dir` exposes only `FsPerms::ReadOnly`/`ReadWrite`, with no quota hook), and hide file mutations inside a single `run_code` audit event. The Guest therefore gets read-only preopens of the Runtime Image at `/` and the Workspace at `/workspace`; every durable write goes through the host `write_file` tool, which is audited, quota-checked, and confirmable.

Consequences: guest code cannot write files — artifacts return via stdout or are written by the model; host tools use `cap-std` while the preopen uses Wasmtime's vendored primitives, so both path resolvers must pass the escape suite.
