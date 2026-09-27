# Development instructions

## Build

Use the existing `build/` directory, which is configured for the installed
deal.II build:

```sh
cmake --build build
```

If CMake needs to be configured again, run `cmake -S . -B build` from the
repository root. Catch2 is fetched by CMake into the build tree.

## Tests

Build and run the unit tests with:

```sh
cmake --build build --target pmf_form_tests
ctest --test-dir build --output-on-failure
```

Add unit tests under `tests/` using Catch2.

## Examples

Look in `reference/` for example implementations and source material relevant
to this project.

## Formatting

Format all project `.h` and `.cc` files with the `indent` target:

```sh
cmake --build build --target indent
```
