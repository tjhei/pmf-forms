# deal.II Form Language

A small C++ expression-template prototype for describing finite-element forms
and converting supported forms into deal.II matrix-free operators. The same
form can drive either `dealii::MatrixFree` with `FEEvaluation`, or
`dealii::Portable::MatrixFree` with `Portable::FEEvaluation`.

The goal is to keep the mathematical expression independent of the execution
backend. A form is passed to `make_matrix_free_operator` or
`make_portable_matrix_free_operator`; the operator inspects the formal field
shapes at compile time and selects scalar or vector evaluation.

For a single-field form, `trial()` and `test()` use default scalar symbols;
write `trial<ValueShape::vector>()` and `test<ValueShape::vector>()` for vector
fields. For mixed forms, `trial_functions<Shapes...>()` and
`test_functions<Shapes...>()` return tuples numbered from zero in declaration
order. Matching trial/test positions identify the same field/block; fields
with the same shape remain distinct. Each call restarts numbering at zero.
Coefficient tags identify independently bound coefficients.

## Examples

### Scalar Laplace

```cpp
using namespace pmf::forms;
using namespace pmf::forms::expression_templates;

auto u = trial();
auto v = test();

const auto laplace = integral(inner(grad(v), grad(u)), dx);
```

This form has no coefficient binding. The factory deduces its type and chooses
scalar `FEEvaluation` from the form's trial/test shapes:

```cpp
auto cpu_operator = make_matrix_free_operator<dim, degree>(matrix_free, laplace);
```

Use the same form with the portable backend:

```cpp
auto portable_operator = make_portable_matrix_free_operator<dim, degree>(
  portable_matrix_free, laplace);
```

Both factories accept const forms and coefficient bindings and store their own
values. No form-type alias or `std::decay` is needed. Explicit operator aliases
also accept `decltype(form)`, including const and reference-qualified types.

### Weighted Helmholtz

The scalar form can combine a weighted diffusion term and a separately
weighted reaction term. Each coefficient has its own tag and value:

```cpp
struct DiffusionTag;
struct ReactionTag;

auto diffusion = coefficient<DiffusionTag>();
auto reaction  = coefficient<ReactionTag>();

auto helmholtz = integral(
  diffusion * inner(grad(v), grad(u)) + reaction * (v * u), dx);

auto coefficients = bind_coefficients(
  bind_coefficient<DiffusionTag>(2.0),
  bind_coefficient<ReactionTag>(0.25));

auto cpu_operator = make_matrix_free_operator<dim, degree>(
  matrix_free, helmholtz, coefficients);
auto portable_operator = make_portable_matrix_free_operator<dim, degree>(
  portable_matrix_free, helmholtz, coefficients);
```

Literal constants can also be written in the form. They are stored in the
expression and applied during quadrature evaluation:

```cpp
auto helmholtz_with_literals = integral(
  2.0 * inner(grad(v), grad(u)) + v * (0.25 * u), dx);
```

### Isotropic elasticity

The expression-template syntax also describes the symmetric-gradient form
with unit weight:

```cpp
auto u = trial<ValueShape::vector>();
auto v = test<ValueShape::vector>();

auto elasticity = integral(inner(sym(grad(v)), sym(grad(u))), dx);
```

The same shape-selecting operator API selects vector `FEEvaluation` for this
form. A constant-coefficient isotropic elasticity pattern with separate Lamé
parameters is also available through `elasticity_coefficients`; see
`tests/test_main.cc` for the current API examples.

### Stokes

Mixed forms use positional field identities: velocity is field 0 and pressure
is field 1. Field lists are sorted by index, independent of expression order.
The expression determines whether each field needs values or gradients:

```cpp
struct ViscosityTag;

auto [u, p] = trial_functions<ValueShape::vector, ValueShape::scalar>();
auto [v, q] = test_functions<ValueShape::vector, ValueShape::scalar>();
auto mu = coefficient<ViscosityTag>();

auto stokes =
  integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))), dx)
  - integral(div(v) * p, dx)
  - integral(q * div(u), dx);
```

The current Stokes lowering uses separate velocity and pressure DoFHandlers
and block vectors. CPU and Portable operators use the same form. `FormFields`
and `FieldRequirements` expose expression-derived compile-time metadata for
field lists and value/gradient requirements. Run the MPI comparison with:

```sh
mpiexec -n 2 ./build/stokes
```

## Current scope

The implemented matrix-free subset is scalar Laplace and Helmholtz forms,
including independently tagged constant diffusion and reaction coefficients,
the supported isotropic elasticity patterns, and a two-field Stokes form.
Coefficient symbols bind to constant values today; spatially varying
coefficient fields and general expression-to-kernel lowering are not yet
implemented. The current mixed lowering supports the Stokes form above.
Assembled `FEValues` operators, boundary and face terms, and diagonal
computation are also future work. See [plan.md](plan.md) for the roadmap and
status.

## Build and test

This project expects a configured deal.II build. Use the existing `build/`
directory:

```sh
cmake --build build
ctest --test-dir build --output-on-failure
```

The elasticity executable applies one unit-weight elasticity form using both
backends, compares the resulting distributed vectors, and exits with an error
if their relative difference exceeds the tolerance. For example, run it on two
MPI ranks with:

```sh
mpiexec -n 2 ./build/elasticity
```

The Laplace executable does the same comparison for a scalar diffusion form
with coefficient `2.5`:

```sh
mpiexec -n 2 ./build/laplace
```

Catch2 is fetched by CMake into the build tree. Format project C++ files with:

```sh
cmake --build build --target indent
```

See [AGENTS.md](AGENTS.md) for development instructions and
[`reference/`](reference/) for deal.II example implementations.
