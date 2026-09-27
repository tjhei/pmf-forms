# deal.II Form Language

A small C++ expression-template prototype for describing finite-element forms
and converting supported forms into deal.II matrix-free operators. The same
form can drive either `dealii::MatrixFree` with `FEEvaluation`, or
`dealii::Portable::MatrixFree` with `Portable::FEEvaluation`.

The goal is to keep the mathematical expression independent of the execution
backend. A form is passed to `MatrixFreeOperator` or
`PortableMatrixFreeOperator`; the operator inspects the formal field shapes at
compile time and selects scalar or vector evaluation.

For a single-field form, `trial()` and `test()` use default scalar symbols;
write `trial<ValueShape::vector>()` and `test<ValueShape::vector>()` for vector
fields. Explicit tags remain available when a form has multiple trial/test
arguments or multiple independently bound coefficients.

## Examples

### Scalar Laplace

```cpp
using namespace pmf::forms;
using namespace pmf::forms::expression_templates;

auto u = trial();
auto v = test();

auto laplace = integral(inner(grad(v), grad(u)), dx);
```

This form has no coefficient binding. The `MatrixFreeOperator` API chooses
scalar `FEEvaluation` from the form's trial/test shapes:

```cpp
using Form = typename std::decay<decltype(laplace)>::type;
MatrixFreeOperator<dim, degree, Form> cpu_operator(matrix_free, laplace);
```

Use the same form with the portable backend:

```cpp
PortableMatrixFreeOperator<dim, degree, Form> portable_operator(
  portable_matrix_free, laplace);
```

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

using HelmholtzForm = typename std::decay<decltype(helmholtz)>::type;
using Coefficients = typename std::decay<decltype(coefficients)>::type;

MatrixFreeOperator<dim, degree, HelmholtzForm, Coefficients> cpu_operator(
  matrix_free, helmholtz, coefficients);
PortableMatrixFreeOperator<dim, degree, HelmholtzForm, Coefficients>
  portable_operator(portable_matrix_free, helmholtz, coefficients);
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

## Current scope

The implemented matrix-free subset is scalar Laplace and Helmholtz forms,
including independently tagged constant diffusion and reaction coefficients,
plus the supported isotropic elasticity patterns. Coefficient symbols bind to
constant values today; spatially varying coefficient fields are not yet
implemented. Stokes and mixed systems can be represented by the expression
types, but are not yet lowered to MatrixFree operators. Assembled `FEValues`
operators, boundary and face terms, and diagonal computation are also future
work. See [plan.md](plan.md) for the roadmap and status.

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
