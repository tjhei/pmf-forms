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
Coefficients own their values in the expression tree: `coefficient(2.0)`.
Each occurrence stores its own copy, so values remain independent even when
the coefficient types match. Modifying the original value or coefficient does
not change an existing form or operator.

## Examples

### Scalar Laplace

```cpp
using namespace pmf::forms;
using namespace pmf::forms::expression_templates;

auto u = trial();
auto v = test();

const auto laplace = integral(inner(grad(v), grad(u)), dx);
```

This form has no coefficients. The factory deduces its type and chooses
scalar `FEEvaluation` from the form's trial/test shapes:

```cpp
auto cpu_operator = make_matrix_free_operator<dim, degree>(matrix_free, laplace);
```

Use the same form with the portable backend:

```cpp
auto portable_operator = make_portable_matrix_free_operator<dim, degree>(
  portable_matrix_free, laplace);
```

Both factories accept const forms and store their own expression values.
No form-type alias or `std::decay` is needed. Explicit operator aliases
also accept `decltype(form)`, including const and reference-qualified types.

Call `op.get_diagonal()` to obtain a const reference to its diagonal
vector (not the inverse). The first call computes and caches it; later calls
reuse the cache. This works on const CPU and Portable operators. Call
`compute_diagonal()` to force recomputation after changing the underlying
MatrixFree data. Computation is collective over the operator's MPI communicator,
so all ranks must call consistently. Constrained entries are one; unconstrained
Stokes pressure entries are zero. Ghost entries are not updated.

### Weighted Helmholtz

The scalar form can combine a weighted diffusion term and a separately
weighted reaction term. Each coefficient stores its own value:

```cpp
auto diffusion = coefficient(2.0);
auto reaction  = coefficient(0.25);

auto helmholtz = integral(
  diffusion * inner(grad(v), grad(u)) + reaction * (v * u), dx);

auto cpu_operator = make_matrix_free_operator<dim, degree>(
  matrix_free, helmholtz);
auto portable_operator = make_portable_matrix_free_operator<dim, degree>(
  portable_matrix_free, helmholtz);
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
form. For isotropic elasticity, constants are embedded directly:

```cpp
auto [u] = trial_functions<ValueShape::vector>();
auto [v] = test_functions<ValueShape::vector>();
auto lambda = coefficient(3.0);
auto mu = coefficient(2.0);

const auto elasticity = integral(
  2.0 * mu * inner(sym(grad(v)), sym(grad(u)))
    + lambda * div(v) * div(u), dx);

auto cpu_operator = make_matrix_free_operator<dim, degree>(
  matrix_free, elasticity);
```

### Stokes

Mixed forms use positional field identities: velocity is field 0 and pressure
is field 1. Field lists are sorted by index, independent of expression order.
The expression determines whether each field needs values or gradients:

```cpp
auto [u, p] = trial_functions<ValueShape::vector, ValueShape::scalar>();
auto [v, q] = test_functions<ValueShape::vector, ValueShape::scalar>();
auto mu = coefficient(1.7);

auto stokes =
  integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))), dx)
  - integral(div(v) * p, dx)
  - integral(q * div(u), dx);
```

The current Stokes lowering uses separate velocity and pressure DoFHandlers
and block vectors. CPU and Portable operators use the same form. `FormFields`
and `FieldRequirements` expose expression-derived compile-time metadata for
field lists and value/gradient requirements. The former Stokes executable is
now a unit test in `tests/test_stokes.cc`. A second test interpolates the
manufactured solution derived from the stream function
`x²(1-x)² y²(1-y)²`, with pressure `x + 2y - 1.5`, using
`VectorTools::interpolate()`. Both backends are checked against independently
integrated analytical forcing using `FEValues`. Run the Stokes tests on two
MPI ranks with:

```sh
mpiexec -n 2 ./build/pmf_form_tests "[stokes]"
```

## Current scope

The implemented matrix-free subset is scalar Laplace and Helmholtz forms,
including independent constant diffusion and reaction coefficients,
the supported isotropic elasticity patterns, and a two-field Stokes form.
The `coefficient(value)` factory currently accepts arithmetic constants and
stores them by value. A future provider-based overload such as
`coefficient(mu_function)` can use ordinary deal.II/C++ objects, but spatial
evaluation, provider lifetimes, device access, and general kernel lowering
are not implemented yet. The current mixed lowering supports the Stokes form
above.
Assembled `FEValues` operators and boundary and face terms are future work.
See [plan.md](plan.md) for the roadmap and
status.

## Build and test

This project expects a configured deal.II build. Use the existing `build/`
directory:

```sh
cmake --build build
ctest --test-dir build --output-on-failure
```

The elasticity executable applies a form with embedded Lamé constants using both
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
