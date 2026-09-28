# deal.II Form Language

A small C++ expression-template prototype for describing finite-element forms
and converting supported forms into deal.II matrix-free operators. The same
form can drive either `dealii::MatrixFree` with `FEEvaluation`, or
`dealii::Portable::MatrixFree` with `Portable::FEEvaluation`.

The goal is to keep the mathematical expression independent of the execution
backend. A form is passed to `make_matrix_free_operator` or
`make_portable_matrix_free_operator`. Both use the same compile-time lowering:
trial fields supply values and gradients, and the expression's adjoint
operations accumulate test-field values and gradients for submission.
Field shapes and evaluation/integration flags are derived from the form.

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
Explicit operator aliases also accept `decltype(form)`, including const and
reference-qualified types.

Call `op.get_diagonal()` to obtain a const reference to its diagonal
vector (not the inverse). The first call computes and caches it; later calls
reuse the cache. This works on const CPU and Portable operators. Call
`compute_diagonal()` to force recomputation after changing the underlying
MatrixFree data. Computation is collective over the operator's MPI communicator,
so all ranks must call consistently. Constrained entries are one; unconstrained
Stokes pressure entries are zero. Ghost entries are not updated.

Operators also provide `get_inverse_diagonal()`, which lazily caches
the reciprocal diagonal for `PreconditionJacobi`. It requires nonzero diagonal
entries and supports const operators on both backends. `compute_diagonal()`
invalidates this cache as well. Ghost entries are not updated. The unstabilized
Stokes form has zero pressure diagonal entries and cannot use this inverse.

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

The Stokes operators use separate velocity and pressure DoFHandlers
and block vectors. CPU and Portable operators use the same form. `FormFields`
and `FieldRequirements` expose expression-derived compile-time metadata for
field lists and value/gradient requirements.

The tests in `tests/test_stokes.cc` check agreement between the two backends
and correctness against a manufactured solution. The manufactured-solution
test interpolates the velocity derived from the stream function
`x²(1-x)² y²(1-y)²`, with pressure `x + 2y - 1.5`, using
`VectorTools::interpolate()`. Both backends are checked against independently
integrated analytical forcing using `FEValues`. Run the Stokes tests on two
MPI ranks with:

```sh
mpiexec -n 2 ./build/pmf_form_tests "[stokes]"
```

## Supported forms

The generic evaluator handles bilinear cell forms built from scalar/vector
field values, field gradients and divergences, symmetrization, inner products,
scalar multiplication, and sums/differences of expressions and integrals.
Every term must be homogeneous of degree one in trial fields and one in test
fields; nonlinear or affine terms are rejected at compile time. Coefficients
and literals are arithmetic constants stored by value.

Operators support two and three spatial dimensions, with one DoFHandler per
field. Supply a single polynomial degree for uniform-degree fields, or one
degree per field, for example `make_matrix_free_operator<3, 2, 1>(data, form)`
for Q2 velocity and Q1 pressure. The portable factory uses the same syntax.
All fields share a quadrature rule with one more point per direction than
the largest field degree. Trial and test
fields must have matching shapes and contiguous indices starting at zero.
Multiple fields use distributed block vectors. Gradients and divergences
apply directly to field symbols, not to composite expressions.

Spatially varying coefficient providers, assembled `FEValues` operators, and
boundary and face terms are not supported.
See [plan.md](plan.md) for the roadmap and status.

## Build and test

This project expects a configured deal.II build. Use the existing `build/`
directory:

```sh
cmake --build build
ctest --test-dir build --output-on-failure
```

### Stokes operator throughput

`stokes/stokes.cc` benchmarks operator application only: no solver, multigrid,
diagonal computation, or assembly is timed. It matches `step-104.cc`:
the 3D unit cube, global refinement starting at level 2, separate Q2 vector
velocity and Q1 scalar pressure spaces, QGauss(3), MappingQ(1), zero velocity
boundary constraints, and unconstrained pressure. Its form uses the full
velocity gradient, `inner(grad(v), grad(u)) - div(v)*p - q*div(u)`, not the
symmetric-gradient viscosity form in the Stokes example above.

```sh
cmake --build build --target stokes
mpiexec -n 1 ./build/stokes 4 100 7
```

Arguments are the maximum global refinement level (default 4), applications
per sample (100), and samples (5). Levels 2 through the requested maximum are
run. The `stokes` target uses Release compilation and the Release deal.II
library even when the other targets use Debug.

Both backends are checked against hand-written reference operators before
timing. The Portable reference implements the Stokes cell operation and
`vmult` from `step-104.cc`; the CPU reference adapts that operation to
`FEEvaluation`. Each generic/reference pair shares its MatrixFree data,
vectors, constraints, and compiler settings.

The benchmark reports median seconds per `vmult`, DoFs/s, MDoFs/s, sample
ranges, and generic/reference time ratios (1 means equal throughput).
Timing includes vector zeroing, communication, and constrained-row handling,
but excludes setup, correctness checks, and host/device transfers. Two warmups
precede repeated samples; ordering alternates, device work is fenced, and the
slowest MPI rank determines each sample time. DoF counts include velocity and
pressure, including constrained entries, as in step-104.

Use these paired timings to assess overhead rather than comparing warmed
samples directly with step-104's single cold application. Interpret small
differences relative to sample variability. The printed execution space
identifies whether Portable runs on a CPU or a GPU.

The elasticity executable applies a form with embedded Lamé constants using both
backends, compares the resulting distributed vectors, and exits with an error
if their relative difference exceeds the tolerance. For example, run it on two
MPI ranks with:

```sh
mpiexec -n 2 ./build/elasticity
```

The Laplace example in `laplace/laplace.cc` solves `-2.5 Δu = f` with
homogeneous Dirichlet conditions and exact solution `u = x(1-x)y(1-y)`.
It uses quadratic elements and CG with Jacobi preconditioning on both CPU
and Portable backends. It reports iteration counts, relative residuals,
solution errors, and the difference between backends, returning a nonzero
exit code if the accuracy checks fail. Run it with:

```sh
mpiexec -n 2 ./build/laplace
```

The manufactured-solution test in `tests/test_laplace.cc` solves
`-2.5 Δu = f` with homogeneous Dirichlet
conditions and exact solution `u = x(1-x)y(1-y)`, using quadratic elements.
Both CPU and Portable operators use `SolverCG` with `PreconditionJacobi`,
backed by the cached inverse diagonal. The test independently assembles the
forcing, checks convergence and the true residual, and compares the solutions with
`VectorTools::interpolate()` and with each other. Run it on two MPI ranks with:

```sh
mpiexec -n 2 ./build/pmf_form_tests "[laplace][solve]"
```

`tests/test_generic_lowering.cc` checks scalar Helmholtz, vector elasticity,
and mixed Stokes actions and diagonals against independent `FEValues`
integration on both backends. Variants exercise reordered operands, coefficient
sums, different coupling signs, and additional mixed-system mass terms.
Run these tests with:

```sh
mpiexec -n 2 ./build/pmf_form_tests "[lowering]"
```

Catch2 is fetched by CMake into the build tree. Format project C++ files with:

```sh
cmake --build build --target indent
```

See [AGENTS.md](AGENTS.md) for development instructions and
[`reference/`](reference/) for deal.II example implementations.
