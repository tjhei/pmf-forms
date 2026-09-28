# deal.II Form Language: Design and Implementation Plan

## Current implementation status

The current prototype uses C++ expression templates and has CPU and Portable
MatrixFree operators for bilinear scalar, vector, and mixed cell forms. The
public `MatrixFreeOperator` and `PortableMatrixFreeOperator` aliases derive
field evaluations and submissions through shared compile-time lowering.

Implemented and build-checked:

- Expression-template symbols, differential operators, algebra, and cell
  integrals.
- Tag-free default `trial()` / `test()` symbols for single-field forms, with
  positional numbering via `trial_functions<Shapes...>()` and
  `test_functions<Shapes...>()` for multi-field forms.
- Scalar Laplace and Helmholtz forms, with embedded constant
  diffusion and reaction coefficients via `coefficient(value)`, or literal
  constants in the form. No separate bindings are required.
- Unit-weight symmetric-gradient elasticity and the current constant Lamé
  elasticity pattern.
- CPU `dealii::MatrixFree` and `dealii::Portable::MatrixFree` execution for
  the supported forms, using the same form description.
- Cached `get_diagonal()` for scalar, elasticity, and Stokes operators on both
  backends, checked against basis-vector applications with constrained rows.
- A runnable MPI elasticity example that applies both backends to the same
  distributed vector and checks their relative error.
- A manufactured-solution Laplace test with embedded diffusion coefficient,
  CG and Jacobi solves on both backends, and serial/two-rank MPI coverage.
- Compile-time trial/test field lists, coefficient occurrence types, and value/gradient
  requirements derived from the expression tree.
- A two-field Stokes operator using separate velocity and pressure DoFHandlers
  and block vectors on CPU and Portable MatrixFree, with MPI unit tests for
  backend agreement and an interpolated manufactured solution.
- Per-field polynomial degrees with a shared quadrature rule, including Q2-Q1.
- A Release Stokes operator benchmark on the globally refined 3D unit cube,
  comparing generic CPU and Portable application against the hand-written
  step-104 operation, with correctness checks and repeated DoFs/s timings.
- Catch2 unit and backend comparison cases.
- Generic recursive trial evaluation and adjoint test submission for cell
  forms, including expression-derived per-field flags and diagonal blocks.
- Independent FEValues action and diagonal checks for scalar Helmholtz,
  vector elasticity, and mixed Stokes, including algebraic rearrangements
  and mixed forms with changed coupling signs and additional mass terms.
- Compile-time checks for bilinearity, matching trial/test shapes, and
  contiguous field indices.

Still incomplete:

- Variable coefficient fields and general coefficient binding to FE data.
- Differential operators on composite expressions and different quadrature
  rules for different fields.
- `FEValues` assembly from the same form.
- Validation of backend capabilities beyond the supported cell-form subset.
- GPU hardware validation and performance comparison against hand-written
  kernels.

The runtime expression tree originally considered below is postponed. The
first implementation path is static expression templates and MatrixFree
execution.

## Goal

The immediate goal is deliberately narrow:

> Express representative Stokes and isotropic-elasticity forms once, and use the same form description to build `dealii::MatrixFree` and `Portable::MatrixFree` operators. Derive mixed field and evaluation metadata from the expression at compile time.

The prototype should establish that a higher-level form description can eventually achieve essentially the same CPU and GPU performance as hand-written deal.II matrix-free kernels.

We explicitly postpone:

- Python bindings
- JIT compilation
- code generation
- automatic differentiation
- nonlinear forms
- DG and face terms
- general anisotropic fourth-order tensors
- a complete form language

The initial implementation may be slow or dynamic in places, provided the design does not prevent a later zero-overhead implementation.

---

## 1. Reference forms

The first prototype should support two representative problems.

### Stokes

Conceptually,

```cpp
auto a =
  integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u)))
           - div(v) * p
           - q * div(u),
           dx);
```

where

- `u` is the velocity trial function,
- `v` is the velocity test function,
- `p` is the pressure trial function,
- `q` is the pressure test function,
- `mu` may be constant or spatially varying.

This exercises:

- scalar and vector fields,
- mixed systems,
- multiple test/trial arguments,
- gradients,
- symmetric gradients,
- divergence,
- nontrivial tensor operations,
- coefficients,
- coupled blocks.

### Linear elasticity

For isotropic elasticity,

```cpp
auto a =
  integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u)))
           + lambda * div(v) * div(u),
           dx);
```

where `lambda` and `mu` may be constants or spatially varying coefficients.

This exercises:

- vector-valued spaces,
- symmetric tensors,
- scalar coefficients,
- tensor contractions,
- variable material parameters.

General anisotropic elasticity,

```cpp
integral(inner(sym(grad(v)),
               C * sym(grad(u))),
         dx);
```

should remain a future requirement, but need not be implemented in the first prototype.

---

## 2. Minimal mathematical expression language

Only implement operations required by the two reference forms.

### Symbols

```text
trial
test
coefficient
```

The representation must distinguish formal test/trial arguments from coefficient or state fields.

### Differential operators

```text
grad
div
sym
```

### Algebra

```text
+
-
*
inner
```

### Integration

```text
integral(expression, dx)
```

For now, only cell integrals are required.

### Value shapes

Initially support:

```text
scalar
vector
rank-2 tensor
symmetric rank-2 tensor
```

Expressions should carry enough shape information to reject invalid operations.

For example,

```text
inner(vector, vector)              -> scalar
inner(symtensor, symtensor)        -> scalar
grad(scalar)                       -> vector
grad(vector)                       -> rank-2 tensor
sym(grad(vector))                  -> symmetric rank-2 tensor
div(vector)                        -> scalar
```

---

## 3. Forms, arity, and linearity

A form's **arity** is determined by its formal arguments.

For example:

```text
0-form: no formal test/trial arguments
1-form: one formal argument
2-form: two formal arguments
```

A valid `n`-form must be multilinear in its `n` formal arguments.

It may have arbitrary nonlinear dependence on coefficients or state fields.

For example,

```text
u^2 * v
```

is nonlinear in the state `u` but linear in the formal argument `v`, and can therefore occur in a 1-form.

The form representation should eventually be able to report information such as

```text
arity = 2

argument v:
    linear

argument u:
    linear

coefficient mu:
    arbitrary dependence
```

and reject something such as

```text
v * v
```

when `v` is supposed to be a formal argument of a 1-form.

---

## 4. Form algebra

Keep form-level algebra deliberately restrictive initially.

Expressions may be combined normally:

```cpp
auto e =
    2.0 * inner(sym(grad(v)), sym(grad(u)))
  + lambda * div(v) * div(u);
```

An integral is created explicitly:

```cpp
integral(e, dx)
```

Forms may be added or subtracted:

```cpp
Form a =
    integral(e1, dx)
  + integral(e2, dx);
```

Initially, do not support potentially ambiguous operations such as

```cpp
2.0 * form
form + 2.0
coefficient * form
```

A factor can always be placed unambiguously inside the integral.

The distinction between

```text
expression
integral
form
```

should remain conceptually clear even if `integral()` directly returns a `Form` in the public API.

---

## 5. Separate form, discretization, and execution

The architecture should have three distinct layers:

```text
Mathematical Form
       |
       v
Discrete Binding
       |
       v
Execution Backend
```

### Mathematical form

Describes expressions such as

```text
integral(inner(grad(v), grad(u)), dx)
```

without depending on `FEValues`, `FEEvaluation`, matrices, or a particular finite element.

### Discrete binding

Connects formal symbols to deal.II objects:

```text
DoFHandler
FiniteElement
component/extractor
Mapping
domain
coefficients
```

deal.II remains responsible for defining the actual finite-element spaces.

The form language should not create a second finite-element hierarchy.

### Execution

Possible execution modes include

```text
compute an integral
assemble a vector
assemble a matrix
MatrixFree operator
Portable::MatrixFree operator
```

The same mathematical form should be usable by multiple execution backends.

---

## 6. Function spaces and capabilities

Do not initially encode mathematical spaces such as

```text
H1
L2
H(div)
H(curl)
```

as fundamental identities in the symbolic IR.

Many different deal.II finite elements ultimately provide the same conformity, and DG formulations may use element-wise derivatives even when the global space is not H1 conforming.

Instead, expressions impose requirements such as

```text
value
gradient
divergence
trace
normal trace
```

and the discrete binding determines whether the selected deal.II space supports the required operations.

Conformity information such as H1/H(div)/H(curl) can still be exposed as metadata or used for validation where appropriate.

---

## 7. Dimensions and codimensions

Dimensions belong primarily to domains and discretizations, rather than every symbolic expression.

deal.II distinguishes

```text
dim       = topological dimension
spacedim  = embedding-space dimension
```

For example, a surface embedded in 3D has

```text
dim      = 2
spacedim = 3
```

The integration measure determines the integration dimension:

```text
dx  -> dim
ds  -> dim - 1
dS  -> dim - 1
```

Codimension is derived rather than fundamental:

```text
codim = spacedim - dim
```

The dynamic mathematical IR may store dimensions as runtime metadata.

The high-performance backend may specialize them at compile time:

```cpp
Backend<dim, spacedim>
```

The design should eventually permit forms coupling domains of different dimensions, so dimensions should conceptually belong to domains rather than globally to the entire form.

---

## 8. Backend-independent quadrature kernel

This is the critical intermediate abstraction.

The mathematical form should lower to a small vocabulary describing operations at quadrature points.

For the initial prototype, the kernel language needs approximately:

### Inputs

```text
get_value(field)
get_gradient(field)
get_coefficient_value(coefficient)
```

### Arithmetic

```text
add
subtract
multiply
```

### Tensor operations

```text
symmetrize
trace
inner / contract
```

### Outputs

```text
submit_value(field, value)
submit_gradient(field, tensor)
```

This kernel representation should contain no direct references to

```text
FEValues
FEEvaluation
Portable::FEEvaluation
```

Those belong to execution backends.

---

## 9. Example lowering: elasticity

The mathematical form

```cpp
integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u)))
         + lambda * div(v) * div(u),
         dx);
```

should lower conceptually to

```text
grad_u = get_gradient(u)

eps = symmetrize(grad_u)

sigma =
    2 * mu * eps
    + lambda * trace(eps) * I

submit_gradient(v, sigma)
```

A `MatrixFree` backend should then produce code essentially equivalent to hand-written deal.II code:

```cpp
const auto grad_u = phi.get_gradient(q);
const auto eps    = symmetrize(grad_u);

const auto sigma =
    2.0 * mu * eps
    + lambda * trace(eps) * identity;

phi.submit_gradient(sigma, q);
```

The abstraction should disappear after compilation.

---

## 10. Example lowering: Stokes

The form

```cpp
integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u)))
         - div(v) * p
         - q * div(u),
         dx);
```

should lower approximately to

```text
grad_u = get_gradient(u)
p_q    = get_value(p)

stress =
    2 * mu * symmetrize(grad_u)
    - p_q * I

submit_gradient(v, stress)

div_u = trace(grad_u)

submit_value(q, -div_u)
```

The exact lowering may later use specialized operations such as divergence submission if that produces better deal.II kernels.

The kernel IR should describe mathematical data flow without prematurely committing to a particular `FEEvaluation` method.

---

## 11. Runtime mathematical IR (postponed)

This runtime DAG/arena proposal is deferred. The current implementation path
uses static C++ expression-template types directly; revisit a dynamic IR after
the supported MatrixFree kernels and their lowering needs are better
understood.

Prototype the mathematical form as an inspectable runtime DAG or arena.

For example:

```text
Integral(dx)
  |
  Add
  |-- Multiply
  |    |-- Multiply
  |    |    |-- 2
  |    |    `-- mu
  |    `-- Inner
  |         |-- Sym
  |         |    `-- Grad(v)
  |         `-- Sym
  |              `-- Grad(u)
  |
  `-- Multiply
       |-- lambda
       `-- Multiply
            |-- Div(v)
            `-- Div(u)
```

The public dynamic types can initially be simple:

```cpp
class Expression;
class Form;
```

An `Expression` can be a cheap handle into an arena:

```cpp
class Expression
{
    FormContext *context;
    NodeId       node;
};
```

The runtime IR initially only needs to:

1. construct the reference forms;
2. validate tensor/value shapes;
3. identify formal arguments;
4. determine form arity;
5. verify multilinearity;
6. print/introspect the form;
7. lower to the quadrature-kernel representation.

Do not optimize the IR initially.

---

## 12. Static versus dynamic representation

The prototype has selected static C++ expression templates for its initial
high-performance path. A dynamic IR remains a possible later addition for
inspection, runtime construction, and transformations; it is not a prerequisite
for current work.

Do not prematurely require the runtime IR to also be the zero-overhead representation.

There are two distinct needs.

### Dynamic representation

Useful for

```text
inspection
symbolic transformations
Python bindings
runtime construction
serialization
debugging
```

### Static representation

Useful for

```text
maximum CPU performance
MatrixFree
Portable::MatrixFree
GPU kernels
arbitrary C++ integration
```

The eventual static C++ syntax may look nearly identical:

```cpp
auto u = trial(...);
auto v = test(...);

auto a =
  integral(inner(grad(v), grad(u)), dx);
```

but internally may use expression-template types rather than runtime DAG nodes.

Alternatively, the runtime IR may eventually generate or JIT-compile static kernels.

Do not choose between these approaches until the minimal kernel language has been tested.

---

## 13. High-performance MatrixFree backends

Implement the high-performance path early.

The same lowered kernel semantics should have at least two implementations:

```text
Kernel
  |
  |-- dealii::MatrixFree / FEEvaluation
  |
  `-- Portable::MatrixFree / Portable::FEEvaluation
```

The resulting compiler-visible code should consist of ordinary straight-line operations such as

```cpp
get_gradient()
get_value()

tensor arithmetic

submit_gradient()
submit_value()
```

with no runtime AST traversal in the quadrature-point loop.

This is especially important for GPU execution.

A runtime AST interpreter may be acceptable as an initial implementation or Python execution mode, but it must not be the only possible execution strategy.

---

## 14. Assembled backend

Add an `FEValues`-based backend for the same mathematical forms.

The eventual user model should look conceptually like

```cpp
Form a = ...;

assemble(a, binding, matrix);

auto A =
    make_matrix_free_operator(a, binding);

auto A_gpu =
    make_portable_matrix_free_operator(a, binding);
```

All three objects must represent the same mathematical 2-form.

The assembled matrix also provides an excellent correctness reference:

```text
matrix.vmult(dst, src)

vs.

matrix_free_operator.vmult(dst, src)
```

should agree to numerical precision.

---

## 15. Performance requirement

The prototype should explicitly compare generated/static kernels against expert hand-written deal.II kernels.

For Stokes and elasticity, inspect both:

```text
runtime performance
compiler-generated code
```

The goal is not merely "reasonably fast."

The goal is:

> The abstraction should permit essentially the same optimized CPU and GPU kernels that an expert would write manually using `FEEvaluation` or `Portable::FEEvaluation`.

If the architecture prevents this, revisit it before expanding the language.

---

## 16. Static templates versus code generation/JIT

This remains intentionally unresolved.

Two possible high-performance routes are:

### Static C++ forms

```text
C++ form expression
       |
template/static lowering
       |
MatrixFree / Portable kernel
```

Advantages:

- ordinary C++ compilation;
- arbitrary user C++ integrates naturally;
- compiler sees the complete kernel;
- natural GPU compilation.

Disadvantages:

- template complexity;
- compile times;
- symbolic transformations become harder;
- Python cannot create arbitrary new static types.

### Runtime IR plus code generation

```text
runtime Form IR
       |
symbolic transformations
       |
Kernel IR
       |
C++ / LLVM / other code generation
       |
compiled CPU/GPU kernel
```

Advantages:

- one rich symbolic representation;
- natural Python support;
- easy differentiation and optimization;
- arbitrary runtime forms.

Disadvantages:

- requires compiler infrastructure;
- JIT caching/loading/toolchain issues;
- GPU compilation is more complicated;
- arbitrary user C++ becomes harder to integrate.

The prototype uses the static C++ path first and may add a dynamic IR later.

---

## 17. Implementation milestones and status

### Milestone 1: Expression templates — partial

Implement:

```text
trial
test
coefficient

grad
div
sym

+
-
*
inner

integral(..., dx)
```

Support scalar and vector arguments and scalar coefficients.

Stokes and elasticity forms can be constructed. Printing and full form
validation are not implemented.

### Milestone 2: Validation — pending

Implement:

```text
value-shape checking
argument identification
form arity
multilinearity checking
```

Invalid expressions should produce useful diagnostics.

### Milestone 3: Kernel lowering — implemented for bilinear cell forms

A shared recursive evaluator lowers trial-field expressions and propagates
adjoints to test-field value/gradient submissions. Scalar Helmholtz, vector
elasticity, and mixed Stokes all use this path on CPU and Portable backends.
Evaluation flags and diagonal blocks are derived from the expression rather
than from PDE-specific patterns. A runtime kernel IR is not implemented.

### Milestone 4: CPU MatrixFree — implemented for the current subset

Map the kernel operations to `FEEvaluation`.

The current subset has Catch2 coverage. A full comparison against hand-written
Stokes and elasticity implementations remains to be done.

### Milestone 5: Portable::MatrixFree — implemented for the current subset

Map the same kernel operations to the portable backend.

The tests compare CPU and Portable backend actions in the configured build.
Validation on GPU-capable hardware remains pending.

### Milestone 6: FEValues assembly — pending

Assemble matrices from the same forms.

Verify

```text
assembled matrix action
≈
MatrixFree action
≈
Portable::MatrixFree action
```

### Milestone 7: Architecture review — pending

Review whether the high-performance public path should primarily use

```text
static C++ expression templates
code generation/JIT
both
```

The decision should be based on concrete experience with the supported kernels
rather than speculation.

---

## 18. Definition of success

Before expanding the language, require all of the following.

### Expressiveness

Stokes and variable-coefficient isotropic elasticity can be written naturally and without backend-specific concepts.

### Multiple execution modes

The same form can produce:

```text
assembled matrix
CPU MatrixFree operator
Portable::MatrixFree operator
```

### Correctness

All execution modes produce equivalent operator actions.

### Performance

CPU and GPU matrix-free kernels are structurally and performance-wise comparable to hand-written deal.II implementations.

### Separation of concerns

The mathematical form does not know whether it will ultimately be executed through

```text
FEValues
MatrixFree
Portable::MatrixFree
```

### Extensibility

Nothing in the design prevents later support for:

```text
general tensor coefficients
anisotropic elasticity
nonlinear expressions
automatic differentiation / linearization
boundary integrals
DG jumps and averages
H(div)
H(curl)
mixed-dimensional problems
time-dependent forms
Python bindings
JIT/code generation
```

---

## 19. Next examples after the prototype

Once the initial milestone succeeds, extend the requirements suite rather than immediately generalizing everything.

Suggested order:

1. convection-diffusion;
2. general tensor-valued coefficients;
3. anisotropic elasticity;
4. nonlinear energy → residual → Jacobian;
5. DG with jumps and averages;
6. H(div) mixed Darcy;
7. H(curl) Maxwell;
8. codimension-one/surface PDE;
9. coupled/mixed-dimensional problems;
10. time-dependent forms.

These examples should collectively become the practical specification of the form language.

---

## Guiding principle

The form language should simplify deal.II, not replace it.

`FiniteElement`, `DoFHandler`, mappings, constraints, coefficient objects, `MatrixFree`, `Portable::MatrixFree`, and arbitrary user C++ should remain first-class deal.II concepts.

The form language provides a declarative description of local mathematical forms and a common route to different execution strategies.

The initial runtime implementation may be slow.

The architecture must not make the eventual high-performance implementation slow.
