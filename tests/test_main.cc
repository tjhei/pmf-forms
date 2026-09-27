#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <elasticity_matrix_free.h>
#include <forms.h>

#include <type_traits>

using namespace pmf::forms;

int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  Catch::Session                           session;
  return session.run(argc, argv);
}

TEST_CASE("Stokes form is represented by expression-template types",
          "[forms][stokes]")
{
  struct VelocityTag
  {};
  struct PressureTag
  {};
  struct TestVelocityTag
  {};
  struct TestPressureTag
  {};
  struct ViscosityTag
  {};

  using namespace expression_templates;

  const auto u  = trial<VelocityTag, ValueShape::vector>();
  const auto p  = trial<PressureTag, ValueShape::scalar>();
  const auto v  = test<TestVelocityTag, ValueShape::vector>();
  const auto q  = test<TestPressureTag, ValueShape::scalar>();
  const auto mu = coefficient<ViscosityTag>();

  const auto stokes =
    integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))), dx) -
    integral(div(v) * p, dx) - integral(q * div(u), dx);

  STATIC_REQUIRE(sizeof(stokes) > 0);
  static_assert(decltype(grad(u))::shape == ValueShape::tensor);
  static_assert(decltype(div(u))::shape == ValueShape::scalar);
}

TEST_CASE("Expression-template forms retain their structure in types",
          "[forms][expression-templates]")
{
  struct DisplacementTag
  {};
  struct TestFunctionTag
  {};
  struct LambdaTag
  {};
  struct MuTag
  {};

  using namespace expression_templates;

  const auto u      = trial<DisplacementTag, ValueShape::vector>();
  const auto v      = test<TestFunctionTag, ValueShape::vector>();
  const auto lambda = coefficient<LambdaTag>();
  const auto mu     = coefficient<MuTag>();

  const auto elasticity =
    integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))), dx) +
    integral(lambda * div(v) * div(u), dx);

  static_assert(decltype(grad(u))::shape == ValueShape::tensor);
  static_assert(decltype(sym(grad(v)))::shape == ValueShape::symmetric_tensor);
  static_assert(decltype(inner(sym(grad(v)), sym(grad(u))))::shape ==
                ValueShape::scalar);
  STATIC_REQUIRE(sizeof(elasticity) > 0);

  const auto coefficients = elasticity_coefficients<LambdaTag, MuTag>(3.0, 2.0);
  dealii::Tensor<2, 2, double> gradient;
  gradient[0][0] = 1.0;
  gradient[0][1] = 2.0;
  gradient[1][0] = 4.0;
  gradient[1][1] = 5.0;
  dealii::Tensor<2, 2, double> stress;
  stress = 0.0;
  expression_templates::internal::apply_form(elasticity,
                                             gradient,
                                             stress,
                                             coefficients);

  REQUIRE(stress[0][0] == Catch::Approx(22.0));
  REQUIRE(stress[0][1] == Catch::Approx(12.0));
  REQUIRE(stress[1][0] == Catch::Approx(12.0));
  REQUIRE(stress[1][1] == Catch::Approx(38.0));
}

TEST_CASE("Scalar Laplace forms accept optional tagged coefficients",
          "[forms][laplace]")
{
  struct ScalarTag
  {};
  struct TestTag
  {};
  struct CoefficientTag
  {};
  struct ReactionTag
  {};
  using namespace expression_templates;

  const auto u             = trial<ScalarTag, ValueShape::scalar>();
  const auto v             = test<TestTag, ValueShape::scalar>();
  const auto unit_form     = integral(inner(grad(v), grad(u)), dx);
  const auto alpha         = coefficient<CoefficientTag>();
  const auto weighted_form = integral(alpha * inner(grad(v), grad(u)), dx);
  const auto binding       = bind_coefficient<CoefficientTag>(3.0);

  dealii::Tensor<1, 2, double> gradient;
  gradient[0] = 2.0;
  gradient[1] = -1.0;
  dealii::Tensor<1, 2, double> flux;
  flux = 0.0;

  internal::apply_scalar_form(unit_form, gradient, flux, NoCoefficients{}, 1.0);
  REQUIRE(flux[0] == Catch::Approx(2.0));
  REQUIRE(flux[1] == Catch::Approx(-1.0));

  flux = 0.0;
  internal::apply_scalar_form(weighted_form, gradient, flux, binding, 1.0);
  REQUIRE(flux[0] == Catch::Approx(6.0));
  REQUIRE(flux[1] == Catch::Approx(-3.0));

  const auto beta = coefficient<ReactionTag>();
  const auto weighted_helmholtz =
    integral(alpha * inner(grad(v), grad(u)) + beta * (v * u), dx);
  const auto bindings = bind_coefficients(bind_coefficient<CoefficientTag>(2.5),
                                          bind_coefficient<ReactionTag>(4.0));
  flux                = 0.0;
  double submitted_value = 0.0;
  internal::apply_scalar_form(
    weighted_helmholtz, 6.0, gradient, submitted_value, flux, bindings, 1.0);
  REQUIRE(flux[0] == Catch::Approx(5.0));
  REQUIRE(flux[1] == Catch::Approx(-2.5));
  REQUIRE(submitted_value == Catch::Approx(24.0));

  const auto literal_helmholtz =
    integral(2.0 * inner(grad(v), grad(u)) + v * (3.0 * u), dx);
  flux            = 0.0;
  submitted_value = 0.0;
  internal::apply_scalar_form(literal_helmholtz,
                              6.0,
                              gradient,
                              submitted_value,
                              flux,
                              NoCoefficients{},
                              1.0);
  REQUIRE(flux[0] == Catch::Approx(4.0));
  REQUIRE(flux[1] == Catch::Approx(-2.0));
  REQUIRE(submitted_value == Catch::Approx(18.0));
}

TEST_CASE("One expression-template form drives both MatrixFree backends",
          "[forms][matrix-free]")
{
  struct DisplacementTag
  {};
  struct TestFunctionTag
  {};
  struct LambdaTag
  {};
  struct MuTag
  {};

  using namespace expression_templates;
  constexpr int dim       = 2;
  constexpr int fe_degree = 1;

  const auto u    = trial<DisplacementTag, ValueShape::vector>();
  const auto v    = test<TestFunctionTag, ValueShape::vector>();
  const auto form = integral(inner(sym(grad(v)), sym(grad(u))), dx);

  dealii::parallel::distributed::Triangulation<dim> triangulation(
    MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);

  dealii::FESystem<dim>   finite_element(dealii::FE_Q<dim>(fe_degree), dim);
  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(finite_element);

  dealii::AffineConstraints<double> constraints;
  constraints.reinit(dof_handler.locally_owned_dofs(),
                     dealii::DoFTools::extract_locally_relevant_dofs(
                       dof_handler));
  constraints.close();

  dealii::MappingQ<dim>   mapping(1);
  const dealii::QGauss<1> quadrature(fe_degree + 1);

  auto cpu_data = std::make_shared<dealii::MatrixFree<dim, double>>();
  dealii::MatrixFree<dim, double>::AdditionalData cpu_additional_data;
  cpu_additional_data.mapping_update_flags =
    dealii::update_gradients | dealii::update_JxW_values;
  cpu_data->reinit(
    mapping, dof_handler, constraints, quadrature, cpu_additional_data);

  using CpuOperator =
    MatrixFreeOperator<dim,
                       fe_degree,
                       typename std::decay<decltype(form)>::type>;
  const CpuOperator            cpu_operator(cpu_data, form);
  typename CpuOperator::Vector cpu_source;
  typename CpuOperator::Vector cpu_destination;
  cpu_operator.initialize_dof_vector(cpu_source);
  cpu_operator.initialize_dof_vector(cpu_destination);
  cpu_source = 0.0;
  for (unsigned int i = 0; i < cpu_source.locally_owned_size(); ++i)
    cpu_source.local_element(i) = static_cast<double>(i % 9 + 1);
  cpu_source.update_ghost_values();
  cpu_operator.vmult(cpu_destination, cpu_source);

  auto portable_data =
    std::make_shared<dealii::Portable::MatrixFree<dim, double>>();
  dealii::Portable::MatrixFree<dim, double>::AdditionalData
    portable_additional_data;
  portable_additional_data.mapping_update_flags =
    dealii::update_gradients | dealii::update_JxW_values;
  portable_data->reinit(
    mapping, dof_handler, constraints, quadrature, portable_additional_data);

  using PortableOperator =
    PortableMatrixFreeOperator<dim,
                               fe_degree,
                               typename std::decay<decltype(form)>::type>;
  const PortableOperator            portable_operator(portable_data, form);
  typename PortableOperator::Vector portable_source;
  typename PortableOperator::Vector portable_destination;
  portable_operator.initialize_dof_vector(portable_source);
  portable_operator.initialize_dof_vector(portable_destination);

  using HostVector =
    dealii::LinearAlgebra::distributed::Vector<double,
                                               dealii::MemorySpace::Host>;
  HostVector host_source;
  portable_data->initialize_dof_vector(host_source, 0);
  host_source = 0.0;
  for (unsigned int i = 0; i < host_source.locally_owned_size(); ++i)
    host_source.local_element(i) = static_cast<double>(i % 9 + 1);
  host_source.compress(dealii::VectorOperation::insert);
  portable_source.import_elements(host_source, dealii::VectorOperation::insert);
  portable_operator.vmult(portable_destination, portable_source);

  REQUIRE(cpu_destination.l2_norm() > 0.0);
  REQUIRE(portable_destination.l2_norm() ==
          Catch::Approx(cpu_destination.l2_norm()).epsilon(1e-10));
}

TEST_CASE("Scalar Laplace form drives CPU and Portable MatrixFree",
          "[forms][matrix-free][laplace]")
{
  struct UTag
  {};
  struct VTag
  {};
  struct AlphaTag
  {};
  struct BetaTag
  {};
  using namespace expression_templates;
  constexpr int dim    = 2;
  constexpr int degree = 1;

  const auto u     = trial<UTag, ValueShape::scalar>();
  const auto v     = test<VTag, ValueShape::scalar>();
  const auto alpha = coefficient<AlphaTag>();
  const auto beta  = coefficient<BetaTag>();
  const auto form =
    integral(alpha * inner(grad(v), grad(u)) + beta * (v * u), dx);
  const auto binding = bind_coefficients(bind_coefficient<AlphaTag>(2.5),
                                         bind_coefficient<BetaTag>(1.75));

  dealii::parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  dealii::FE_Q<dim>       fe(degree);
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  dealii::AffineConstraints<double> constraints;
  constraints.reinit(dof_handler.locally_owned_dofs(),
                     dealii::DoFTools::extract_locally_relevant_dofs(
                       dof_handler));
  constraints.close();
  dealii::MappingQ<dim>   mapping(1);
  const dealii::QGauss<1> quadrature(degree + 1);

  auto cpu_data = std::make_shared<dealii::MatrixFree<dim, double>>();
  dealii::MatrixFree<dim, double>::AdditionalData cpu_additional_data;
  cpu_additional_data.mapping_update_flags =
    dealii::update_gradients | dealii::update_JxW_values;
  cpu_data->reinit(
    mapping, dof_handler, constraints, quadrature, cpu_additional_data);
  using FormType    = typename std::decay<decltype(form)>::type;
  using BindingType = typename std::decay<decltype(binding)>::type;
  MatrixFreeOperator<dim, degree, FormType, BindingType> cpu_op(cpu_data,
                                                                form,
                                                                binding);
  dealii::LinearAlgebra::distributed::Vector<double>     cpu_src, cpu_dst;
  cpu_op.initialize_dof_vector(cpu_src);
  cpu_op.initialize_dof_vector(cpu_dst);
  cpu_src = 0.0;
  for (unsigned int i = 0; i < cpu_src.locally_owned_size(); ++i)
    cpu_src.local_element(i) = static_cast<double>(i + 1);
  cpu_src.update_ghost_values();
  cpu_op.vmult(cpu_dst, cpu_src);

  const auto unit_form = integral(inner(grad(v), grad(u)) + v * u, dx);
  using UnitFormType   = typename std::decay<decltype(unit_form)>::type;
  MatrixFreeOperator<dim, degree, UnitFormType>      cpu_unit_op(cpu_data,
                                                            unit_form);
  dealii::LinearAlgebra::distributed::Vector<double> cpu_unit_dst;
  cpu_unit_op.initialize_dof_vector(cpu_unit_dst);
  cpu_unit_op.vmult(cpu_unit_dst, cpu_src);

  auto portable_data =
    std::make_shared<dealii::Portable::MatrixFree<dim, double>>();
  dealii::Portable::MatrixFree<dim, double>::AdditionalData
    portable_additional_data;
  portable_additional_data.mapping_update_flags =
    dealii::update_gradients | dealii::update_JxW_values;
  portable_data->reinit(
    mapping, dof_handler, constraints, quadrature, portable_additional_data);
  PortableMatrixFreeOperator<dim, degree, FormType, BindingType> portable_op(
    portable_data, form, binding);
  typename PortableMatrixFreeOperator<dim, degree, FormType, BindingType>::
    Vector portable_src,
    portable_dst;
  portable_op.initialize_dof_vector(portable_src);
  portable_op.initialize_dof_vector(portable_dst);
  using HostVector =
    dealii::LinearAlgebra::distributed::Vector<double,
                                               dealii::MemorySpace::Host>;
  HostVector host_src;
  portable_data->initialize_dof_vector(host_src, 0);
  host_src = 0.0;
  for (unsigned int i = 0; i < host_src.locally_owned_size(); ++i)
    host_src.local_element(i) = static_cast<double>(i + 1);
  host_src.compress(dealii::VectorOperation::insert);
  portable_src.import_elements(host_src, dealii::VectorOperation::insert);
  portable_op.vmult(portable_dst, portable_src);

  PortableMatrixFreeOperator<dim, degree, UnitFormType> portable_unit_op(
    portable_data, unit_form);
  typename PortableMatrixFreeOperator<dim, degree, UnitFormType>::Vector
    portable_unit_src,
    portable_unit_dst;
  portable_unit_op.initialize_dof_vector(portable_unit_src);
  portable_unit_op.initialize_dof_vector(portable_unit_dst);
  portable_unit_src.import_elements(host_src, dealii::VectorOperation::insert);
  portable_unit_op.vmult(portable_unit_dst, portable_unit_src);

  REQUIRE(cpu_dst.l2_norm() > 0.0);
  REQUIRE(cpu_unit_dst.l2_norm() > 0.0);
  REQUIRE(portable_dst.l2_norm() ==
          Catch::Approx(cpu_dst.l2_norm()).epsilon(1e-10));
  REQUIRE(portable_unit_dst.l2_norm() ==
          Catch::Approx(cpu_unit_dst.l2_norm()).epsilon(1e-10));
}
