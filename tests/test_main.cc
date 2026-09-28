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
#include <forms.h>
#include <matrix_free_operator.h>

#include <type_traits>

using namespace pmf::forms;

int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  Catch::Session                           session;
  return session.run(argc, argv);
}

TEST_CASE("Field tuples assign stable positional identities", "[forms]")
{
  using namespace expression_templates;

  const auto [first_trial, second_trial, unused_trial] =
    trial_functions<ValueShape::scalar,
                    ValueShape::scalar,
                    ValueShape::vector>();
  const auto [first_test, second_test, unused_test] =
    test_functions<ValueShape::scalar,
                   ValueShape::scalar,
                   ValueShape::vector>();
  STATIC_REQUIRE(decltype(first_trial)::index == 0);
  STATIC_REQUIRE(decltype(second_trial)::index == 1);
  STATIC_REQUIRE(decltype(unused_trial)::index == 2);
  STATIC_REQUIRE(decltype(first_test)::index == 0);
  STATIC_REQUIRE(decltype(second_test)::index == 1);
  STATIC_REQUIRE(decltype(unused_test)::index == 2);
  STATIC_REQUIRE(decltype(unused_trial)::shape == ValueShape::vector);
  STATIC_REQUIRE(decltype(unused_test)::shape == ValueShape::vector);

  const auto form =
    integral(second_test * second_trial + first_test * first_trial +
               second_test * second_trial,
             dx);
  using Fields = FormFields<decltype(form)>;
  STATIC_REQUIRE(Fields::n_trial_fields == 2);
  STATIC_REQUIRE(Fields::n_test_fields == 2);
  STATIC_REQUIRE(std::is_same<typename Fields::trial_fields,
                              TypeList<Trial<0, ValueShape::scalar>,
                                       Trial<1, ValueShape::scalar>>>::value);
  STATIC_REQUIRE(std::is_same<typename Fields::test_fields,
                              TypeList<Test<0, ValueShape::scalar>,
                                       Test<1, ValueShape::scalar>>>::value);
  STATIC_REQUIRE(
    std::is_same<decltype(trial()),
                 typename std::decay<decltype(first_trial)>::type>::value);
  STATIC_REQUIRE(
    std::is_same<decltype(test()),
                 typename std::decay<decltype(first_test)>::type>::value);

  const auto sparse_form = integral(second_test * second_trial, dx);
  using SparseFields     = FormFields<decltype(sparse_form)>;
  STATIC_REQUIRE(std::is_same<typename SparseFields::trial_fields,
                              TypeList<Trial<1, ValueShape::scalar>>>::value);
  STATIC_REQUIRE(std::is_same<typename SparseFields::test_fields,
                              TypeList<Test<1, ValueShape::scalar>>>::value);
}

TEST_CASE("Stokes form is represented by expression-template types",
          "[forms][stokes]")
{
  using namespace expression_templates;

  const auto [u, p] = trial_functions<ValueShape::vector, ValueShape::scalar>();
  const auto [v, q] = test_functions<ValueShape::vector, ValueShape::scalar>();
  const auto mu     = coefficient(1.7);

  const auto stokes =
    integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))), dx) -
    integral(div(v) * p, dx) - integral(q * div(u), dx);

  using StokesForm   = typename std::decay<decltype(stokes)>::type;
  using StokesFields = FormFields<StokesForm>;
  STATIC_REQUIRE(StokesFields::n_trial_fields == 2);
  STATIC_REQUIRE(StokesFields::n_test_fields == 2);
  STATIC_REQUIRE(StokesFields::n_coefficients == 1);
  STATIC_REQUIRE(FormFields<decltype(stokes)>::n_trial_fields == 2);
  STATIC_REQUIRE(FormFields<decltype((stokes))>::n_test_fields == 2);
  STATIC_REQUIRE(std::is_same<MatrixFreeOperator<2, 1, decltype(stokes)>,
                              MatrixFreeOperator<2, 1, StokesForm>>::value);
  STATIC_REQUIRE(
    std::is_same<PortableMatrixFreeOperator<2, 1, decltype((stokes))>,
                 PortableMatrixFreeOperator<2, 1, StokesForm>>::value);
  const auto reordered =
    integral(q * div(u), dx) + integral(div(v) * p, dx) -
    integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))), dx);
  using ReorderedForm   = typename std::decay<decltype(reordered)>::type;
  using ReorderedFields = FormFields<ReorderedForm>;
  STATIC_REQUIRE(std::is_same<typename StokesFields::trial_fields,
                              typename ReorderedFields::trial_fields>::value);
  STATIC_REQUIRE(std::is_same<typename StokesFields::test_fields,
                              typename ReorderedFields::test_fields>::value);
  StokesQuadratureKernel<2, ReorderedForm> reordered_kernel(reordered);
  static_assert(internal::FieldRequirements<StokesForm, decltype(u)>::gradient,
                "velocity trial must require gradients");
  static_assert(internal::FieldRequirements<StokesForm, decltype(p)>::value,
                "pressure trial must require values");
  static_assert(internal::FieldRequirements<StokesForm, decltype(v)>::gradient,
                "velocity test must receive gradients");
  static_assert(internal::FieldRequirements<StokesForm, decltype(q)>::value,
                "pressure test must receive values");

  STATIC_REQUIRE(sizeof(stokes) > 0);
  static_assert(decltype(grad(u))::shape == ValueShape::tensor);
  static_assert(decltype(div(u))::shape == ValueShape::scalar);
}

TEST_CASE("Expression-template forms retain their structure in types",
          "[forms][expression-templates]")
{
  using namespace expression_templates;

  const auto u      = trial<ValueShape::vector>();
  const auto v      = test<ValueShape::vector>();
  const auto lambda = coefficient(3.0);
  const auto mu     = coefficient(2.0);

  const auto elasticity =
    integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))), dx) +
    integral(lambda * div(v) * div(u), dx);

  static_assert(decltype(grad(u))::shape == ValueShape::tensor);
  static_assert(decltype(sym(grad(v)))::shape == ValueShape::symmetric_tensor);
  static_assert(decltype(inner(sym(grad(v)), sym(grad(u))))::shape ==
                ValueShape::scalar);
  STATIC_REQUIRE(sizeof(elasticity) > 0);

  dealii::Tensor<2, 2, double> gradient;
  gradient[0][0] = 1.0;
  gradient[0][1] = 2.0;
  gradient[1][0] = 4.0;
  gradient[1][1] = 5.0;
  dealii::Tensor<2, 2, double> stress;
  stress = 0.0;
  expression_templates::internal::apply_form(elasticity, gradient, stress);

  REQUIRE(stress[0][0] == Catch::Approx(22.0));
  REQUIRE(stress[0][1] == Catch::Approx(12.0));
  REQUIRE(stress[1][0] == Catch::Approx(12.0));
  REQUIRE(stress[1][1] == Catch::Approx(38.0));

  const auto combined = integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))) +
                                   lambda * div(v) * div(u),
                                 dx);
  STATIC_REQUIRE(FormFields<decltype(combined)>::n_coefficients == 2);
  dealii::Tensor<2, 2, double> combined_stress;
  internal::apply_form(combined, gradient, combined_stress);
  REQUIRE((combined_stress - stress).norm() == Catch::Approx(0.0));
}

TEST_CASE("Scalar forms read independent embedded coefficients",
          "[forms][laplace]")
{
  using namespace expression_templates;

  const auto u             = trial<ValueShape::scalar>();
  const auto v             = test<ValueShape::scalar>();
  const auto unit_form     = integral(inner(grad(v), grad(u)), dx);
  const auto alpha         = coefficient(3.0);
  const auto weighted_form = integral(alpha * inner(grad(v), grad(u)), dx);

  dealii::Tensor<1, 2, double> gradient;
  gradient[0] = 2.0;
  gradient[1] = -1.0;
  dealii::Tensor<1, 2, double> flux;
  flux = 0.0;

  internal::apply_scalar_form(unit_form, gradient, flux, 1.0);
  REQUIRE(flux[0] == Catch::Approx(2.0));
  REQUIRE(flux[1] == Catch::Approx(-1.0));

  flux = 0.0;
  internal::apply_scalar_form(weighted_form, gradient, flux, 1.0);
  REQUIRE(flux[0] == Catch::Approx(6.0));
  REQUIRE(flux[1] == Catch::Approx(-3.0));

  const auto beta      = coefficient(4.0);
  const auto diffusion = coefficient(2.5);
  const auto weighted_helmholtz =
    integral(diffusion * inner(grad(v), grad(u)) + beta * (v * u), dx);
  flux                   = 0.0;
  double submitted_value = 0.0;
  internal::apply_scalar_form(
    weighted_helmholtz, 6.0, gradient, submitted_value, flux, 1.0);
  REQUIRE(flux[0] == Catch::Approx(5.0));
  REQUIRE(flux[1] == Catch::Approx(-2.5));
  REQUIRE(submitted_value == Catch::Approx(24.0));

  STATIC_REQUIRE(FormFields<decltype(weighted_helmholtz)>::n_coefficients == 2);
  STATIC_REQUIRE(std::is_same<decltype(beta), decltype(diffusion)>::value);
  auto       owned_coefficient = coefficient(-2.0);
  const auto copied_form       = integral(owned_coefficient * (v * u), dx);
  owned_coefficient.value      = 9.0;
  submitted_value              = 0.0;
  internal::apply_scalar_form(
    copied_form, 6.0, gradient, submitted_value, flux, 1.0);
  REQUIRE(submitted_value == Catch::Approx(-12.0));

  const auto literal_helmholtz =
    integral(2.0 * inner(grad(v), grad(u)) + v * (3.0 * u), dx);
  flux            = 0.0;
  submitted_value = 0.0;
  internal::apply_scalar_form(
    literal_helmholtz, 6.0, gradient, submitted_value, flux, 1.0);
  REQUIRE(flux[0] == Catch::Approx(4.0));
  REQUIRE(flux[1] == Catch::Approx(-2.0));
  REQUIRE(submitted_value == Catch::Approx(18.0));
}

TEST_CASE("One expression-template form drives both MatrixFree backends",
          "[forms][matrix-free]")
{
  using namespace expression_templates;
  constexpr int dim       = 2;
  constexpr int fe_degree = 1;

  const auto u      = trial<ValueShape::vector>();
  const auto v      = test<ValueShape::vector>();
  const auto lambda = coefficient(3.0);
  const auto mu     = coefficient(2.0);
  const auto form   = integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))) +
                               lambda * div(v) * div(u),
                             dx);
  using Form        = std::decay_t<decltype(form)>;
  STATIC_REQUIRE(
    std::is_same<MatrixFreeOperator<dim, fe_degree, decltype(form)>,
                 MatrixFreeOperator<dim, fe_degree, Form>>::value);
  STATIC_REQUIRE(
    std::is_same<PortableMatrixFreeOperator<dim, fe_degree, decltype((form))>,
                 PortableMatrixFreeOperator<dim, fe_degree, Form>>::value);

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

  std::shared_ptr<const dealii::MatrixFree<dim, double>> const_cpu_data =
    cpu_data;
  const auto cpu_operator =
    make_matrix_free_operator<dim, fe_degree>(const_cpu_data, form);
  typename decltype(cpu_operator)::Vector cpu_source;
  typename decltype(cpu_operator)::Vector cpu_destination;
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

  const auto portable_operator =
    make_portable_matrix_free_operator<dim, fe_degree>(portable_data, form);
  typename decltype(portable_operator)::Vector portable_source;
  typename decltype(portable_operator)::Vector portable_destination;
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
  using namespace expression_templates;
  constexpr int dim    = 2;
  constexpr int degree = 1;

  const auto u     = trial<ValueShape::scalar>();
  const auto v     = test<ValueShape::scalar>();
  const auto alpha = coefficient(2.5);
  const auto beta  = coefficient(1.75);
  const auto form =
    integral(alpha * inner(grad(v), grad(u)) + beta * (v * u), dx);

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
  auto cpu_op = make_matrix_free_operator<dim, degree>(cpu_data, form);
  STATIC_REQUIRE(
    std::is_same<decltype(cpu_op),
                 MatrixFreeOperator<dim, degree, decltype((form))>>::value);
  dealii::LinearAlgebra::distributed::Vector<double> cpu_src, cpu_dst;
  cpu_op.initialize_dof_vector(cpu_src);
  cpu_op.initialize_dof_vector(cpu_dst);
  cpu_src = 0.0;
  for (unsigned int i = 0; i < cpu_src.locally_owned_size(); ++i)
    cpu_src.local_element(i) = static_cast<double>(i + 1);
  cpu_src.update_ghost_values();
  cpu_op.vmult(cpu_dst, cpu_src);

  const auto unit_form = integral(inner(grad(v), grad(u)) + v * u, dx);
  auto       cpu_unit_op =
    make_matrix_free_operator<dim, degree>(cpu_data, unit_form);
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
  auto portable_op =
    make_portable_matrix_free_operator<dim, degree>(portable_data, form);
  typename decltype(portable_op)::Vector portable_src, portable_dst;
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

  auto portable_unit_op =
    make_portable_matrix_free_operator<dim, degree>(portable_data, unit_form);
  typename decltype(portable_unit_op)::Vector portable_unit_src,
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
