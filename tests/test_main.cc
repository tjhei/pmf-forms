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

namespace
{
  template <int rank, int dim, typename Number>
  struct PointEvaluation
  {
    using value_type =
      std::conditional_t<rank == 1, Number, dealii::Tensor<1, dim, Number>>;
    using gradient_type = dealii::Tensor<rank, dim, Number>;
    value_type    value{};
    gradient_type gradient{};

    value_type
    get_value(unsigned int) const
    {
      return value;
    }
    gradient_type
    get_gradient(unsigned int) const
    {
      return gradient;
    }
  };

  template <typename Form, int rank, int dim, typename Number>
  auto
  point_action(const Form                              &form,
               const dealii::Tensor<rank, dim, Number> &gradient,
               const Number                             value = Number())
  {
    using namespace expression_templates;
    PointEvaluation<rank, dim, Number> evaluation;
    evaluation.gradient = gradient;
    if constexpr (rank == 1)
      evaluation.value = value;
    using Field = Trial<0, rank == 1 ? ValueShape::scalar : ValueShape::vector>;
    internal::DiagonalContext<dim, Field, decltype(evaluation)> context{
      evaluation, 0};
    BilinearCellKernel<Form>{form}(context);
    return std::make_pair(context.submitted_value, context.submitted_gradient);
  }

  template <typename Destination, typename Source>
  void
  copy_vector(Destination &destination, const Source &source)
  {
    destination.import_elements(source, dealii::VectorOperation::insert);
  }

  template <typename Number, typename DestinationSpace, typename SourceSpace>
  void
  copy_vector(
    dealii::LinearAlgebra::distributed::BlockVector<Number, DestinationSpace>
      &destination,
    const dealii::LinearAlgebra::distributed::BlockVector<Number, SourceSpace>
      &source)
  {
    for (unsigned int block = 0; block < source.n_blocks(); ++block)
      copy_vector(destination.block(block), source.block(block));
  }

  template <typename CpuOperator, typename PortableOperator>
  void
  check_diagonals(const CpuOperator &cpu, const PortableOperator &portable)
  {
    const auto &cpu_diagonal      = cpu.get_diagonal();
    const auto &portable_diagonal = portable.get_diagonal();
    CHECK(&cpu.get_diagonal() == &cpu_diagonal);
    CHECK(&portable.get_diagonal() == &portable_diagonal);

    typename CpuOperator::Vector source, image, portable_image, host_diagonal;
    cpu.initialize_dof_vector(source);
    cpu.initialize_dof_vector(image);
    cpu.initialize_dof_vector(portable_image);
    cpu.initialize_dof_vector(host_diagonal);
    copy_vector(host_diagonal, portable_diagonal);
    typename PortableOperator::Vector portable_source, portable_result;
    portable.initialize_dof_vector(portable_source);
    portable.initialize_dof_vector(portable_result);

    const auto owned = source.locally_owned_elements();
    for (dealii::types::global_dof_index index = 0; index < source.size();
         ++index)
      {
        source = 0.0;
        if (owned.is_element(index))
          source[index] = 1.0;
        source.compress(dealii::VectorOperation::insert);
        source.update_ghost_values();
        cpu.vmult(image, source);
        copy_vector(portable_source, source);
        portable.vmult(portable_result, portable_source);
        copy_vector(portable_image, portable_result);
        if (owned.is_element(index))
          {
            INFO("diagonal index " << index);
            CHECK(cpu_diagonal[index] ==
                  Catch::Approx(image[index]).margin(1e-12));
            CHECK(host_diagonal[index] ==
                  Catch::Approx(portable_image[index]).margin(1e-12));
            CHECK(host_diagonal[index] ==
                  Catch::Approx(cpu_diagonal[index]).margin(1e-12));
          }
        source.zero_out_ghost_values();
      }
  }
} // namespace

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
  BilinearCellKernel<ReorderedForm> reordered_kernel(reordered);
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
  stress = point_action(elasticity, gradient).second;

  REQUIRE(stress[0][0] == Catch::Approx(22.0));
  REQUIRE(stress[0][1] == Catch::Approx(12.0));
  REQUIRE(stress[1][0] == Catch::Approx(12.0));
  REQUIRE(stress[1][1] == Catch::Approx(38.0));

  const auto combined = integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))) +
                                   lambda * div(v) * div(u),
                                 dx);
  STATIC_REQUIRE(FormFields<decltype(combined)>::n_coefficients == 2);
  dealii::Tensor<2, 2, double> combined_stress;
  combined_stress = point_action(combined, gradient).second;
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

  flux = point_action(unit_form, gradient).second;
  REQUIRE(flux[0] == Catch::Approx(2.0));
  REQUIRE(flux[1] == Catch::Approx(-1.0));

  flux = 0.0;
  flux = point_action(weighted_form, gradient).second;
  REQUIRE(flux[0] == Catch::Approx(6.0));
  REQUIRE(flux[1] == Catch::Approx(-3.0));

  const auto beta      = coefficient(4.0);
  const auto diffusion = coefficient(2.5);
  const auto weighted_helmholtz =
    integral(diffusion * inner(grad(v), grad(u)) + beta * (v * u), dx);
  flux                   = 0.0;
  double submitted_value = 0.0;
  std::tie(submitted_value, flux) =
    point_action(weighted_helmholtz, gradient, 6.0);
  REQUIRE(flux[0] == Catch::Approx(5.0));
  REQUIRE(flux[1] == Catch::Approx(-2.5));
  REQUIRE(submitted_value == Catch::Approx(24.0));

  STATIC_REQUIRE(FormFields<decltype(weighted_helmholtz)>::n_coefficients == 2);
  STATIC_REQUIRE(std::is_same<decltype(beta), decltype(diffusion)>::value);
  auto       owned_coefficient    = coefficient(-2.0);
  const auto copied_form          = integral(owned_coefficient * (v * u), dx);
  owned_coefficient.value         = 9.0;
  submitted_value                 = 0.0;
  std::tie(submitted_value, flux) = point_action(copied_form, gradient, 6.0);
  REQUIRE(submitted_value == Catch::Approx(-12.0));

  const auto literal_helmholtz =
    integral(2.0 * inner(grad(v), grad(u)) + v * (3.0 * u), dx);
  flux            = 0.0;
  submitted_value = 0.0;
  std::tie(submitted_value, flux) =
    point_action(literal_helmholtz, gradient, 6.0);
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
  if (dealii::DoFTools::extract_locally_relevant_dofs(dof_handler)
        .is_element(0))
    constraints.add_line(0);
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
  constraints.set_zero(cpu_source);
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
  constraints.set_zero(host_source);
  host_source.compress(dealii::VectorOperation::insert);
  portable_source.import_elements(host_source, dealii::VectorOperation::insert);
  portable_operator.vmult(portable_destination, portable_source);

  REQUIRE(cpu_destination.l2_norm() > 0.0);
  REQUIRE(portable_destination.l2_norm() ==
          Catch::Approx(cpu_destination.l2_norm()).epsilon(1e-10));
  check_diagonals(cpu_operator, portable_operator);
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
  if (dealii::DoFTools::extract_locally_relevant_dofs(dof_handler)
        .is_element(0))
    constraints.add_line(0);
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
  constraints.set_zero(cpu_src);
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
  constraints.set_zero(host_src);
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
  check_diagonals(cpu_op, portable_op);
  check_diagonals(cpu_unit_op, portable_unit_op);
  const double diagonal_norm = cpu_op.get_diagonal().l2_norm();
  const double inverse_norm  = cpu_op.get_inverse_diagonal().l2_norm();
  CHECK(portable_op.get_inverse_diagonal().l2_norm() ==
        Catch::Approx(inverse_norm));
  cpu_op.compute_diagonal();
  portable_op.compute_diagonal();
  CHECK(cpu_op.get_inverse_diagonal().l2_norm() == Catch::Approx(inverse_norm));
  CHECK(portable_op.get_inverse_diagonal().l2_norm() ==
        Catch::Approx(inverse_norm));
  CHECK(cpu_op.get_diagonal().l2_norm() == Catch::Approx(diagonal_norm));
  CHECK(portable_op.get_diagonal().l2_norm() == Catch::Approx(diagonal_norm));
  CHECK(&cpu_op.get_matrix_diagonal()->get_vector() == &cpu_op.get_diagonal());
}

TEST_CASE("Stokes diagonals include the zero pressure block",
          "[forms][diagonal][stokes]")
{
  using namespace expression_templates;
  constexpr int dim    = 2;
  constexpr int degree = 1;
  const auto [velocity, pressure] =
    trial_functions<ValueShape::vector, ValueShape::scalar>();
  const auto [test_velocity, test_pressure] =
    test_functions<ValueShape::vector, ValueShape::scalar>();
  const auto form =
    integral(2.0 * coefficient(1.7) *
               inner(sym(grad(test_velocity)), sym(grad(velocity))),
             dx) -
    integral(div(test_velocity) * pressure, dx) -
    integral(test_pressure * div(velocity), dx);

  dealii::parallel::distributed::Triangulation<dim> triangulation(
    MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  dealii::FESystem<dim>   velocity_fe(dealii::FE_Q<dim>(degree), dim);
  dealii::FE_Q<dim>       pressure_fe(degree);
  dealii::DoFHandler<dim> velocity_dofs(triangulation),
    pressure_dofs(triangulation);
  velocity_dofs.distribute_dofs(velocity_fe);
  pressure_dofs.distribute_dofs(pressure_fe);
  dealii::AffineConstraints<double> velocity_constraints, pressure_constraints;
  velocity_constraints.reinit(velocity_dofs.locally_owned_dofs(),
                              dealii::DoFTools::extract_locally_relevant_dofs(
                                velocity_dofs));
  pressure_constraints.reinit(pressure_dofs.locally_owned_dofs(),
                              dealii::DoFTools::extract_locally_relevant_dofs(
                                pressure_dofs));
  if (dealii::DoFTools::extract_locally_relevant_dofs(velocity_dofs)
        .is_element(0))
    velocity_constraints.add_line(0);
  if (dealii::DoFTools::extract_locally_relevant_dofs(pressure_dofs)
        .is_element(0))
    pressure_constraints.add_line(0);
  velocity_constraints.close();
  pressure_constraints.close();
  const std::vector<const dealii::DoFHandler<dim> *> dof_handlers = {
    &velocity_dofs, &pressure_dofs};
  const std::vector<const dealii::AffineConstraints<double> *> constraints = {
    &velocity_constraints, &pressure_constraints};
  dealii::MappingQ<dim> mapping(1);
  dealii::QGauss<1>     quadrature(degree + 1);
  auto cpu_data = std::make_shared<dealii::MatrixFree<dim, double>>();
  dealii::MatrixFree<dim, double>::AdditionalData cpu_settings;
  cpu_settings.mapping_update_flags = dealii::update_values |
                                      dealii::update_gradients |
                                      dealii::update_JxW_values;
  cpu_data->reinit(
    mapping, dof_handlers, constraints, quadrature, cpu_settings);
  auto portable_data =
    std::make_shared<dealii::Portable::MatrixFree<dim, double>>();
  dealii::Portable::MatrixFree<dim, double>::AdditionalData portable_settings;
  portable_settings.mapping_update_flags = cpu_settings.mapping_update_flags;
  portable_data->reinit(
    mapping, dof_handlers, constraints, quadrature, portable_settings);
  const auto cpu = make_matrix_free_operator<dim, degree>(cpu_data, form);
  const auto portable =
    make_portable_matrix_free_operator<dim, degree>(portable_data, form);
  check_diagonals(cpu, portable);
  const auto &pressure_diagonal = cpu.get_diagonal().block(1);
  for (const auto index : pressure_diagonal.locally_owned_elements())
    CHECK(pressure_diagonal[index] == (index == 0 ? 1.0 : 0.0));
}
