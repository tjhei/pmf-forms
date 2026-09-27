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
#include <deal.II/lac/la_parallel_block_vector.h>

#include <forms.h>
#include <matrix_free_operator.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <type_traits>

int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  constexpr int dim         = 2;
  constexpr int degree      = 1;
  constexpr int refinements = 3;

  using namespace pmf::forms;
  using namespace pmf::forms::expression_templates;

  struct VelocityTag
  {};
  struct PressureTag
  {};
  struct ViscosityTag
  {};

  const auto u    = trial<VelocityTag, ValueShape::vector>();
  const auto p    = trial<PressureTag, ValueShape::scalar>();
  const auto v    = test<VelocityTag, ValueShape::vector>();
  const auto q    = test<PressureTag, ValueShape::scalar>();
  const auto mu   = coefficient<ViscosityTag>();
  const auto form = integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))), dx) -
                    integral(div(v) * p, dx) - integral(q * div(u), dx);
  const auto bindings = bind_coefficient<ViscosityTag>(1.7);
  using Form          = typename std::decay<decltype(form)>::type;
  using Coefficients  = typename std::decay<decltype(bindings)>::type;

  static_assert(FormFields<Form>::n_trial_fields == 2,
                "Stokes requires two trial fields");
  static_assert(FormFields<Form>::n_test_fields == 2,
                "Stokes requires two test fields");

  dealii::parallel::distributed::Triangulation<dim> triangulation(
    MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(refinements);

  dealii::FESystem<dim>   velocity_fe(dealii::FE_Q<dim>(degree), dim);
  dealii::FE_Q<dim>       pressure_fe(degree);
  dealii::DoFHandler<dim> velocity_dof_handler(triangulation);
  dealii::DoFHandler<dim> pressure_dof_handler(triangulation);
  velocity_dof_handler.distribute_dofs(velocity_fe);
  pressure_dof_handler.distribute_dofs(pressure_fe);

  dealii::AffineConstraints<double> velocity_constraints;
  dealii::AffineConstraints<double> pressure_constraints;
  velocity_constraints.reinit(velocity_dof_handler.locally_owned_dofs(),
                              dealii::DoFTools::extract_locally_relevant_dofs(
                                velocity_dof_handler));
  pressure_constraints.reinit(pressure_dof_handler.locally_owned_dofs(),
                              dealii::DoFTools::extract_locally_relevant_dofs(
                                pressure_dof_handler));
  velocity_constraints.close();
  pressure_constraints.close();

  const std::vector<const dealii::DoFHandler<dim> *> dof_handlers = {
    &velocity_dof_handler, &pressure_dof_handler};
  const std::vector<const dealii::AffineConstraints<double> *> constraints = {
    &velocity_constraints, &pressure_constraints};

  dealii::MappingQ<dim>   mapping(1);
  const dealii::QGauss<1> quadrature(degree + 1);

  auto cpu_data = std::make_shared<dealii::MatrixFree<dim, double>>();
  dealii::MatrixFree<dim, double>::AdditionalData cpu_additional_data;
  cpu_additional_data.mapping_update_flags = dealii::update_values |
                                             dealii::update_gradients |
                                             dealii::update_JxW_values;
  cpu_data->reinit(
    mapping, dof_handlers, constraints, quadrature, cpu_additional_data);

  using CpuOperator = MatrixFreeOperator<dim, degree, Form, Coefficients>;
  const CpuOperator            cpu_operator(cpu_data, form, bindings);
  typename CpuOperator::Vector cpu_source;
  typename CpuOperator::Vector cpu_result;
  cpu_operator.initialize_dof_vector(cpu_source);
  cpu_operator.initialize_dof_vector(cpu_result);
  for (unsigned int block = 0; block < cpu_source.n_blocks(); ++block)
    {
      auto &values = cpu_source.block(block);
      for (unsigned int i = 0; i < values.locally_owned_size(); ++i)
        values.local_element(i) = std::sin(0.37 * (i + 1 + 17 * block));
      values.update_ghost_values();
    }
  cpu_operator.vmult(cpu_result, cpu_source);

  auto portable_data =
    std::make_shared<dealii::Portable::MatrixFree<dim, double>>();
  dealii::Portable::MatrixFree<dim, double>::AdditionalData
    portable_additional_data;
  portable_additional_data.mapping_update_flags = dealii::update_values |
                                                  dealii::update_gradients |
                                                  dealii::update_JxW_values;
  portable_data->reinit(
    mapping, dof_handlers, constraints, quadrature, portable_additional_data);

  using PortableOperator =
    PortableMatrixFreeOperator<dim, degree, Form, Coefficients>;
  const PortableOperator portable_operator(portable_data, form, bindings);
  typename PortableOperator::Vector portable_source;
  typename PortableOperator::Vector portable_result;
  portable_operator.initialize_dof_vector(portable_source);
  portable_operator.initialize_dof_vector(portable_result);

  using HostBlockVector =
    dealii::LinearAlgebra::distributed::BlockVector<double,
                                                    dealii::MemorySpace::Host>;
  HostBlockVector host_source;
  HostBlockVector host_result;
  portable_data->initialize_dof_vector(host_source);
  portable_data->initialize_dof_vector(host_result);
  for (unsigned int block = 0; block < host_source.n_blocks(); ++block)
    {
      auto &values = host_source.block(block);
      for (unsigned int i = 0; i < values.locally_owned_size(); ++i)
        values.local_element(i) = std::sin(0.37 * (i + 1 + 17 * block));
      values.compress(dealii::VectorOperation::insert);
      portable_source.block(block).import_elements(
        values, dealii::VectorOperation::insert);
    }
  portable_operator.vmult(portable_result, portable_source);
  for (unsigned int block = 0; block < host_result.n_blocks(); ++block)
    host_result.block(block).import_elements(portable_result.block(block),
                                             dealii::VectorOperation::insert);

  double local_error_squared     = 0.0;
  double local_reference_squared = 0.0;
  for (unsigned int block = 0; block < cpu_result.n_blocks(); ++block)
    for (unsigned int i = 0; i < cpu_result.block(block).locally_owned_size();
         ++i)
      {
        const double difference = cpu_result.block(block).local_element(i) -
                                  host_result.block(block).local_element(i);
        local_error_squared += difference * difference;
        local_reference_squared += cpu_result.block(block).local_element(i) *
                                   cpu_result.block(block).local_element(i);
      }

  const double error =
    std::sqrt(dealii::Utilities::MPI::sum(local_error_squared, MPI_COMM_WORLD));
  const double reference_norm = std::sqrt(
    dealii::Utilities::MPI::sum(local_reference_squared, MPI_COMM_WORLD));
  const double relative_error = error / std::max(reference_norm, 1e-30);

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Viscosity: 1.7\n"
              << "Relative CPU / Portable MatrixFree error: " << relative_error
              << '\n';

  return relative_error < 1e-11 ? 0 : 1;
}
