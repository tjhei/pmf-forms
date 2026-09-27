#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <elasticity_matrix_free.h>
#include <forms.h>

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

  struct DiffusionTag
  {};

  const auto u            = trial();
  const auto v            = test();
  const auto diffusion    = coefficient<DiffusionTag>();
  const auto form         = integral(diffusion * inner(grad(v), grad(u)), dx);
  const auto coefficients = bind_coefficient<DiffusionTag>(2.5);
  using Form              = typename std::decay<decltype(form)>::type;
  using Coefficients      = typename std::decay<decltype(coefficients)>::type;

  dealii::parallel::distributed::Triangulation<dim> triangulation(
    MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(refinements);

  dealii::FE_Q<dim>       finite_element(degree);
  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(finite_element);

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

  using CpuOperator = MatrixFreeOperator<dim, degree, Form, Coefficients>;
  const CpuOperator            cpu_operator(cpu_data, form, coefficients);
  typename CpuOperator::Vector cpu_source;
  typename CpuOperator::Vector cpu_result;
  cpu_operator.initialize_dof_vector(cpu_source);
  cpu_operator.initialize_dof_vector(cpu_result);
  for (unsigned int i = 0; i < cpu_source.locally_owned_size(); ++i)
    cpu_source.local_element(i) = std::sin(0.37 * (i + 1));
  cpu_source.update_ghost_values();
  cpu_operator.vmult(cpu_result, cpu_source);

  auto portable_data =
    std::make_shared<dealii::Portable::MatrixFree<dim, double>>();
  dealii::Portable::MatrixFree<dim, double>::AdditionalData
    portable_additional_data;
  portable_additional_data.mapping_update_flags =
    dealii::update_gradients | dealii::update_JxW_values;
  portable_data->reinit(
    mapping, dof_handler, constraints, quadrature, portable_additional_data);

  using PortableOperator =
    PortableMatrixFreeOperator<dim, degree, Form, Coefficients>;
  const PortableOperator portable_operator(portable_data, form, coefficients);
  typename PortableOperator::Vector portable_source;
  typename PortableOperator::Vector portable_result;
  portable_operator.initialize_dof_vector(portable_source);
  portable_operator.initialize_dof_vector(portable_result);

  using HostVector =
    dealii::LinearAlgebra::distributed::Vector<double,
                                               dealii::MemorySpace::Host>;
  HostVector host_source;
  HostVector host_result;
  portable_data->initialize_dof_vector(host_source, 0);
  portable_data->initialize_dof_vector(host_result, 0);
  for (unsigned int i = 0; i < host_source.locally_owned_size(); ++i)
    host_source.local_element(i) = std::sin(0.37 * (i + 1));
  host_source.compress(dealii::VectorOperation::insert);
  portable_source.import_elements(host_source, dealii::VectorOperation::insert);
  portable_operator.vmult(portable_result, portable_source);
  host_result.import_elements(portable_result, dealii::VectorOperation::insert);

  double local_error_squared     = 0.0;
  double local_reference_squared = 0.0;
  for (unsigned int i = 0; i < cpu_result.locally_owned_size(); ++i)
    {
      const double difference =
        cpu_result.local_element(i) - host_result.local_element(i);
      local_error_squared += difference * difference;
      local_reference_squared +=
        cpu_result.local_element(i) * cpu_result.local_element(i);
    }

  const double error =
    std::sqrt(dealii::Utilities::MPI::sum(local_error_squared, MPI_COMM_WORLD));
  const double reference_norm = std::sqrt(
    dealii::Utilities::MPI::sum(local_reference_squared, MPI_COMM_WORLD));
  const double relative_error = error / std::max(reference_norm, 1e-30);

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Diffusion coefficient: 2.5\n"
              << "Relative CPU / Portable MatrixFree error: " << relative_error
              << '\n';

  return relative_error < 1e-11 ? 0 : 1;
}
