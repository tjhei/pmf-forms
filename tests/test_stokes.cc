#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <catch2/catch_test_macros.hpp>
#include <forms.h>
#include <matrix_free_operator.h>

#include <algorithm>
#include <cmath>
#include <memory>

namespace
{
  double
  bubble(const double coordinate)
  {
    return coordinate * coordinate * (1.0 - coordinate) * (1.0 - coordinate);
  }

  double
  bubble_derivative(const double coordinate)
  {
    return 2.0 * coordinate - 6.0 * coordinate * coordinate +
           4.0 * coordinate * coordinate * coordinate;
  }

  double
  bubble_second_derivative(const double coordinate)
  {
    return 2.0 - 12.0 * coordinate + 12.0 * coordinate * coordinate;
  }

  double
  bubble_third_derivative(const double coordinate)
  {
    return -12.0 + 24.0 * coordinate;
  }

  class ManufacturedVelocity : public dealii::Function<2>
  {
  public:
    ManufacturedVelocity()
      : dealii::Function<2>(2)
    {}

    double
    value(const dealii::Point<2> &point,
          const unsigned int      component) const override
    {
      return component == 0 ? bubble(point[0]) * bubble_derivative(point[1]) :
                              -bubble_derivative(point[0]) * bubble(point[1]);
    }
  };

  class ManufacturedPressure : public dealii::Function<2>
  {
  public:
    double
    value(const dealii::Point<2> &point, const unsigned int = 0) const override
    {
      return point[0] + 2.0 * point[1] - 1.5;
    }
  };

  dealii::Tensor<1, 2>
  manufactured_force(const dealii::Point<2> &point, const double viscosity)
  {
    dealii::Tensor<1, 2> force;
    force[0] =
      1.0 - viscosity * (bubble_second_derivative(point[0]) *
                           bubble_derivative(point[1]) +
                         bubble(point[0]) * bubble_third_derivative(point[1]));
    force[1] =
      2.0 + viscosity * (bubble_third_derivative(point[0]) * bubble(point[1]) +
                         bubble_derivative(point[0]) *
                           bubble_second_derivative(point[1]));
    return force;
  }
} // namespace

TEST_CASE("Stokes action matches a manufactured solution",
          "[forms][stokes][manufactured]")
{
  using namespace pmf::forms;
  using namespace pmf::forms::expression_templates;
  constexpr int    dim       = 2;
  constexpr int    degree    = 4;
  constexpr double viscosity = 1.7;
  const auto [velocity, pressure] =
    trial_functions<ValueShape::vector, ValueShape::scalar>();
  const auto [test_velocity, test_pressure] =
    test_functions<ValueShape::vector, ValueShape::scalar>();
  const auto form =
    integral(2.0 * coefficient(viscosity) *
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
  dealii::VectorTools::interpolate_boundary_values(velocity_dofs,
                                                   0,
                                                   ManufacturedVelocity{},
                                                   velocity_constraints);
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

  typename decltype(cpu)::Vector source, expected, cpu_result, portable_result;
  cpu.initialize_dof_vector(source);
  cpu.initialize_dof_vector(expected);
  cpu.initialize_dof_vector(cpu_result);
  cpu.initialize_dof_vector(portable_result);
  dealii::VectorTools::interpolate(mapping,
                                   velocity_dofs,
                                   ManufacturedVelocity{},
                                   source.block(0));
  dealii::VectorTools::interpolate(mapping,
                                   pressure_dofs,
                                   ManufacturedPressure{},
                                   source.block(1));
  velocity_constraints.distribute(source.block(0));
  source.update_ghost_values();

  expected = 0.0;
  dealii::FEValues<dim>  evaluation(mapping,
                                   velocity_fe,
                                   dealii::QGauss<dim>(degree + 2),
                                   dealii::update_values |
                                     dealii::update_quadrature_points |
                                     dealii::update_JxW_values);
  dealii::Vector<double> local_rhs(velocity_fe.n_dofs_per_cell());
  std::vector<dealii::types::global_dof_index> indices(
    velocity_fe.n_dofs_per_cell());
  for (const auto &cell : velocity_dofs.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        evaluation.reinit(cell);
        local_rhs = 0.0;
        for (unsigned int point = 0; point < evaluation.n_quadrature_points;
             ++point)
          {
            const auto force =
              manufactured_force(evaluation.quadrature_point(point), viscosity);
            for (unsigned int basis = 0; basis < velocity_fe.n_dofs_per_cell();
                 ++basis)
              {
                const auto component =
                  velocity_fe.system_to_component_index(basis).first;
                local_rhs[basis] +=
                  evaluation.shape_value_component(basis, point, component) *
                  force[component] * evaluation.JxW(point);
              }
          }
        cell->get_dof_indices(indices);
        velocity_constraints.distribute_local_to_global(local_rhs,
                                                        indices,
                                                        expected.block(0));
      }
  expected.compress(dealii::VectorOperation::add);
  cpu.vmult(cpu_result, source);
  typename decltype(portable)::Vector device_source, device_result;
  portable.initialize_dof_vector(device_source);
  portable.initialize_dof_vector(device_result);
  for (unsigned int block = 0; block < source.n_blocks(); ++block)
    device_source.block(block).import_elements(source.block(block),
                                               dealii::VectorOperation::insert);
  portable.vmult(device_result, device_source);
  for (unsigned int block = 0; block < source.n_blocks(); ++block)
    portable_result.block(block).import_elements(
      device_result.block(block), dealii::VectorOperation::insert);

  const double reference_norm = expected.l2_norm();
  REQUIRE(reference_norm > 1e-3);
  cpu_result -= expected;
  portable_result -= expected;
  CHECK(cpu_result.l2_norm() / reference_norm < 1e-11);
  CHECK(portable_result.l2_norm() / reference_norm < 1e-11);
}

TEST_CASE("Stokes CPU and Portable operators agree",
          "[forms][stokes][matrix-free]")
{
  constexpr int dim         = 2;
  constexpr int degree      = 1;
  constexpr int refinements = 3;

  using namespace pmf::forms;
  using namespace pmf::forms::expression_templates;


  const auto [u, p] = trial_functions<ValueShape::vector, ValueShape::scalar>();
  const auto [v, q] = test_functions<ValueShape::vector, ValueShape::scalar>();
  const auto mu     = coefficient(1.7);
  const auto form = integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))), dx) -
                    integral(div(v) * p, dx) - integral(q * div(u), dx);

  static_assert(FormFields<decltype(form)>::n_trial_fields == 2,
                "Stokes requires two trial fields");
  static_assert(FormFields<decltype(form)>::n_test_fields == 2,
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

  const auto cpu_operator =
    make_matrix_free_operator<dim, degree>(cpu_data, form);
  typename decltype(cpu_operator)::Vector cpu_source;
  typename decltype(cpu_operator)::Vector cpu_result;
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

  const auto portable_operator =
    make_portable_matrix_free_operator<dim, degree>(portable_data, form);
  typename decltype(portable_operator)::Vector portable_source;
  typename decltype(portable_operator)::Vector portable_result;
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

  REQUIRE(reference_norm > 0.0);
  REQUIRE(relative_error < 1e-11);
}
