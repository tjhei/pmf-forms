#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>

#include <catch2/catch_test_macros.hpp>
#include <matrix_free_operator.h>

#include <array>
#include <cmath>

using namespace pmf::forms;
using namespace pmf::forms::expression_templates;

namespace
{
  struct Weights
  {
    double diffusion         = 2.3;
    double reaction          = 0.7;
    double mu                = 1.7;
    double lambda            = 0.0;
    double velocity_mass     = 0.0;
    double velocity_pressure = -1.0;
    double pressure_velocity = -1.0;
    double pressure_mass     = 0.0;
    double pressure_gradient = 0.0;
  };

  template <bool mixed, typename Vector>
  decltype(auto)
  block(Vector &vector, const unsigned int index)
  {
    if constexpr (mixed)
      return vector.block(index);
    else
      return (vector);
  }

  template <bool mixed, typename Destination, typename Source>
  void
  copy(Destination &destination, const Source &source)
  {
    for (unsigned int index = 0; index < (mixed ? 2 : 1); ++index)
      block<mixed>(destination, index)
        .import_elements(block<mixed>(source, index),
                         dealii::VectorOperation::insert);
  }

  template <int components, bool mixed, typename Form>
  void
  check_against_fe_values(const Form &form, const Weights weights)
  {
    constexpr int                                     dim    = 2;
    constexpr int                                     degree = 2;
    dealii::parallel::distributed::Triangulation<dim> triangulation(
      MPI_COMM_WORLD);
    dealii::GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(1);
    dealii::FESystem<dim>   primary_fe(dealii::FE_Q<dim>(degree), components);
    dealii::FE_Q<dim>       pressure_fe(degree);
    dealii::DoFHandler<dim> primary_dofs(triangulation),
      pressure_dofs(triangulation);
    primary_dofs.distribute_dofs(primary_fe);
    pressure_dofs.distribute_dofs(pressure_fe);
    std::vector<const dealii::DoFHandler<dim> *> dof_handlers{&primary_dofs};
    if constexpr (mixed)
      dof_handlers.push_back(&pressure_dofs);
    std::array<dealii::AffineConstraints<double>, mixed ? 2 : 1> constraints;
    std::vector<const dealii::AffineConstraints<double> *> constraint_pointers;
    for (unsigned int field = 0; field < dof_handlers.size(); ++field)
      {
        const auto relevant =
          dealii::DoFTools::extract_locally_relevant_dofs(*dof_handlers[field]);
        constraints[field].reinit(dof_handlers[field]->locally_owned_dofs(),
                                  relevant);
        if (relevant.is_element(0))
          constraints[field].add_line(0);
        constraints[field].close();
        constraint_pointers.push_back(&constraints[field]);
      }
    dealii::MappingQ<dim> mapping(1);
    dealii::QGauss<1>     quadrature(degree + 1);
    auto cpu_data = std::make_shared<dealii::MatrixFree<dim, double>>();
    dealii::MatrixFree<dim, double>::AdditionalData cpu_settings;
    cpu_settings.mapping_update_flags = dealii::update_values |
                                        dealii::update_gradients |
                                        dealii::update_JxW_values;
    cpu_data->reinit(
      mapping, dof_handlers, constraint_pointers, quadrature, cpu_settings);
    auto portable_data =
      std::make_shared<dealii::Portable::MatrixFree<dim, double>>();
    dealii::Portable::MatrixFree<dim, double>::AdditionalData portable_settings;
    portable_settings.mapping_update_flags = cpu_settings.mapping_update_flags;
    portable_data->reinit(mapping,
                          dof_handlers,
                          constraint_pointers,
                          quadrature,
                          portable_settings);
    const auto cpu = make_matrix_free_operator<dim, degree>(cpu_data, form);
    const auto portable =
      make_portable_matrix_free_operator<dim, degree>(portable_data, form);
    typename decltype(cpu)::Vector source, reference, diagonal, cpu_result,
      portable_result;
    cpu.initialize_dof_vector(source);
    cpu.initialize_dof_vector(reference);
    cpu.initialize_dof_vector(diagonal);
    cpu.initialize_dof_vector(cpu_result);
    cpu.initialize_dof_vector(portable_result);
    for (unsigned int field = 0; field < dof_handlers.size(); ++field)
      {
        auto &values = block<mixed>(source, field);
        for (const auto index : values.locally_owned_elements())
          values[index] = std::sin(0.37 * (index + 1)) + 0.2 * field;
        constraints[field].set_zero(values);
      }
    source.update_ghost_values();

    const dealii::QGauss<dim>                reference_quadrature(degree + 2);
    dealii::FEValues<dim>                    primary_values(mapping,
                                         primary_fe,
                                         reference_quadrature,
                                         cpu_settings.mapping_update_flags);
    dealii::FEValues<dim>                    pressure_values(mapping,
                                          pressure_fe,
                                          reference_quadrature,
                                          cpu_settings.mapping_update_flags);
    const dealii::FEValuesExtractors::Scalar scalar(0);
    const dealii::FEValuesExtractors::Vector vector(0);
    std::vector<dealii::types::global_dof_index> indices(
      primary_fe.n_dofs_per_cell());
    std::vector<dealii::types::global_dof_index> pressure_indices(
      pressure_fe.n_dofs_per_cell());
    for (const auto &cell : primary_dofs.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          primary_values.reinit(cell);
          cell->get_dof_indices(indices);
          if constexpr (mixed)
            {
              const typename dealii::DoFHandler<dim>::active_cell_iterator
                pressure_cell(&triangulation,
                              cell->level(),
                              cell->index(),
                              &pressure_dofs);
              pressure_values.reinit(pressure_cell);
              pressure_cell->get_dof_indices(pressure_indices);
            }
          dealii::Vector<double> local_result(indices.size()),
            local_diagonal(indices.size());
          dealii::Vector<double> pressure_result(pressure_indices.size()),
            pressure_diagonal(pressure_indices.size());
          for (unsigned int point = 0; point < reference_quadrature.size();
               ++point)
            {
              double                 pressure = 0.0;
              dealii::Tensor<1, dim> pressure_gradient;
              if constexpr (mixed)
                for (unsigned int basis = 0; basis < pressure_indices.size();
                     ++basis)
                  {
                    pressure += source.block(1)[pressure_indices[basis]] *
                                pressure_values.shape_value(basis, point);
                    pressure_gradient +=
                      source.block(1)[pressure_indices[basis]] *
                      pressure_values.shape_grad(basis, point);
                  }
              if constexpr (components == 1)
                {
                  double                 value = 0.0;
                  dealii::Tensor<1, dim> gradient;
                  for (unsigned int basis = 0; basis < indices.size(); ++basis)
                    {
                      const double entry = source[indices[basis]];
                      value +=
                        entry * primary_values[scalar].value(basis, point);
                      gradient +=
                        entry * primary_values[scalar].gradient(basis, point);
                    }
                  for (unsigned int basis = 0; basis < indices.size(); ++basis)
                    {
                      const double shape_value =
                        primary_values[scalar].value(basis, point);
                      const auto shape_gradient =
                        primary_values[scalar].gradient(basis, point);
                      local_result[basis] +=
                        (weights.diffusion * (shape_gradient * gradient) +
                         weights.reaction * shape_value * value) *
                        primary_values.JxW(point);
                      local_diagonal[basis] +=
                        (weights.diffusion * shape_gradient.norm_square() +
                         weights.reaction * shape_value * shape_value) *
                        primary_values.JxW(point);
                    }
                }
              else
                {
                  dealii::Tensor<1, dim> value;
                  dealii::Tensor<2, dim> gradient;
                  for (unsigned int basis = 0; basis < indices.size(); ++basis)
                    {
                      const double entry =
                        block<mixed>(source, 0)[indices[basis]];
                      value +=
                        entry * primary_values[vector].value(basis, point);
                      gradient +=
                        entry * primary_values[vector].gradient(basis, point);
                    }
                  const auto   strain     = dealii::symmetrize(gradient);
                  const double divergence = dealii::trace(gradient);
                  for (unsigned int basis = 0; basis < indices.size(); ++basis)
                    {
                      const auto shape_value =
                        primary_values[vector].value(basis, point);
                      const auto shape_strain =
                        primary_values[vector].symmetric_gradient(basis, point);
                      const double shape_divergence =
                        primary_values[vector].divergence(basis, point);
                      local_result[basis] +=
                        (2.0 * weights.mu *
                           dealii::scalar_product(shape_strain, strain) +
                         weights.lambda * shape_divergence * divergence +
                         weights.velocity_mass * (shape_value * value) +
                         weights.pressure_gradient *
                           (shape_value * pressure_gradient) +
                         weights.velocity_pressure * shape_divergence *
                           pressure) *
                        primary_values.JxW(point);
                      local_diagonal[basis] +=
                        (2.0 * weights.mu *
                           dealii::scalar_product(shape_strain, shape_strain) +
                         weights.lambda * shape_divergence * shape_divergence +
                         weights.velocity_mass * shape_value.norm_square()) *
                        primary_values.JxW(point);
                    }
                  if constexpr (mixed)
                    for (unsigned int basis = 0;
                         basis < pressure_indices.size();
                         ++basis)
                      {
                        const double shape_value =
                          pressure_values.shape_value(basis, point);
                        pressure_result[basis] +=
                          shape_value *
                          (weights.pressure_velocity * divergence +
                           weights.pressure_mass * pressure) *
                          pressure_values.JxW(point);
                        pressure_diagonal[basis] += weights.pressure_mass *
                                                    shape_value * shape_value *
                                                    pressure_values.JxW(point);
                      }
                }
            }
          constraints[0].distribute_local_to_global(local_result,
                                                    indices,
                                                    block<mixed>(reference, 0));
          constraints[0].distribute_local_to_global(local_diagonal,
                                                    indices,
                                                    block<mixed>(diagonal, 0));
          if constexpr (mixed)
            {
              constraints[1].distribute_local_to_global(pressure_result,
                                                        pressure_indices,
                                                        reference.block(1));
              constraints[1].distribute_local_to_global(pressure_diagonal,
                                                        pressure_indices,
                                                        diagonal.block(1));
            }
        }
    reference.compress(dealii::VectorOperation::add);
    diagonal.compress(dealii::VectorOperation::add);
    for (unsigned int field = 0; field < dof_handlers.size(); ++field)
      if (block<mixed>(diagonal, field).locally_owned_elements().is_element(0))
        block<mixed>(diagonal, field)[0] = 1.0;
    cpu.vmult(cpu_result, source);
    typename decltype(portable)::Vector device_source, device_result;
    portable.initialize_dof_vector(device_source);
    portable.initialize_dof_vector(device_result);
    copy<mixed>(device_source, source);
    portable.vmult(device_result, device_source);
    copy<mixed>(portable_result, device_result);
    REQUIRE(reference.l2_norm() > 0.0);
    cpu_result -= reference;
    portable_result -= reference;
    CHECK(cpu_result.l2_norm() / reference.l2_norm() < 1e-11);
    CHECK(portable_result.l2_norm() / reference.l2_norm() < 1e-11);
    cpu_result = cpu.get_diagonal();
    copy<mixed>(portable_result, portable.get_diagonal());
    cpu_result -= diagonal;
    portable_result -= diagonal;
    REQUIRE(diagonal.l2_norm() > 0.0);
    CHECK(cpu_result.l2_norm() / diagonal.l2_norm() < 1e-11);
    CHECK(portable_result.l2_norm() / diagonal.l2_norm() < 1e-11);
  }
} // namespace

TEST_CASE("Generic scalar lowering matches independent FEValues integration",
          "[forms][lowering]")
{
  const auto trial_field = trial();
  const auto test_field  = test();
  const auto alpha       = coefficient(2.3);
  const auto beta        = coefficient(0.7);
  SECTION("diffusion and left-associated reaction")
  {
    const auto form =
      integral(alpha * inner(grad(test_field), grad(trial_field)) +
                 beta * test_field * trial_field,
               dx);
    check_against_fe_values<1, false>(form, Weights{});
  }
  SECTION("reversed contraction and right-associated coefficients")
  {
    const auto form =
      integral(inner(grad(trial_field), alpha * grad(test_field)) +
                 trial_field * (test_field * beta),
               dx);
    check_against_fe_values<1, false>(form, Weights{});
  }
  STATIC_REQUIRE_FALSE(internal::is_bilinear_cell_form<decltype(integral(
                         trial_field * trial_field * test_field, dx))>);
  STATIC_REQUIRE_FALSE(internal::is_bilinear_cell_form<decltype(integral(
                         test_field * test_field * trial_field, dx))>);
  STATIC_REQUIRE_FALSE(internal::is_bilinear_cell_form<decltype(integral(
                         trial_field * test_field + coefficient(1.0), dx))>);
}

TEST_CASE("Generic vector lowering matches independent FEValues integration",
          "[forms][lowering]")
{
  const auto [trial_field] = trial_functions<ValueShape::vector>();
  const auto [test_field]  = test_functions<ValueShape::vector>();
  const auto mu            = coefficient(1.7);
  const auto lambda        = coefficient(3.1);
  Weights    weights;
  weights.lambda = 3.1;
  SECTION("isotropic elasticity")
  {
    const auto form =
      integral(2 * mu * inner(sym(grad(test_field)), sym(grad(trial_field))) +
                 lambda * div(test_field) * div(trial_field),
               dx);
    check_against_fe_values<2, false>(form, weights);
  }
  SECTION("reordered terms and coefficient sums")
  {
    const auto form =
      integral(div(trial_field) * (lambda * div(test_field)), dx) +
      integral(inner(sym(grad(trial_field)), sym(grad(test_field)) * (mu + mu)),
               dx);
    check_against_fe_values<2, false>(form, weights);
  }
}

TEST_CASE(
  "Generic mixed lowering follows the expression rather than a Stokes pattern",
  "[forms][lowering]")
{
  const auto [velocity, pressure] =
    trial_functions<ValueShape::vector, ValueShape::scalar>();
  const auto [test_velocity, test_pressure] =
    test_functions<ValueShape::vector, ValueShape::scalar>();
  const auto mu = coefficient(1.7);
  SECTION("Stokes")
  {
    const auto form =
      integral(2 * mu * inner(sym(grad(test_velocity)), sym(grad(velocity))),
               dx) -
      integral(div(test_velocity) * pressure, dx) -
      integral(test_pressure * div(velocity), dx);
    check_against_fe_values<2, true>(form, Weights{});
  }
  SECTION("different coupling signs, multiple coefficients, and mass blocks")
  {
    Weights weights;
    weights.velocity_mass     = 0.4;
    weights.pressure_mass     = 0.7;
    weights.velocity_pressure = -2.3;
    weights.pressure_velocity = 0.8;
    const auto form =
      integral(coefficient(0.7) * test_pressure * pressure +
                 coefficient(0.8) * div(velocity) * test_pressure -
                 coefficient(2.3) * pressure * div(test_velocity) +
                 coefficient(0.4) * inner(velocity, test_velocity) +
                 inner(sym(grad(velocity)), sym(grad(test_velocity))) *
                   (2 * mu),
               dx);
    check_against_fe_values<2, true>(form, weights);
  }
  SECTION("trial gradients and test values use different evaluation flags")
  {
    Weights weights;
    weights.mu                = 0.0;
    weights.velocity_pressure = 0.0;
    weights.pressure_velocity = 0.8;
    weights.pressure_gradient = 1.3;
    const auto form =
      integral(coefficient(1.3) * inner(test_velocity, grad(pressure)) +
                 coefficient(0.8) * test_pressure * div(velocity),
               dx);
    STATIC_REQUIRE(internal::FieldRequirements<decltype(form),
                                               decltype(velocity)>::gradient);
    STATIC_REQUIRE_FALSE(
      internal::FieldRequirements<decltype(form), decltype(velocity)>::value);
    STATIC_REQUIRE(internal::FieldRequirements<decltype(form),
                                               decltype(test_velocity)>::value);
    STATIC_REQUIRE_FALSE(
      internal::FieldRequirements<decltype(form),
                                  decltype(test_velocity)>::gradient);
    check_against_fe_values<2, true>(form, weights);
  }
}
