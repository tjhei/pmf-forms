#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>

#include <catch2/catch_test_macros.hpp>
#include <forms.h>
#include <matrix_free_operator.h>

namespace
{
  struct LinearDiffusion
  {
    double scale = 1.0;

    DEAL_II_HOST_DEVICE double
    operator()(const dealii::Point<2> &point) const
    {
      return scale * (1.0 + point[0]);
    }
  };

  template <typename Form, typename Provider, bool run_portable = true>
  void
  check_spatial_laplace(const Form &form, const Provider &provider)
  {
    constexpr int dim    = 2;
    constexpr int degree = 2;

    dealii::parallel::distributed::Triangulation<dim> triangulation(
      MPI_COMM_WORLD);
    dealii::GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(1);
    dealii::FE_Q<dim>       finite_element(degree);
    dealii::DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(finite_element);
    dealii::AffineConstraints<double> constraints;
    constraints.reinit(dof_handler.locally_owned_dofs(),
                       dealii::DoFTools::extract_locally_relevant_dofs(
                         dof_handler));
    if (constraints.n_constraints() == 0 &&
        dof_handler.locally_owned_dofs().is_element(0))
      constraints.add_line(0);
    constraints.close();

    dealii::MappingQ<dim> mapping(1);
    dealii::QGauss<1>     quadrature(degree + 1);
    auto                  cpu_data =
      std::make_shared<dealii::MatrixFree<dim, double>>();
    dealii::MatrixFree<dim, double>::AdditionalData cpu_settings;
    cpu_settings.mapping_update_flags = dealii::update_values |
                                        dealii::update_gradients |
                                        dealii::update_quadrature_points |
                                        dealii::update_JxW_values;
    cpu_data->reinit(mapping, dof_handler, constraints, quadrature, cpu_settings);
    const auto cpu = make_matrix_free_operator<dim, degree>(cpu_data, form);

    auto portable_data =
      std::make_shared<dealii::Portable::MatrixFree<dim, double>>();
    dealii::Portable::MatrixFree<dim, double>::AdditionalData portable_settings;
    portable_settings.mapping_update_flags = cpu_settings.mapping_update_flags;
    if constexpr (run_portable)
      portable_data->reinit(mapping,
                            dof_handler,
                            constraints,
                            quadrature,
                            portable_settings);
    dealii::LinearAlgebra::distributed::Vector<double> source, reference,
      cpu_result, portable_result;
    cpu.initialize_dof_vector(source);
    cpu.initialize_dof_vector(reference);
    cpu.initialize_dof_vector(cpu_result);
    cpu.initialize_dof_vector(portable_result);
    for (const auto index : source.locally_owned_elements())
      source[index] = std::sin(0.31 * (index + 1));
    constraints.set_zero(source);
    source.update_ghost_values();

    const dealii::QGauss<dim> reference_quadrature(degree + 2);
    dealii::FEValues<dim>     fe_values(mapping,
                                    finite_element,
                                    reference_quadrature,
                                    dealii::update_values |
                                      dealii::update_gradients |
                                      dealii::update_quadrature_points |
                                      dealii::update_JxW_values);
    const dealii::FEValuesExtractors::Scalar scalar(0);
    std::vector<dealii::types::global_dof_index> indices(
      finite_element.n_dofs_per_cell());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(indices);
          dealii::Vector<double> local_result(indices.size());
          for (unsigned int point = 0; point < reference_quadrature.size();
               ++point)
            {
              const auto   location = fe_values.quadrature_point(point);
              const double diffusion =
                pmf::forms::expression_templates::internal::
                  evaluate_coefficient_at_point<dim, double>(provider, location);
              double                 value = 0.0;
              dealii::Tensor<1, dim> gradient;
              for (unsigned int basis = 0; basis < indices.size(); ++basis)
                {
                  const double entry = source[indices[basis]];
                  value += entry * fe_values[scalar].value(basis, point);
                  gradient +=
                    entry * fe_values[scalar].gradient(basis, point);
                }
              for (unsigned int basis = 0; basis < indices.size(); ++basis)
                {
                  const double shape_value =
                    fe_values[scalar].value(basis, point);
                  const auto shape_gradient =
                    fe_values[scalar].gradient(basis, point);
                  local_result[basis] += diffusion *
                                         (shape_gradient * gradient) *
                                         fe_values.JxW(point);
                }
            }
          constraints.distribute_local_to_global(local_result,
                                                 indices,
                                                 reference);
        }
    reference.compress(dealii::VectorOperation::add);

    cpu.vmult(cpu_result, source);

    REQUIRE(reference.l2_norm() > 0.0);
    cpu_result -= reference;
    CHECK(cpu_result.l2_norm() / reference.l2_norm() < 1e-11);

    if constexpr (run_portable)
      {
        const auto portable =
          make_portable_matrix_free_operator<dim, degree>(portable_data, form);
        dealii::LinearAlgebra::distributed::Vector<double,
                                                     dealii::MemorySpace::Default>
          device_source, device_result;
        portable.initialize_dof_vector(device_source);
        portable.initialize_dof_vector(device_result);
        device_source.import_elements(source, dealii::VectorOperation::insert);
        portable.vmult(device_result, device_source);
        portable_result.import_elements(device_result,
                                        dealii::VectorOperation::insert);
        portable_result -= reference;
        CHECK(portable_result.l2_norm() / reference.l2_norm() < 1e-11);
      }
  }
} // namespace

TEST_CASE("Spatial Laplace coefficients match FEValues on CPU and Portable",
          "[forms][variable-coefficient][lowering]")
{
  using namespace pmf::forms;
  using namespace pmf::forms::expression_templates;

  const auto trial_field = trial();
  const auto test_field  = test();
  const LinearDiffusion diffusion{.scale = 2.5};
  const auto form =
    integral(coefficient(diffusion) * inner(grad(test_field), grad(trial_field)),
             dx);

  STATIC_REQUIRE(FormFields<decltype(form)>::n_coefficients == 1);
  check_spatial_laplace(form, diffusion);
}

TEST_CASE("FunctionCoefficient can drive a spatial Laplace operator",
          "[forms][variable-coefficient]")
{
  using namespace pmf::forms;
  using namespace pmf::forms::expression_templates;

  class DiffusionFunction : public dealii::Function<2>
  {
  public:
    double
    value(const dealii::Point<2> &point,
          const unsigned int = 0) const override
    {
      return 1.25 * (1.0 + point[1]);
    }
  };

  const auto trial_field = trial();
  const auto test_field  = test();
  DiffusionFunction      function;
  const auto             provider = FunctionCoefficient<2>(function);
  const auto form =
    integral(coefficient(provider) * inner(grad(test_field), grad(trial_field)),
             dx);

  check_spatial_laplace<decltype(form), FunctionCoefficient<2>, false>(form,
                                                                      provider);
}

TEST_CASE("Spatial coefficients appear in inspect_lowering output",
          "[forms][variable-coefficient][inspect]")
{
  using namespace pmf::forms;
  using namespace pmf::forms::expression_templates;

  const auto trial_field = trial();
  const auto test_field  = test();
  const LinearDiffusion diffusion{.scale = 1.0};
  const auto form =
    integral(coefficient(diffusion) * inner(grad(test_field), grad(trial_field)),
             dx);

  const auto kernel = inspect_lowering(form);
  unsigned int provider_count = 0;
  for (const auto &operation : kernel.operations)
    if (operation.opcode == LoweringOpcode::coefficient &&
        operation.literal == "<provider>")
      ++provider_count;
  CHECK(provider_count == 1);
}
