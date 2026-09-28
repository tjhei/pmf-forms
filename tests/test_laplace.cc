#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/vector_tools.h>

#include <catch2/catch_test_macros.hpp>
#include <forms.h>
#include <matrix_free_operator.h>

#include <memory>

namespace
{
  constexpr double diffusion_value = 2.5;

  class ExactSolution : public dealii::Function<2>
  {
  public:
    double
    value(const dealii::Point<2> &point, const unsigned int = 0) const override
    {
      return point[0] * (1.0 - point[0]) * point[1] * (1.0 - point[1]);
    }
  };

  class RightHandSide : public dealii::Function<2>
  {
  public:
    double
    value(const dealii::Point<2> &point, const unsigned int = 0) const override
    {
      return 2.0 * diffusion_value *
             (point[0] * (1.0 - point[0]) + point[1] * (1.0 - point[1]));
    }
  };

  template <typename Operator, typename MemorySpace>
  void
  solve_and_check(
    const Operator &matrix,
    const dealii::LinearAlgebra::distributed::Vector<double, MemorySpace> &rhs,
    const dealii::LinearAlgebra::distributed::Vector<double, MemorySpace>
                                                                    &exact,
    dealii::LinearAlgebra::distributed::Vector<double, MemorySpace> &solution)
  {
    using Vector =
      dealii::LinearAlgebra::distributed::Vector<double, MemorySpace>;
    matrix.initialize_dof_vector(solution);
    solution              = 0.0;
    const double rhs_norm = rhs.l2_norm();
    REQUIRE(rhs_norm > 0.0);
    dealii::SolverControl                control(400, 1e-12 * rhs_norm);
    dealii::SolverCG<Vector>             solver(control);
    dealii::PreconditionJacobi<Operator> jacobi;
    jacobi.initialize(matrix);
    solver.solve(matrix, solution, rhs, jacobi);
    const auto &inverse = matrix.get_inverse_diagonal();
    CHECK(&inverse == &matrix.get_inverse_diagonal());
    Vector diagonal_product;
    diagonal_product.reinit(inverse);
    diagonal_product = inverse;
    diagonal_product.scale(matrix.get_diagonal());
    diagonal_product.add(-1.0);
    CHECK(diagonal_product.linfty_norm() < 1e-14);
    CHECK(control.last_check() == dealii::SolverControl::success);
    CHECK(control.last_step() > 0);
    CHECK(control.last_step() < 400);

    Vector residual, error;
    matrix.initialize_dof_vector(residual);
    matrix.initialize_dof_vector(error);
    matrix.vmult(residual, solution);
    residual -= rhs;
    CHECK(residual.l2_norm() / rhs_norm < 1e-11);
    error = solution;
    error -= exact;
    REQUIRE(exact.l2_norm() > 0.0);
    CHECK(error.l2_norm() / exact.l2_norm() < 1e-10);
  }
} // namespace

TEST_CASE("Laplace manufactured solution is recovered by CG and Jacobi",
          "[forms][laplace][manufactured][solve]")
{
  constexpr int dim         = 2;
  constexpr int degree      = 2;
  constexpr int refinements = 3;
  using namespace pmf::forms;
  using namespace pmf::forms::expression_templates;
  const auto trial_field = trial();
  const auto test_field  = test();
  const auto form        = integral(coefficient(diffusion_value) *
                               inner(grad(test_field), grad(trial_field)),
                             dx);

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
  dealii::VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   ExactSolution{},
                                                   constraints);
  constraints.close();
  dealii::MappingQ<dim>   mapping(1);
  const dealii::QGauss<1> quadrature(degree + 1);

  auto cpu_data = std::make_shared<dealii::MatrixFree<dim, double>>();
  dealii::MatrixFree<dim, double>::AdditionalData cpu_settings;
  cpu_settings.mapping_update_flags =
    dealii::update_gradients | dealii::update_JxW_values;
  cpu_data->reinit(mapping, dof_handler, constraints, quadrature, cpu_settings);
  const auto cpu = make_matrix_free_operator<dim, degree>(cpu_data, form);
  dealii::LinearAlgebra::distributed::Vector<double> rhs, exact, cpu_solution,
    portable_solution;
  cpu.initialize_dof_vector(rhs);
  cpu.initialize_dof_vector(exact);
  cpu.initialize_dof_vector(portable_solution);
  dealii::VectorTools::create_right_hand_side(mapping,
                                              dof_handler,
                                              dealii::QGauss<dim>(degree + 2),
                                              RightHandSide{},
                                              rhs,
                                              constraints);
  dealii::VectorTools::interpolate(mapping,
                                   dof_handler,
                                   ExactSolution{},
                                   exact);
  constraints.distribute(exact);
  {
    INFO("CPU MatrixFree");
    solve_and_check(cpu, rhs, exact, cpu_solution);
  }

  auto portable_data =
    std::make_shared<dealii::Portable::MatrixFree<dim, double>>();
  dealii::Portable::MatrixFree<dim, double>::AdditionalData portable_settings;
  portable_settings.mapping_update_flags = cpu_settings.mapping_update_flags;
  portable_data->reinit(
    mapping, dof_handler, constraints, quadrature, portable_settings);
  const auto portable =
    make_portable_matrix_free_operator<dim, degree>(portable_data, form);
  dealii::LinearAlgebra::distributed::Vector<double,
                                             dealii::MemorySpace::Default>
    device_rhs, device_exact, device_solution;
  portable.initialize_dof_vector(device_rhs);
  portable.initialize_dof_vector(device_exact);
  device_rhs.import_elements(rhs, dealii::VectorOperation::insert);
  device_exact.import_elements(exact, dealii::VectorOperation::insert);
  {
    INFO("Portable MatrixFree");
    solve_and_check(portable, device_rhs, device_exact, device_solution);
  }
  portable_solution.import_elements(device_solution,
                                    dealii::VectorOperation::insert);
  portable_solution -= cpu_solution;
  CHECK(portable_solution.l2_norm() / exact.l2_norm() < 1e-10);
}
