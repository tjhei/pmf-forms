// -----------------------------------------------------------------------------
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// -----------------------------------------------------------------------------

// Linear elasticity with Portable::MatrixFree.

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

using namespace dealii;

namespace
{
  // Love's surface-pressure benchmark, in SI units.
  constexpr double domain_length_x            = 5.0e3;
  constexpr double domain_length_y            = 5.0e3;
  constexpr double domain_height              = 2.5e3;
  constexpr double load_half_width_x          = 0.5555e3;
  constexpr double load_half_width_y          = 1.1111e3;
  constexpr double water_density              = 1000.0;
  constexpr double water_depth                = 100.0;
  constexpr double gravitational_acceleration = 9.82;
  constexpr double pressure =
    water_density * water_depth * gravitational_acceleration;
  constexpr double             lame_lambda                 = 24.0e9;
  constexpr double             lame_mu                     = 24.0e9;
  constexpr types::boundary_id top_boundary_id             = 5;
  constexpr types::boundary_id first_dirichlet_boundary_id = 0;
  constexpr types::boundary_id last_dirichlet_boundary_id  = 4;

  double
  rj0(const unsigned int j,
      const double       x,
      const double       y,
      const double       yy,
      const double       z,
      const double       a)
  {
    const double dx = j == 1 ? a - x : a + x;
    return std::sqrt(dx * dx + (yy - y) * (yy - y) + z * z);
  }

  double
  r0j(const unsigned int j,
      const double       x,
      const double       xx,
      const double       y,
      const double       z,
      const double       b)
  {
    const double dy = j == 1 ? b - y : b + y;
    return std::sqrt((xx - x) * (xx - x) + dy * dy + z * z);
  }

  double
  betaj0(const unsigned int j, const double x, const double z, const double a)
  {
    const double dx = j == 1 ? a - x : a + x;
    return std::sqrt(dx * dx + z * z);
  }

  double
  beta0j(const unsigned int j, const double y, const double z, const double b)
  {
    const double dy = j == 1 ? b - y : b + y;
    return std::sqrt(dy * dy + z * z);
  }

  double
  psi_j0(const unsigned int j,
         const double       x,
         const double       y,
         const double       yy,
         const double       z,
         const double       a)
  {
    return (yy - y) / (rj0(j, x, y, yy, z, a) + betaj0(j, x, z, a));
  }

  double
  psi_0j(const unsigned int j,
         const double       x,
         const double       xx,
         const double       y,
         const double       z,
         const double       b)
  {
    return (xx - x) / (r0j(j, x, xx, y, z, b) + beta0j(j, y, z, b));
  }

  double
  Jj(const unsigned int j,
     const double       x,
     const double       y,
     const double       yy,
     const double       z,
     const double       a)
  {
    const double psi = psi_j0(j, x, y, yy, z, a);
    const double ax  = std::abs(j == 1 ? a - x : a + x);
    const double term_1 =
      (yy - y) * (std::log(z + rj0(j, x, y, yy, z, a)) - 1.0);
    const double term_2 =
      z == 0.0 ? 0.0 : z * std::log((1.0 + psi) / (1.0 - psi));
    const double term_3 =
      2.0 * ax * std::atan(ax * psi / (z + betaj0(j, x, z, a)));
    return term_1 + term_2 + term_3;
  }

  double
  Kj(const unsigned int j,
     const double       x,
     const double       xx,
     const double       y,
     const double       z,
     const double       b)
  {
    const double psi = psi_0j(j, x, xx, y, z, b);
    const double by  = std::abs(j == 1 ? b - y : b + y);
    return (xx - x) * (std::log(z + r0j(j, x, xx, y, z, b)) - 1.0) +
           z * std::log((1.0 + psi) / (1.0 - psi)) +
           2.0 * by * std::atan(by * psi / (z + beta0j(j, y, z, b)));
  }

  double
  Lj(const unsigned int j,
     const double       x,
     const double       y,
     const double       yy,
     const double       z,
     const double       a)
  {
    const double psi = psi_j0(j, x, y, yy, z, a);
    const double ax  = j == 1 ? a - x : -a - x;
    const double term_3 =
      z == 0.0 ? 0.0 : 2.0 * z * std::atan(z * psi / (ax + betaj0(j, x, z, a)));
    return (yy - y) * std::log(ax + rj0(j, x, y, yy, z, a)) +
           ax * std::log((1.0 + psi) / (1.0 - psi)) + term_3;
  }

  double
  love_u(const double x, const double y, const double z)
  {
    const auto term = [=](const double yy) {
      const double logarithmic_term =
        z == 0.0 ?
          0.0 :
          z / lame_mu *
            std::log((yy - y + rj0(2, x, y, yy, z, load_half_width_x)) /
                     (yy - y + rj0(1, x, y, yy, z, load_half_width_x)));
      return -pressure / (4.0 * numbers::PI) *
             ((Jj(2, x, y, yy, z, load_half_width_x) -
               Jj(1, x, y, yy, z, load_half_width_x)) /
                (lame_lambda + lame_mu) +
              logarithmic_term);
    };
    return term(load_half_width_y) - term(-load_half_width_y);
  }

  double
  love_v(const double x, const double y, const double z)
  {
    const auto term = [=](const double xx) {
      return -pressure / (4.0 * numbers::PI) *
             ((Kj(2, x, xx, y, z, load_half_width_y) -
               Kj(1, x, xx, y, z, load_half_width_y)) /
                (lame_lambda + lame_mu) +
              z / lame_mu *
                std::log((xx - x + r0j(2, x, xx, y, z, load_half_width_y)) /
                         (xx - x + r0j(1, x, xx, y, z, load_half_width_y))));
    };
    return term(load_half_width_x) - term(-load_half_width_x);
  }

  double
  love_w(const double x, const double y, const double z)
  {
    const auto term = [=](const double yy) {
      const double arctangent_term =
        z == 0.0 ?
          0.0 :
          z * (std::atan((load_half_width_x - x) * (yy - y) /
                         (z * rj0(1, x, y, yy, z, load_half_width_x))) +
               std::atan((load_half_width_x + x) * (yy - y) /
                         (z * rj0(2, x, y, yy, z, load_half_width_x))));
      return pressure / (4.0 * numbers::PI * lame_mu) *
             ((lame_lambda + 2.0 * lame_mu) / (lame_lambda + lame_mu) *
                (Lj(1, x, y, yy, z, load_half_width_x) -
                 Lj(2, x, y, yy, z, load_half_width_x)) +
              arctangent_term);
    };
    return term(load_half_width_y) - term(-load_half_width_y);
  }

  template <int dim>
  class LoveSolution : public Function<dim>
  {
  public:
    LoveSolution()
      : Function<dim>(dim)
    {}

    double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      AssertIndexRange(component, dim);
      AssertDimension(dim, 3);
      const double depth = domain_height - p[2];
      if (component == 0)
        return love_u(p[0], p[1], depth);
      else if (component == 1)
        return love_v(p[0], p[1], depth);
      else
        return -love_w(p[0], p[1], depth);
    }
  };

  template <int dim>
  class SurfacePressure : public Function<dim>
  {
  public:
    SurfacePressure()
      : Function<dim>(dim)
    {}

    double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      AssertIndexRange(component, dim);
      return component == dim - 1 && p[0] <= load_half_width_x &&
                 p[1] <= load_half_width_y ?
               -pressure :
               0.0;
    }
  };
} // namespace

template <int dim, int fe_degree, typename Number>
class ElasticityOperatorQuad
{
public:
  DEAL_II_HOST_DEVICE void
  operator()(
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> *fe_eval,
    const int q_point) const
  {
    const Tensor<2, dim, Number> gradient_u = fe_eval->get_gradient(q_point);
    Tensor<2, dim, Number>       stress =
      lame_mu * (gradient_u + transpose(gradient_u));
    const Number volumetric_stress = lame_lambda * trace(gradient_u);
    for (unsigned int d = 0; d < dim; ++d)
      stress[d][d] += volumetric_stress;

    fe_eval->submit_gradient(stress, q_point);
  }
};

template <int dim, int fe_degree, typename Number, int n_q_points_1d>
class ElasticityCellOperator
{
public:
  static const unsigned int n_q_points =
    dealii::Utilities::pow(n_q_points_1d, dim);

  DEAL_II_HOST_DEVICE void
  operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
             const Portable::DeviceVector<Number>                   &src,
             Portable::DeviceVector<Number>                         &dst) const
  {
    Portable::FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> fe_u(
      data, 0);

    fe_u.read_dof_values(src);
    fe_u.evaluate(EvaluationFlags::gradients);

    ElasticityOperatorQuad<dim, fe_degree, Number> quad_operation;
    data->for_each_quad_point(
      [&](const int q_point) { quad_operation(&fe_u, q_point); });

    fe_u.integrate(EvaluationFlags::gradients);
    fe_u.distribute_local_to_global(dst);
  }
};

template <int dim,
          int fe_degree,
          typename Number = double,
          typename VectorType =
            LinearAlgebra::distributed::Vector<double, MemorySpace::Default>,
          int n_q_points_1d = fe_degree + 1>
class PortableMFElasticityOperator : public EnableObserverPointer
{
public:
  PortableMFElasticityOperator() = default;

  explicit PortableMFElasticityOperator(
    std::shared_ptr<Portable::MatrixFree<dim, Number>> data_in)
    : data(std::move(data_in))
  {}

  void
  reinit(std::shared_ptr<Portable::MatrixFree<dim, Number>> data_in)
  {
    data = std::move(data_in);
  }

  void
  initialize_dof_vector(VectorType &vec) const
  {
    data->initialize_dof_vector(vec, 0);
  }

  types::global_dof_index
  m() const
  {
    return data->get_vector_partitioner(0)->size();
  }

  double
  el(const types::global_dof_index row, const types::global_dof_index col) const
  {
    Assert(row == col, ExcNotImplemented());
    Assert(inverse_diagonal_entries.get() != nullptr &&
             inverse_diagonal_entries->m() > 0,
           ExcNotInitialized());
    return 1.0 / (*inverse_diagonal_entries)(row, row);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    dst = static_cast<Number>(0.);
    ElasticityCellOperator<dim, fe_degree, Number, n_q_points_1d>
      elasticity_operator;
    data->cell_loop(elasticity_operator, src, dst);
    data->copy_constrained_values(src, dst, 0);
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  std::shared_ptr<DiagonalMatrix<VectorType>>
  get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }

  void
  compute_diagonal()
  {
    Assert(data.get() != nullptr, ExcNotInitialized());

    inverse_diagonal_entries = std::make_shared<DiagonalMatrix<VectorType>>();
    VectorType &inverse_diagonal = inverse_diagonal_entries->get_vector();
    data->initialize_dof_vector(inverse_diagonal, 0);

    ElasticityOperatorQuad<dim, fe_degree, Number> quad_operation;
    MatrixFreeTools::
      compute_diagonal<dim, fe_degree, fe_degree + 1, dim, Number>(
        *data,
        inverse_diagonal,
        quad_operation,
        EvaluationFlags::gradients,
        EvaluationFlags::gradients,
        0);

    double *raw_diagonal = inverse_diagonal.get_values();
    Kokkos::parallel_for(
      "invert diagonal",
      inverse_diagonal.locally_owned_size(),
      KOKKOS_LAMBDA(const int i) {
        Assert(raw_diagonal[i] > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        raw_diagonal[i] = 1. / raw_diagonal[i];
      });
  }

private:
  std::shared_ptr<Portable::MatrixFree<dim, Number>> data;
  std::shared_ptr<DiagonalMatrix<VectorType>>        inverse_diagonal_entries;
};

template <int dim>
class StressPostprocessor : public DataPostprocessor<dim>
{
public:
  std::vector<std::string>
  get_names() const override
  {
    if constexpr (dim == 2)
      return {"sigma_xx", "sigma_yy", "sigma_xy"};
    else
      return {
        "sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_xz", "sigma_yz"};
  }

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override
  {
    return std::vector<
      DataComponentInterpretation::DataComponentInterpretation>(
      get_names().size(), DataComponentInterpretation::component_is_scalar);
  }

  UpdateFlags
  get_needed_update_flags() const override
  {
    return update_gradients;
  }

  void
  evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const override
  {
    AssertDimension(computed_quantities.size(),
                    inputs.solution_gradients.size());

    for (unsigned int q_point = 0; q_point < computed_quantities.size();
         ++q_point)
      {
        Tensor<2, dim> gradient_u;
        for (unsigned int component = 0; component < dim; ++component)
          gradient_u[component] = inputs.solution_gradients[q_point][component];
        Tensor<2, dim> stress = lame_mu * (gradient_u + transpose(gradient_u));
        const double   volumetric_stress = lame_lambda * trace(gradient_u);
        for (unsigned int d = 0; d < dim; ++d)
          stress[d][d] += volumetric_stress;

        computed_quantities[q_point][0] = stress[0][0];
        computed_quantities[q_point][1] = stress[1][1];
        if constexpr (dim == 2)
          computed_quantities[q_point][2] = stress[0][1];
        else
          {
            computed_quantities[q_point][2] = stress[2][2];
            computed_quantities[q_point][3] = stress[0][1];
            computed_quantities[q_point][4] = stress[0][2];
            computed_quantities[q_point][5] = stress[1][2];
          }
      }
  }
};

template <int dim, int fe_degree, typename Number = double>
class ElasticityProblem
{
public:
  using VectorType =
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

  ElasticityProblem();

  void
  run();

private:
  void
  setup_dofs();

  void
  solve();

  void
  postprocess(const unsigned int refinement);

  parallel::distributed::Triangulation<dim> tria;
  MappingQ<dim>                             mapping;
  FESystem<dim>                             fe;
  DoFHandler<dim>                           dof_handler;
  AffineConstraints<double>                 constraints;

  std::shared_ptr<Portable::MatrixFree<dim, Number>> mf_data;
  VectorType                                         solution;
  VectorType                                         rhs;
  ConditionalOStream                                 pcout;
};

template <int dim, int fe_degree, typename Number>
ElasticityProblem<dim, fe_degree, Number>::ElasticityProblem()
  : tria(MPI_COMM_WORLD)
  , mapping(1)
  , fe(FE_Q<dim>(fe_degree), dim)
  , dof_handler(tria)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{}

template <int dim, int fe_degree, typename Number>
void
ElasticityProblem<dim, fe_degree, Number>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);
  constraints.reinit(dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler));
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  for (types::boundary_id boundary_id = first_dirichlet_boundary_id;
       boundary_id <= last_dirichlet_boundary_id;
       ++boundary_id)
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_id,
                                             LoveSolution<dim>(),
                                             constraints);
  constraints.close();

  mf_data = std::make_shared<Portable::MatrixFree<dim, Number>>();
  const QGauss<1> quad(fe_degree + 1);
  typename Portable::MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.mapping_update_flags = update_values | update_gradients;
  mf_data->reinit(mapping, dof_handler, constraints, quad, additional_data);

  LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> rhs_host;
  mf_data->initialize_dof_vector(rhs_host);

  FEFaceValues<dim>                    face_values(mapping,
                                fe,
                                QGauss<dim - 1>(fe_degree + 1),
                                update_values | update_quadrature_points |
                                  update_JxW_values);
  std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
  Vector<Number>                       cell_rhs(fe.n_dofs_per_cell());

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const unsigned int face : cell->face_indices())
        if (cell->face(face)->at_boundary() &&
            cell->face(face)->boundary_id() == top_boundary_id)
          {
            face_values.reinit(cell, face);
            cell_rhs = 0.0;
            for (unsigned int q_point = 0;
                 q_point < face_values.n_quadrature_points;
                 ++q_point)
              {
                const Point<dim> &point = face_values.quadrature_point(q_point);
                if (point[0] <= load_half_width_x &&
                    point[1] <= load_half_width_y)
                  for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                    if (fe.system_to_component_index(i).first == dim - 1)
                      cell_rhs(i) -= pressure *
                                     face_values.shape_value(i, q_point) *
                                     face_values.JxW(q_point);
              }

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   rhs_host);
          }
  rhs_host.compress(VectorOperation::add);
  constraints.distribute(rhs_host);

  mf_data->initialize_dof_vector(rhs);
  rhs.import_elements(rhs_host, VectorOperation::insert);
}

template <int dim, int fe_degree, typename Number>
void
ElasticityProblem<dim, fe_degree, Number>::solve()
{
  using LevelMatrixType = PortableMFElasticityOperator<dim, fe_degree>;
  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                                           VectorType,
                                                           SmootherPreconditionerType>;
  using MGTransferType =
    MGTransferMatrixFree<dim, Number, MemorySpace::Default>;

  LevelMatrixType elasticity_operator(mf_data);

  // Convert the inhomogeneous Dirichlet problem A u = b into the homogeneous
  // correction problem A v = b - A g, where u = v + g and g contains the
  // prescribed Love displacement. This keeps the CG operator symmetric on
  // the constrained subspace.
  VectorType lifting;
  VectorType operator_lifting;
  mf_data->initialize_dof_vector(lifting);
  mf_data->initialize_dof_vector(operator_lifting);
  LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> lifting_host;
  mf_data->initialize_dof_vector(lifting_host);
  lifting_host = 0.0;
  constraints.distribute(lifting_host);
  lifting.import_elements(lifting_host, VectorOperation::insert);
  elasticity_operator.vmult(operator_lifting, lifting);
  rhs -= operator_lifting;
  mf_data->set_constrained_values(0.0, rhs, 0);

  mf_data->initialize_dof_vector(solution);
  {
    Kokkos::Timer t;
    elasticity_operator.vmult(solution, rhs);
    const double time = t.seconds();
    pcout << "operator time: " << time << " dofs/s: " << solution.size() / time
          << std::endl;
    solution = 0.0;
  }

  const auto coarse_grid_triangulations =
    MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(tria);
  const unsigned int max_level = coarse_grid_triangulations.size() - 1;
  // Skip the very small coarsest levels when possible, but retain level zero
  // for shallow hierarchies instead of underflowing max_level - 1.
  const unsigned int min_level = max_level > 2 ? 2U : 0U;

  MGLevelObject<DoFHandler<dim>> mg_dof_handlers(min_level, max_level);
  MGLevelObject<AffineConstraints<Number>> mg_constraints(min_level, max_level);
  MGLevelObject<LevelMatrixType>           mg_matrices(min_level, max_level);
  MGLevelObject<MGTwoLevelTransferCopyToHost<dim, VectorType>> mg_transfers(
    min_level, max_level);
  std::vector<std::shared_ptr<Portable::MatrixFree<dim, Number>>>
    mf_data_levels;

  for (unsigned int level = min_level; level <= max_level; ++level)
    {
      auto &level_dof_handler = mg_dof_handlers[level];
      auto &level_constraints = mg_constraints[level];
      level_dof_handler.reinit(*coarse_grid_triangulations[level]);
      level_dof_handler.distribute_dofs(fe);
      level_constraints.reinit(level_dof_handler.locally_owned_dofs(),
                               DoFTools::extract_locally_relevant_dofs(
                                 level_dof_handler));
      for (types::boundary_id boundary_id = first_dirichlet_boundary_id;
           boundary_id <= last_dirichlet_boundary_id;
           ++boundary_id)
        VectorTools::interpolate_boundary_values(level_dof_handler,
                                                 boundary_id,
                                                 Functions::ZeroFunction<dim>(
                                                   dim),
                                                 level_constraints);
      level_constraints.close();

      if (level == max_level)
        mf_data_levels.emplace_back(mf_data);
      else
        {
          typename Portable::MatrixFree<dim, Number>::AdditionalData
            additional_data;
          additional_data.mapping_update_flags =
            update_JxW_values | update_gradients;
          auto level_data =
            std::make_shared<Portable::MatrixFree<dim, Number>>();
          level_data->reinit(mapping,
                             level_dof_handler,
                             level_constraints,
                             QGauss<1>(fe_degree + 1),
                             additional_data);
          mf_data_levels.emplace_back(std::move(level_data));
        }
      mg_matrices[level].reinit(mf_data_levels.back());
    }

  for (unsigned int level = min_level; level < max_level; ++level)
    mg_transfers[level + 1].reinit(mg_dof_handlers[level + 1],
                                   mg_dof_handlers[level],
                                   mg_constraints[level + 1],
                                   mg_constraints[level]);

  MGTransferType mg_transfer(mg_transfers, [&](const auto level, auto &vector) {
    mg_matrices[level].initialize_dof_vector(vector);
  });
  mg::Matrix<VectorType> mg_matrix(mg_matrices);

  MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level,
                                                                     max_level);
  for (unsigned int level = min_level; level <= max_level; ++level)
    {
      mg_matrices[level].compute_diagonal();
      smoother_data[level].preconditioner =
        std::make_shared<SmootherPreconditionerType>(
          *mg_matrices[level].get_matrix_diagonal_inverse());
      smoother_data[level].smoothing_range     = 20;
      smoother_data[level].degree              = 5;
      smoother_data[level].eig_cg_n_iterations = 20;
      smoother_data[level].constraints.copy_from(mg_constraints[level]);
    }

  MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;
  mg_smoother.initialize(mg_matrices, smoother_data);
  MGCoarseGridApplySmoother<VectorType> mg_coarse;
  mg_coarse.initialize(mg_smoother);
  Multigrid<VectorType>                           mg(mg_matrix,
                           mg_coarse,
                           mg_transfer,
                           mg_smoother,
                           mg_smoother,
                           min_level,
                           max_level);
  PreconditionMG<dim, VectorType, MGTransferType> preconditioner(dof_handler,
                                                                 mg,
                                                                 mg_transfer);

  SolverControl        solver_control(100, 1e-7 * rhs.l2_norm());
  SolverCG<VectorType> solver(solver_control);
  Kokkos::Timer        t;
  solver.solve(elasticity_operator, solution, rhs, preconditioner);
  solution += lifting;
  pcout << "converged in " << solver_control.last_step() << " iterations in "
        << t.seconds() << " seconds" << std::endl;
}

template <int dim, int fe_degree, typename Number>
void
ElasticityProblem<dim, fe_degree, Number>::postprocess(
  const unsigned int refinement)
{
  LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> solution_host;
  mf_data->initialize_dof_vector(solution_host);
  solution_host.import_elements(solution, VectorOperation::insert);
  constraints.distribute(solution_host);
  solution_host.update_ghost_values();

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector(solution_host,
                           "displacement",
                           DataOut<dim>::type_dof_data,
                           component_interpretation);
  StressPostprocessor<dim> stress_postprocessor;
  data_out.add_data_vector(solution_host, stress_postprocessor);
  data_out.build_patches(mapping, fe_degree);
  data_out.write_vtu_with_pvtu_record(
    "./", "love", refinement, MPI_COMM_WORLD, 3);
}

template <int dim, int fe_degree, typename Number>
void
ElasticityProblem<dim, fe_degree, Number>::run()
{
  static_assert(dim == 3, "This example currently runs in three dimensions.");
  pcout << std::setprecision(10);
  pcout << "Running on " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
        << " MPI ranks and " << MultithreadInfo::n_threads() << " threads in "
#ifdef DEBUG
        << "DEBUG mode" << std::endl
#else
        << "RELEASE mode" << std::endl
#endif
        << "dim: " << dim << std::endl
        << "Element: Q" << fe_degree << std::endl
        << "Love benchmark (SI units): E = " << 0.6e11
        << " Pa, nu = 0.25, lambda = " << lame_lambda << " Pa, mu = " << lame_mu
        << " Pa, pressure = " << pressure << " Pa" << std::endl;

  constexpr unsigned int n_refinements = 2;
  for (unsigned int refinement = 0; refinement < n_refinements; ++refinement)
    {
      if (refinement == 0)
        {
          GridGenerator::subdivided_hyper_rectangle(
            tria,
            std::vector<unsigned int>{2, 2, 1},
            Point<dim>(0.0, 0.0, 0.0),
            Point<dim>(domain_length_x, domain_length_y, domain_height),
            true);

          tria.refine_global(3);
        }
      else
        tria.refine_global(1);

      setup_dofs();
      pcout << "\nrefinement: " << refinement
            << ", n_dofs: " << dof_handler.n_dofs() << std::endl;
      mf_data->initialize_dof_vector(solution);
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>
        boundary_displacement;
      mf_data->initialize_dof_vector(boundary_displacement);
      boundary_displacement = 0.0;
      constraints.distribute(boundary_displacement);
      solution.import_elements(boundary_displacement, VectorOperation::insert);
      postprocess(0);
      solution = 0.0;
      solve();
      postprocess(refinement + 1);
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  ElasticityProblem<3, 2>          problem;
  problem.run();
}
