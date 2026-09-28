#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <matrix_free_operator.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
  constexpr int dim             = 3;
  constexpr int velocity_degree = 2;
  constexpr int pressure_degree = 1;
  constexpr int quadrature_size = 3;
  using HostVector   = dealii::LinearAlgebra::distributed::BlockVector<double>;
  using DeviceVector = dealii::LinearAlgebra::distributed::
    BlockVector<double, dealii::MemorySpace::Default>;
  using CpuData      = dealii::MatrixFree<dim, double>;
  using PortableData = dealii::Portable::MatrixFree<dim, double>;

  template <typename VelocityEvaluation, typename PressureEvaluation>
  DEAL_II_HOST_DEVICE void
  step104_point(VelocityEvaluation &velocity,
                PressureEvaluation &pressure,
                const unsigned int  point)
  {
    const auto gradient       = velocity.get_gradient(point);
    const auto pressure_value = pressure.get_value(point);
    auto       velocity_term  = gradient;
    for (unsigned int direction = 0; direction < dim; ++direction)
      velocity_term[direction][direction] -= pressure_value;
    velocity.submit_gradient(velocity_term, point);
    pressure.submit_value(-dealii::trace(gradient), point);
  }

  struct Step104PortableCell
  {
    static constexpr unsigned int n_q_points =
      dealii::Utilities::pow(quadrature_size, dim);

    DEAL_II_HOST_DEVICE void
    operator()(const PortableData::Data                          *data,
               const dealii::Portable::DeviceBlockVector<double> &source,
               dealii::Portable::DeviceBlockVector<double> &destination) const
    {
      dealii::Portable::FEEvaluation<dim, velocity_degree, quadrature_size, dim>
        velocity(data, 0);
      dealii::Portable::FEEvaluation<dim, pressure_degree, quadrature_size, 1>
        pressure(data, 1);
      velocity.read_dof_values(source.block(0));
      pressure.read_dof_values(source.block(1));
      velocity.evaluate(dealii::EvaluationFlags::gradients);
      pressure.evaluate(dealii::EvaluationFlags::values);
      data->for_each_quad_point(
        [&](const int point) { step104_point(velocity, pressure, point); });
      velocity.integrate(dealii::EvaluationFlags::gradients);
      pressure.integrate(dealii::EvaluationFlags::values);
      velocity.distribute_local_to_global(destination.block(0));
      pressure.distribute_local_to_global(destination.block(1));
    }
  };

  class Step104PortableOperator
  {
  public:
    explicit Step104PortableOperator(std::shared_ptr<PortableData> data)
      : data(std::move(data))
    {}

    void
    vmult(DeviceVector &destination, const DeviceVector &source) const
    {
      destination = 0.0;
      data->cell_loop(Step104PortableCell{}, source, destination);
      data->copy_constrained_values(source, destination);
    }

  private:
    std::shared_ptr<PortableData> data;
  };

  class Step104CpuOperator
    : public dealii::MatrixFreeOperators::Base<dim, HostVector>
  {
  public:
    explicit Step104CpuOperator(std::shared_ptr<CpuData> data)
    {
      initialize(std::move(data));
    }

    void
    compute_diagonal() override
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "The timing reference does not compute a diagonal."));
    }

  private:
    void
    apply_add(HostVector &destination, const HostVector &source) const override
    {
      data->cell_loop(&Step104CpuOperator::local_apply,
                      this,
                      destination,
                      source);
    }

    void
    local_apply(const CpuData                               &matrix_free,
                HostVector                                  &destination,
                const HostVector                            &source,
                const std::pair<unsigned int, unsigned int> &range) const
    {
      dealii::FEEvaluation<dim, velocity_degree, quadrature_size, dim> velocity(
        matrix_free, 0);
      dealii::FEEvaluation<dim, pressure_degree, quadrature_size, 1> pressure(
        matrix_free, 1);
      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          velocity.reinit(cell);
          pressure.reinit(cell);
          velocity.read_dof_values(source.block(0));
          pressure.read_dof_values(source.block(1));
          velocity.evaluate(dealii::EvaluationFlags::gradients);
          pressure.evaluate(dealii::EvaluationFlags::values);
          for (unsigned int point = 0; point < velocity.n_q_points; ++point)
            step104_point(velocity, pressure, point);
          velocity.integrate(dealii::EvaluationFlags::gradients);
          pressure.integrate(dealii::EvaluationFlags::values);
          velocity.distribute_local_to_global(destination.block(0));
          pressure.distribute_local_to_global(destination.block(1));
        }
    }
  };

  template <typename Destination, typename Source>
  void
  copy(Destination &destination, const Source &source)
  {
    for (unsigned int block = 0; block < 2; ++block)
      destination.block(block).import_elements(source.block(block),
                                               dealii::VectorOperation::insert);
  }

  template <typename Operator, typename Vector>
  double
  measure(const Operator    &matrix,
          Vector            &destination,
          const Vector      &source,
          const unsigned int repetitions)
  {
    Kokkos::fence();
    AssertThrowMPI(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    for (unsigned int repetition = 0; repetition < repetitions; ++repetition)
      matrix.vmult(destination, source);
    Kokkos::fence();
    const double elapsed = MPI_Wtime() - start;
    return dealii::Utilities::MPI::max(elapsed, MPI_COMM_WORLD) / repetitions;
  }

  template <typename Generic, typename Reference, typename Vector>
  void
  measure_pair(const Generic       &generic,
               const Reference     &reference,
               Vector              &destination,
               const Vector        &source,
               const unsigned int   repetitions,
               const bool           reverse,
               std::vector<double> &generic_times,
               std::vector<double> &reference_times)
  {
    if (reverse)
      reference_times.push_back(
        measure(reference, destination, source, repetitions));
    generic_times.push_back(measure(generic, destination, source, repetitions));
    if (!reverse)
      reference_times.push_back(
        measure(reference, destination, source, repetitions));
  }

  double
  report(dealii::ConditionalOStream           &output,
         const std::string                    &name,
         std::vector<double>                   times,
         const dealii::types::global_dof_index dofs)
  {
    std::sort(times.begin(), times.end());
    const double median =
      0.5 * (times[(times.size() - 1) / 2] + times[times.size() / 2]);
    output << name << ": " << median << " s/vmult, DoFs/s: " << dofs / median
           << ", MDoFs/s: " << 1e-6 * dofs / median << ", sample range: ["
           << times.front() << ", " << times.back() << "] s\n";
    return median;
  }

  unsigned int
  argument(const char *text)
  {
    const std::string value(text);
    AssertThrow(!value.empty() &&
                  value.find_first_not_of("0123456789") == std::string::npos,
                dealii::ExcMessage("Arguments must be positive integers."));
    const auto number = std::stoul(value);
    AssertThrow(number > 0 &&
                  number <= std::numeric_limits<unsigned int>::max(),
                dealii::ExcMessage("Argument out of range."));
    return static_cast<unsigned int>(number);
  }

  void
  run(const unsigned int max_refinement,
      const unsigned int repetitions,
      const unsigned int samples)
  {
    using namespace pmf::forms;
    using namespace pmf::forms::expression_templates;
    const auto [velocity, pressure] =
      trial_functions<ValueShape::vector, ValueShape::scalar>();
    const auto [test_velocity, test_pressure] =
      test_functions<ValueShape::vector, ValueShape::scalar>();
    const auto form = integral(inner(grad(test_velocity), grad(velocity)), dx) -
                      integral(div(test_velocity) * pressure, dx) -
                      integral(test_pressure * div(velocity), dx);

    dealii::ConditionalOStream output(
      std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    output << std::setprecision(8)
           << "3D Q2-Q1 Stokes, full velocity gradient (step-104)\n"
           << "MPI ranks: "
           << dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
           << ", threads: " << dealii::MultithreadInfo::n_threads()
           << ", Kokkos: " << Kokkos::DefaultExecutionSpace::name() << '\n'
           << "Build: "
           << (dealii::running_in_debug_mode() ?
                 "Debug (not suitable for timings)" :
                 "Release")
           << '\n'
           << "Warmups: 2, repetitions/sample: " << repetitions
           << ", samples: " << samples << '\n';
    dealii::parallel::distributed::Triangulation<dim> triangulation(
      MPI_COMM_WORLD);
    dealii::GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(2);
    for (unsigned int refinement = 2; refinement <= max_refinement;
         ++refinement)
      {
        if (refinement > 2)
          triangulation.refine_global(1);
        dealii::FESystem<dim>   velocity_fe(dealii::FE_Q<dim>(velocity_degree),
                                          dim);
        dealii::FE_Q<dim>       pressure_fe(pressure_degree);
        dealii::DoFHandler<dim> velocity_dofs(triangulation),
          pressure_dofs(triangulation);
        velocity_dofs.distribute_dofs(velocity_fe);
        pressure_dofs.distribute_dofs(pressure_fe);
        dealii::AffineConstraints<double> velocity_constraints,
          pressure_constraints;
        velocity_constraints.reinit(
          velocity_dofs.locally_owned_dofs(),
          dealii::DoFTools::extract_locally_relevant_dofs(velocity_dofs));
        dealii::DoFTools::make_hanging_node_constraints(velocity_dofs,
                                                        velocity_constraints);
        dealii::VectorTools::interpolate_boundary_values(
          velocity_dofs,
          0,
          dealii::Functions::ZeroFunction<dim>(dim),
          velocity_constraints);
        velocity_constraints.close();
        pressure_constraints.reinit(
          pressure_dofs.locally_owned_dofs(),
          dealii::DoFTools::extract_locally_relevant_dofs(pressure_dofs));
        dealii::DoFTools::make_hanging_node_constraints(pressure_dofs,
                                                        pressure_constraints);
        pressure_constraints.close();
        const std::vector<const dealii::DoFHandler<dim> *> dof_handlers{
          &velocity_dofs, &pressure_dofs};
        const std::vector<const dealii::AffineConstraints<double> *>
          constraints{&velocity_constraints, &pressure_constraints};
        const dealii::MappingQ<dim> mapping(1);
        const dealii::QGauss<1>     quadrature(quadrature_size);
        auto                        cpu_data = std::make_shared<CpuData>();
        CpuData::AdditionalData     cpu_settings;
        cpu_settings.mapping_update_flags = dealii::update_values |
                                            dealii::update_gradients |
                                            dealii::update_JxW_values;
        cpu_data->reinit(
          mapping, dof_handlers, constraints, quadrature, cpu_settings);
        auto portable_data = std::make_shared<PortableData>();
        PortableData::AdditionalData portable_settings;
        portable_settings.mapping_update_flags =
          dealii::update_values | dealii::update_gradients;
        portable_data->reinit(
          mapping, dof_handlers, constraints, quadrature, portable_settings);
        const auto cpu =
          make_matrix_free_operator<dim, velocity_degree, pressure_degree>(
            cpu_data, form);
        const auto portable =
          make_portable_matrix_free_operator<dim,
                                             velocity_degree,
                                             pressure_degree>(portable_data,
                                                              form);
        const Step104CpuOperator      cpu_reference(cpu_data);
        const Step104PortableOperator portable_reference(portable_data);
        HostVector                    source, result, reference, host_result;
        cpu.initialize_dof_vector(source);
        cpu.initialize_dof_vector(result);
        cpu.initialize_dof_vector(reference);
        cpu.initialize_dof_vector(host_result);
        for (unsigned int block = 0; block < 2; ++block)
          for (const auto index : source.block(block).locally_owned_elements())
            source.block(block)[index] =
              std::sin(0.013 * (index + 1)) + 0.2 * block;
        velocity_constraints.set_zero(source.block(0));
        DeviceVector device_source, device_result, device_reference;
        portable.initialize_dof_vector(device_source);
        portable.initialize_dof_vector(device_result);
        portable.initialize_dof_vector(device_reference);
        copy(device_source, source);
        cpu_reference.vmult(reference, source);
        cpu.vmult(result, source);
        result -= reference;
        const double reference_norm = reference.l2_norm();
        AssertThrow(reference_norm > 0.0,
                    dealii::ExcMessage("Zero reference action."));
        const double cpu_error = result.l2_norm() / reference_norm;
        portable_reference.vmult(device_reference, device_source);
        copy(host_result, device_reference);
        host_result -= reference;
        const double backend_error = host_result.l2_norm() / reference_norm;
        portable.vmult(device_result, device_source);
        device_result -= device_reference;
        const double portable_error = device_result.l2_norm() / reference_norm;
        AssertThrow(
          cpu_error < 1e-11 && portable_error < 1e-11 && backend_error < 1e-11,
          dealii::ExcMessage("Generic/reference Stokes operator mismatch."));
        const auto dofs = velocity_dofs.n_dofs() + pressure_dofs.n_dofs();
        output
          << "\nrefinement: " << refinement - 2
          << ", global refinements: " << refinement
          << ", cells: " << triangulation.n_global_active_cells()
          << ", n_dofs: " << dofs << " = " << velocity_dofs.n_dofs() << " + "
          << pressure_dofs.n_dofs() << '\n'
          << "Relative errors (CPU/reference, Portable/reference, CPU/Portable): "
          << cpu_error << ", " << portable_error << ", " << backend_error
          << '\n';
        for (unsigned int warmup = 0; warmup < 2; ++warmup)
          {
            cpu.vmult(result, source);
            cpu_reference.vmult(result, source);
            portable.vmult(device_result, device_source);
            portable_reference.vmult(device_result, device_source);
          }
        std::vector<double> cpu_times, cpu_reference_times, portable_times,
          portable_reference_times;
        for (unsigned int sample = 0; sample < samples; ++sample)
          {
            const bool reverse = sample % 2 != 0;
            if (reverse)
              measure_pair(portable,
                           portable_reference,
                           device_result,
                           device_source,
                           repetitions,
                           reverse,
                           portable_times,
                           portable_reference_times);
            measure_pair(cpu,
                         cpu_reference,
                         result,
                         source,
                         repetitions,
                         reverse,
                         cpu_times,
                         cpu_reference_times);
            if (!reverse)
              measure_pair(portable,
                           portable_reference,
                           device_result,
                           device_source,
                           repetitions,
                           reverse,
                           portable_times,
                           portable_reference_times);
          }
        const double cpu_time = report(output, "CPU generic", cpu_times, dofs);
        const double cpu_reference_time =
          report(output, "CPU step-104 adaptation", cpu_reference_times, dofs);
        const double portable_time =
          report(output, "Portable generic", portable_times, dofs);
        const double portable_reference_time =
          report(output, "Portable step-104", portable_reference_times, dofs);
        output << "Generic/reference time ratio (CPU, Portable): "
               << cpu_time / cpu_reference_time << ", "
               << portable_time / portable_reference_time << '\n';
      }
  }
} // namespace

int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize initialization(argc, argv, 1);
  try
    {
      if (argc == 2 && std::string(argv[1]) == "--help")
        {
          if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout
              << "Usage: stokes [max_global_refinement=4] [repetitions=100] [samples=5]\n"
              << "Starts at global refinement 2. No solver or multigrid.\n";
          return 0;
        }
      AssertThrow(
        argc <= 4,
        dealii::ExcMessage(
          "Usage: stokes [max_global_refinement] [repetitions] [samples]"));
      const unsigned int max_refinement = argc > 1 ? argument(argv[1]) : 4;
      const unsigned int repetitions    = argc > 2 ? argument(argv[2]) : 100;
      const unsigned int samples        = argc > 3 ? argument(argv[3]) : 5;
      AssertThrow(max_refinement >= 2 && max_refinement <= 8,
                  dealii::ExcMessage("Refinement range is 2 through 8."));
      run(max_refinement, repetitions, samples);
    }
  catch (const std::exception &exception)
    {
      std::cerr << exception.what() << '\n';
      return 1;
    }
}
