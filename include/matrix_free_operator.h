#ifndef PMF_FORM_MATRIX_FREE_OPERATOR_H
#define PMF_FORM_MATRIX_FREE_OPERATOR_H

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <cell_form_lowering.h>

#include <functional>
#include <memory>

namespace pmf
{
  namespace forms
  {
    namespace expression_templates
    {
      /**
       * @brief Polynomial degrees in field-index order, or one degree for all fields.
       * @tparam Degrees Positive field degrees; quadrature uses the largest plus one.
       */
      template <int... Degrees>
      struct FieldDegrees
      {
        static_assert(sizeof...(Degrees) > 0 && ((Degrees > 0) && ...),
                      "provide positive field degrees");
        /** @brief Number of supplied field degrees. */
        static constexpr unsigned int size = sizeof...(Degrees);
        /** @brief Return the degree of a field. @tparam index Field index. */
        template <unsigned int index>
        static constexpr int
        degree()
        {
          static_assert(size == 1 || index < size, "missing field degree");
          constexpr int degrees[] = {Degrees...};
          return degrees[size == 1 ? 0 : index];
        }
        /** @brief Number of quadrature points in each coordinate direction. */
        static constexpr int n_q_points_1d = []() {
          int maximum = 0;
          for (const int degree : {Degrees...})
            if (degree > maximum)
              maximum = degree;
          return maximum + 1;
        }();
      };

      namespace internal
      {
        template <typename Form, typename Field>
        constexpr dealii::EvaluationFlags::EvaluationFlags field_flags =
          (FieldRequirements<Form, Field>::value ?
             dealii::EvaluationFlags::values :
             dealii::EvaluationFlags::nothing) |
          (FieldRequirements<Form, Field>::gradient ?
             dealii::EvaluationFlags::gradients :
             dealii::EvaluationFlags::nothing);

        template <typename Field, int dim>
        constexpr unsigned int field_components =
          Field::shape == ValueShape::scalar ? 1 : dim;

        template <bool mixed, typename Vector>
        DEAL_II_HOST_DEVICE decltype(auto)
        field_vector(Vector &vector, const unsigned int index)
        {
          if constexpr (mixed)
            return vector.block(index);
          else
            return (vector);
        }

        template <typename Form, typename Field, typename Evaluation>
        struct EvaluationSlot
        {
          using TestField = Test<Field::index, Field::shape>;
          Evaluation                         evaluation;
          typename Evaluation::value_type    trial_value;
          typename Evaluation::gradient_type trial_gradient;
          typename Evaluation::value_type    submitted_value;
          typename Evaluation::gradient_type submitted_gradient;

          template <typename Data>
          DEAL_II_HOST_DEVICE explicit EvaluationSlot(const Data &data)
            : evaluation(data, Field::index)
          {}

          DEAL_II_HOST_DEVICE void
          prepare(const unsigned int point)
          {
            if constexpr (FieldRequirements<Form, Field>::value)
              trial_value = evaluation.get_value(point);
            if constexpr (FieldRequirements<Form, Field>::gradient)
              trial_gradient = evaluation.get_gradient(point);
            submitted_value    = typename Evaluation::value_type();
            submitted_gradient = typename Evaluation::gradient_type();
          }

          template <bool mixed, typename Vector>
          DEAL_II_HOST_DEVICE void
          read(const Vector &source)
          {
            evaluation.read_dof_values(
              field_vector<mixed>(source, Field::index));
            evaluation.evaluate(field_flags<Form, Field>);
          }

          DEAL_II_HOST_DEVICE void
          submit(const unsigned int point)
          {
            if constexpr (FieldRequirements<Form, TestField>::value)
              evaluation.submit_value(submitted_value, point);
            if constexpr (FieldRequirements<Form, TestField>::gradient)
              evaluation.submit_gradient(submitted_gradient, point);
          }

          template <bool mixed, typename Vector>
          DEAL_II_HOST_DEVICE void
          scatter(Vector &destination)
          {
            evaluation.integrate(field_flags<Form, TestField>);
            evaluation.distribute_local_to_global(
              field_vector<mixed>(destination, Field::index));
          }
        };

        template <int dim,
                  typename Form,
                  template <typename>
                  class Evaluation,
                  typename Fields>
        struct EvaluationPack;

        template <int dim,
                  typename Form,
                  template <typename>
                  class Evaluation,
                  typename... Fields>
        struct EvaluationPack<dim, Form, Evaluation, TypeList<Fields...>>
          : EvaluationSlot<Form, Fields, Evaluation<Fields>>...
        {
          using Number =
            typename Evaluation<Trial<0, ValueShape::scalar>>::value_type;
          static constexpr unsigned int dimension = dim;
          static constexpr bool         mixed     = sizeof...(Fields) > 1;
          unsigned int                  point     = 0;

          template <typename Field>
          using Slot =
            EvaluationSlot<Form,
                           Trial<Field::index, Field::shape>,
                           Evaluation<Trial<Field::index, Field::shape>>>;

          template <typename Data>
          DEAL_II_HOST_DEVICE explicit EvaluationPack(const Data &data)
            : EvaluationSlot<Form, Fields, Evaluation<Fields>>(data)...
          {}

          void
          reinit(const unsigned int cell)
          {
            (static_cast<Slot<Fields> &>(*this).evaluation.reinit(cell), ...);
          }

          template <typename Vector>
          DEAL_II_HOST_DEVICE void
          read(const Vector &source)
          {
            (static_cast<Slot<Fields> &>(*this).template read<mixed>(source),
             ...);
          }

          DEAL_II_HOST_DEVICE void
          clear()
          {
            (static_cast<Slot<Fields> &>(*this).prepare(point), ...);
          }

          DEAL_II_HOST_DEVICE void
          submit()
          {
            (static_cast<Slot<Fields> &>(*this).submit(point), ...);
          }

          template <typename Vector>
          DEAL_II_HOST_DEVICE void
          scatter(Vector &destination)
          {
            (static_cast<Slot<Fields> &>(*this).template scatter<mixed>(
               destination),
             ...);
          }

          template <typename Field>
          DEAL_II_HOST_DEVICE auto
          value() const
          {
            return static_cast<const Slot<Field> &>(*this).trial_value;
          }

          template <typename Field>
          DEAL_II_HOST_DEVICE auto
          gradient() const
          {
            return static_cast<const Slot<Field> &>(*this).trial_gradient;
          }

          template <typename Field, typename Value>
          DEAL_II_HOST_DEVICE void
          submit_value(const Value &value)
          {
            static_cast<Slot<Field> &>(*this).submitted_value += value;
          }

          template <typename Field, typename GradientType>
          DEAL_II_HOST_DEVICE void
          submit_gradient(const GradientType &gradient)
          {
            static_cast<Slot<Field> &>(*this).submitted_gradient += gradient;
          }
        };

        template <int dim, typename Degrees, typename Number>
        struct CpuEvaluation
        {
          template <typename Field>
          using type =
            dealii::FEEvaluation<dim,
                                 Degrees::template degree<Field::index>(),
                                 Degrees::n_q_points_1d,
                                 field_components<Field, dim>,
                                 Number>;
        };

        template <int dim, typename Degrees, typename Number>
        struct PortableEvaluation
        {
          template <typename Field>
          using type = dealii::Portable::FEEvaluation<
            dim,
            Degrees::template degree<Field::index>(),
            Degrees::n_q_points_1d,
            field_components<Field, dim>,
            Number>;
        };

        template <typename TensorType>
        struct TensorNumber;
        template <int rank, int dim, typename Number>
        struct TensorNumber<dealii::Tensor<rank, dim, Number>>
        {
          using type = Number;
        };

        template <int dim, typename SelectedField, typename Evaluation>
        struct DiagonalContext
        {
          using Number =
            typename TensorNumber<typename Evaluation::gradient_type>::type;
          static constexpr unsigned int      dimension = dim;
          Evaluation                        &evaluation;
          unsigned int                       point;
          typename Evaluation::value_type    submitted_value{};
          typename Evaluation::gradient_type submitted_gradient{};

          template <typename Field>
          DEAL_II_HOST_DEVICE auto
          value() const
          {
            if constexpr (Field::index == SelectedField::index)
              return evaluation.get_value(point);
            else if constexpr (Field::shape == ValueShape::scalar)
              return Number();
            else
              return dealii::Tensor<1, dim, Number>();
          }

          template <typename Field>
          DEAL_II_HOST_DEVICE auto
          gradient() const
          {
            if constexpr (Field::index == SelectedField::index)
              return evaluation.get_gradient(point);
            else
              return dealii::Tensor < Field::shape == ValueShape::scalar ? 1 :
                                                                           2,
                     dim, Number > ();
          }

          template <typename Field, typename Value>
          DEAL_II_HOST_DEVICE void
          submit_value(const Value &value)
          {
            if constexpr (Field::index == SelectedField::index)
              submitted_value += value;
          }

          template <typename Field, typename GradientType>
          DEAL_II_HOST_DEVICE void
          submit_gradient(const GradientType &gradient)
          {
            if constexpr (Field::index == SelectedField::index)
              submitted_gradient += gradient;
          }
        };

        template <int dim, typename Form, typename Field>
        struct DiagonalKernel
        {
          BilinearCellKernel<Form> kernel;
          static constexpr auto    flags =
            field_flags<Form, Field> |
            field_flags<Form, Test<Field::index, Field::shape>>;

          template <typename Evaluation>
          DEAL_II_HOST_DEVICE void
          operator()(Evaluation &evaluation, const unsigned int point) const
          {
            DiagonalContext<dim, Field, Evaluation> context{evaluation, point};
            kernel(context);
            if constexpr ((flags & dealii::EvaluationFlags::values) != 0)
              evaluation.submit_value(context.submitted_value, point);
            if constexpr ((flags & dealii::EvaluationFlags::gradients) != 0)
              evaluation.submit_gradient(context.submitted_gradient, point);
          }

          template <typename Evaluation>
          DEAL_II_HOST_DEVICE void
          operator()(Evaluation *evaluation, const unsigned int point) const
          {
            (*this)(*evaluation, point);
          }
        };

        template <int dim,
                  typename Degrees,
                  typename Form,
                  typename Field,
                  typename Number,
                  typename Vector>
        void
        compute_field_diagonal(const dealii::MatrixFree<dim, Number> &data,
                               Vector                                &diagonal,
                               const BilinearCellKernel<Form>        &kernel)
        {
          using Evaluation =
            typename CpuEvaluation<dim, Degrees, Number>::template type<Field>;
          using Kernel = DiagonalKernel<dim, Form, Field>;
          const std::function<void(Evaluation &)> operation =
            [&kernel](Evaluation &evaluation) {
              evaluation.evaluate(Kernel::flags);
              const Kernel diagonal_kernel{kernel};
              for (unsigned int point = 0; point < evaluation.n_q_points;
                   ++point)
                diagonal_kernel(evaluation, point);
              evaluation.integrate(Kernel::flags);
            };
          dealii::MatrixFreeTools::compute_diagonal(data,
                                                    diagonal,
                                                    operation,
                                                    Field::index);
          for (const auto index : data.get_constrained_dofs(Field::index))
            diagonal.local_element(index) = Number(1);
        }

        template <int dim,
                  typename Degrees,
                  typename Form,
                  typename Field,
                  typename Number,
                  typename Vector>
        void
        compute_field_diagonal(
          const dealii::Portable::MatrixFree<dim, Number> &data,
          Vector                                          &diagonal,
          const BilinearCellKernel<Form>                  &kernel)
        {
          using Kernel = DiagonalKernel<dim, Form, Field>;
          dealii::MatrixFreeTools::compute_diagonal<
            dim,
            Degrees::template degree<Field::index>(),
            Degrees::n_q_points_1d,
            field_components<Field, dim>,
            Number>(data,
                    diagonal,
                    Kernel{kernel},
                    Kernel::flags,
                    Kernel::flags,
                    Field::index);
        }

        template <int dim,
                  typename Degrees,
                  typename Form,
                  typename Data,
                  typename Vector,
                  typename... Fields>
        void
        compute_diagonal(const Data                     &data,
                         Vector                         &diagonal,
                         const BilinearCellKernel<Form> &kernel,
                         TypeList<Fields...>)
        {
          data.initialize_dof_vector(diagonal);
          (compute_field_diagonal<dim, Degrees, Form, Fields>(
             data,
             field_vector<(sizeof...(Fields) > 1)>(diagonal, Fields::index),
             kernel),
           ...);
        }

        template <typename Number, typename Space>
        void
        invert_diagonal(
          dealii::LinearAlgebra::distributed::Vector<Number, Space> &inverse,
          const dealii::LinearAlgebra::distributed::Vector<Number, Space>
            &diagonal)
        {
          using ExecutionSpace = typename Space::kokkos_space::execution_space;
          auto       *entries  = inverse.get_values();
          const auto *diagonal_entries = diagonal.get_values();
          Kokkos::parallel_for(
            "pmf inverse diagonal",
            Kokkos::RangePolicy<ExecutionSpace>(0,
                                                inverse.locally_owned_size()),
            KOKKOS_LAMBDA(const unsigned int index) {
              entries[index] = Number(1) / diagonal_entries[index];
            });
          ExecutionSpace().fence();
        }

        template <typename Number, typename Space>
        void
        invert_diagonal(
          dealii::LinearAlgebra::distributed::BlockVector<Number, Space>
            &inverse,
          const dealii::LinearAlgebra::distributed::BlockVector<Number, Space>
            &diagonal)
        {
          for (unsigned int block = 0; block < inverse.n_blocks(); ++block)
            invert_diagonal(inverse.block(block), diagonal.block(block));
        }

        template <typename Form, typename Fields>
        struct ValidateFields;

        template <typename Form, typename... Fields>
        struct ValidateFields<Form, TypeList<Fields...>>
        {
          static_assert(sizeof...(Fields) > 0,
                        "a bilinear form must contain fields");
          static_assert(((Fields::shape == ValueShape::scalar ||
                          Fields::shape == ValueShape::vector) &&
                         ...),
                        "only scalar and vector fields are supported");
          static_assert(
            std::is_same<
              typename FormFields<Form>::test_fields,
              TypeList<Test<Fields::index, Fields::shape>...>>::value,
            "trial and test fields must have matching indices and shapes");
          static_assert(((Fields::index < sizeof...(Fields)) && ...),
                        "field indices must be contiguous starting at zero");
        };
      } // namespace internal

      /**
       * @brief CPU MatrixFree application of a statically lowered bilinear cell form.
       * @tparam dim Spatial dimension (two or three).
       * @tparam fe_degree Default polynomial degree.
       * @tparam Degrees Field-degree policy, defaulting to a uniform degree.
       * @tparam Form The owned cell-form expression type.
       * @tparam Number Scalar number type.
       * @tparam VectorType Distributed vector, or block vector for multiple fields.
       * Fields use matching trial/test indices and one DoFHandler per field.
       */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number,
                typename VectorType,
                typename Degrees = FieldDegrees<fe_degree>>
      class MatrixFreeCellOperator
        : public dealii::MatrixFreeOperators::Base<dim, VectorType>,
          private internal::
            ValidateFields<Form, typename FormFields<Form>::trial_fields>
      {
        static_assert(dim == 2 || dim == 3,
                      "cell lowering supports dimensions two and three");
        using Fields = typename FormFields<Form>::trial_fields;
        static_assert(Degrees::size == 1 ||
                        Degrees::size == FormFields<Form>::n_trial_fields,
                      "provide one degree or one degree per field");

      public:
        using Data   = dealii::MatrixFree<dim, Number>;
        using Vector = VectorType;

        /** @brief Initialize an operator. @param data Initialized field data. @param form Cell form to store. */
        MatrixFreeCellOperator(std::shared_ptr<const Data> data, Form form)
          : kernel(std::move(form))
        {
          this->initialize(std::move(data));
        }

        /** @brief Initialize a distributed vector with the field partitioning. @param vector Vector to initialize. */
        void
        initialize_dof_vector(VectorType &vector) const
        {
          this->data->initialize_dof_vector(vector);
        }

        /**
         * @brief Return the lazily cached diagonal, with constrained entries set to one.
         * @return The owned diagonal vector; ghost values are not updated.
         * The first call is collective. References expire at recomputation or
         * destruction.
         */
        const VectorType &
        get_diagonal() const
        {
          if (!cached_diagonal)
            {
              auto result =
                std::make_shared<dealii::DiagonalMatrix<VectorType>>();
              internal::compute_diagonal<dim, Degrees>(*this->data,
                                                       result->get_vector(),
                                                       kernel,
                                                       Fields{});
              cached_diagonal = std::move(result);
            }
          return cached_diagonal->get_vector();
        }

        /** @brief Collectively recompute the diagonal and invalidate both caches' references. */
        void
        compute_diagonal() override
        {
          cached_inverse_diagonal.reset();
          cached_diagonal.reset();
          get_diagonal();
          this->diagonal_entries = cached_diagonal;
        }

        /**
         * @brief Return the lazily cached reciprocal diagonal.
         * @return Owned reciprocal entries; ghost values are not updated.
         * @pre Every locally owned diagonal entry must be nonzero.
         * The first call is collective. References expire at recomputation or
         * destruction.
         */
        const VectorType &
        get_inverse_diagonal() const
        {
          if (!cached_inverse_diagonal)
            {
              const auto &diagonal = get_diagonal();
              auto        result =
                std::make_shared<dealii::DiagonalMatrix<VectorType>>();
              auto &inverse = result->get_vector();
              inverse.reinit(diagonal);
              internal::invert_diagonal(inverse, diagonal);
              cached_inverse_diagonal = std::move(result);
            }
          return cached_inverse_diagonal->get_vector();
        }

        /**
         * @brief Apply weighted Jacobi using the cached inverse diagonal.
         * @param dst Initialized destination. @param src Source with matching partitioning.
         * @param omega Relaxation factor. @pre Diagonal entries must be nonzero.
         */
        void
        precondition_Jacobi(VectorType       &dst,
                            const VectorType &src,
                            const Number      omega = Number(1)) const
        {
          const auto &inverse = get_inverse_diagonal();
          dst                 = src;
          dst.scale(inverse);
          dst *= omega;
        }

      private:
        void
        apply_add(VectorType       &destination,
                  const VectorType &source) const override
        {
          this->data->cell_loop(&MatrixFreeCellOperator::local_apply,
                                this,
                                destination,
                                source);
        }

        void
        local_apply(const Data                                  &data,
                    VectorType                                  &destination,
                    const VectorType                            &source,
                    const std::pair<unsigned int, unsigned int> &range) const
        {
          internal::EvaluationPack<
            dim,
            Form,
            internal::CpuEvaluation<dim, Degrees, Number>::template type,
            Fields>
            evaluations(data);
          for (unsigned int cell = range.first; cell < range.second; ++cell)
            {
              evaluations.reinit(cell);
              evaluations.read(source);
              for (unsigned int point = 0;
                   point < dealii::Utilities::pow(Degrees::n_q_points_1d, dim);
                   ++point)
                {
                  evaluations.point = point;
                  evaluations.clear();
                  kernel(evaluations);
                  evaluations.submit();
                }
              evaluations.scatter(destination);
            }
        }

        BilinearCellKernel<Form> kernel;
        mutable std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
          cached_diagonal, cached_inverse_diagonal;
      };

      /**
       * @brief Portable MatrixFree application using the same bilinear cell lowering.
       * @tparam dim Spatial dimension (two or three).
       * @tparam fe_degree Default polynomial degree.
       * @tparam Degrees Field-degree policy, defaulting to a uniform degree.
       * @tparam Form The owned cell-form expression type.
       * @tparam Number Scalar number type.
       * @tparam VectorType Distributed vector, or block vector for multiple fields.
       * Fields use matching trial/test indices and one DoFHandler per field.
       */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number,
                typename VectorType,
                typename Degrees = FieldDegrees<fe_degree>>
      class PortableMatrixFreeCellOperator
        : public dealii::EnableObserverPointer,
          private internal::
            ValidateFields<Form, typename FormFields<Form>::trial_fields>
      {
        static_assert(dim == 2 || dim == 3,
                      "cell lowering supports dimensions two and three");
        using Fields = typename FormFields<Form>::trial_fields;
        static_assert(Degrees::size == 1 ||
                        Degrees::size == FormFields<Form>::n_trial_fields,
                      "provide one degree or one degree per field");

      public:
        using Data   = dealii::Portable::MatrixFree<dim, Number>;
        using Vector = VectorType;

        /** @brief Initialize an operator. @param data Initialized field data. @param form Cell form to store. */
        PortableMatrixFreeCellOperator(std::shared_ptr<Data> data, Form form)
          : data(std::move(data))
          , cell_operation{BilinearCellKernel<Form>(std::move(form))}
        {}

        /** @brief Initialize a distributed vector with the field partitioning. @param vector Vector to initialize. */
        void
        initialize_dof_vector(VectorType &vector) const
        {
          data->initialize_dof_vector(vector);
        }

        /** @brief Apply the form. @param dst Initialized destination. @param src Source with matching partitioning. */
        void
        vmult(VectorType &dst, const VectorType &src) const
        {
          dst = Number();
          data->cell_loop(cell_operation, src, dst);
          data->copy_constrained_values(src, dst);
        }

        /**
         * @brief Return the lazily cached diagonal, with constrained entries set to one.
         * @return The owned diagonal vector; ghost values are not updated.
         * The first call is collective. References expire at recomputation or
         * destruction.
         */
        const VectorType &
        get_diagonal() const
        {
          if (!cached_diagonal)
            {
              auto result =
                std::make_shared<dealii::DiagonalMatrix<VectorType>>();
              internal::compute_diagonal<dim, Degrees>(*data,
                                                       result->get_vector(),
                                                       cell_operation.kernel,
                                                       Fields{});
              cached_diagonal = std::move(result);
            }
          return cached_diagonal->get_vector();
        }

        /** @brief Collectively recompute the diagonal and invalidate both caches' references. */
        void
        compute_diagonal()
        {
          cached_inverse_diagonal.reset();
          cached_diagonal.reset();
          get_diagonal();
        }

        /**
         * @brief Return the lazily cached reciprocal diagonal.
         * @return Owned reciprocal entries; ghost values are not updated.
         * @pre Every locally owned diagonal entry must be nonzero.
         * The first call is collective. References expire at recomputation or
         * destruction.
         */
        const VectorType &
        get_inverse_diagonal() const
        {
          if (!cached_inverse_diagonal)
            {
              const auto &diagonal = get_diagonal();
              auto        result =
                std::make_shared<dealii::DiagonalMatrix<VectorType>>();
              auto &inverse = result->get_vector();
              inverse.reinit(diagonal);
              internal::invert_diagonal(inverse, diagonal);
              cached_inverse_diagonal = std::move(result);
            }
          return cached_inverse_diagonal->get_vector();
        }

        /**
         * @brief Apply weighted Jacobi using the cached inverse diagonal.
         * @param dst Initialized destination. @param src Source with matching partitioning.
         * @param omega Relaxation factor. @pre Diagonal entries must be nonzero.
         */
        void
        precondition_Jacobi(VectorType       &dst,
                            const VectorType &src,
                            const Number      omega = Number(1)) const
        {
          const auto &inverse = get_inverse_diagonal();
          dst                 = src;
          dst.scale(inverse);
          dst *= omega;
        }

      private:
        struct CellOperation
        {
          static constexpr unsigned int n_q_points =
            dealii::Utilities::pow(Degrees::n_q_points_1d, dim);
          BilinearCellKernel<Form> kernel;

          template <typename DeviceVector>
          DEAL_II_HOST_DEVICE void
          operator()(const typename Data::Data *cell_data,
                     const DeviceVector        &source,
                     DeviceVector              &destination) const
          {
            internal::EvaluationPack<
              dim,
              Form,
              internal::PortableEvaluation<dim, Degrees, Number>::template type,
              Fields>
              evaluations(cell_data);
            evaluations.read(source);
            cell_data->for_each_quad_point([&](const int point) {
              evaluations.point = point;
              evaluations.clear();
              kernel(evaluations);
              evaluations.submit();
            });
            evaluations.scatter(destination);
          }
        };

        std::shared_ptr<Data> data;
        CellOperation         cell_operation;
        mutable std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
          cached_diagonal, cached_inverse_diagonal;
      };

      /** @brief CPU bilinear cell operator; form cv/ref qualifiers are ignored. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number     = double,
                typename VectorType = std::conditional_t<
                  (FormFields<Form>::n_trial_fields > 1),
                  dealii::LinearAlgebra::distributed::BlockVector<Number>,
                  dealii::LinearAlgebra::distributed::Vector<Number>>,
                typename Degrees = FieldDegrees<fe_degree>>
      using MatrixFreeOperator = MatrixFreeCellOperator<dim,
                                                        fe_degree,
                                                        std::decay_t<Form>,
                                                        Number,
                                                        VectorType,
                                                        Degrees>;

      /** @brief Portable bilinear cell operator; form cv/ref qualifiers are ignored. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number     = double,
                typename VectorType = std::conditional_t<
                  (FormFields<Form>::n_trial_fields > 1),
                  dealii::LinearAlgebra::distributed::
                    BlockVector<Number, dealii::MemorySpace::Default>,
                  dealii::LinearAlgebra::distributed::
                    Vector<Number, dealii::MemorySpace::Default>>,
                typename Degrees = FieldDegrees<fe_degree>>
      using PortableMatrixFreeOperator =
        PortableMatrixFreeCellOperator<dim,
                                       fe_degree,
                                       std::decay_t<Form>,
                                       Number,
                                       VectorType,
                                       Degrees>;

      /**
       * @brief Create a CPU operator from a bilinear cell form.
       * @tparam dim Spatial dimension. @tparam fe_degree First or uniform field degree.
       * @tparam OtherDegrees Remaining degrees in field order; omit for a uniform degree.
       * @param data Initialized MatrixFree data, which may be const.
       * @param form Expression to store by value; const forms are accepted.
       * @return An operator with statically selected field evaluations.
       */
      template <int dim,
                int fe_degree,
                int... OtherDegrees,
                typename Data,
                typename Form>
      auto
      make_matrix_free_operator(std::shared_ptr<Data> data, Form form)
      {
        using DefaultOperator =
          MatrixFreeOperator<dim, fe_degree, Form, typename Data::value_type>;
        return MatrixFreeOperator<dim,
                                  fe_degree,
                                  Form,
                                  typename Data::value_type,
                                  typename DefaultOperator::Vector,
                                  FieldDegrees<fe_degree, OtherDegrees...>>(
          std::move(data), std::move(form));
      }

      /**
       * @brief Create a Portable operator from a bilinear cell form.
       * @tparam dim Spatial dimension. @tparam fe_degree First or uniform field degree.
       * @tparam OtherDegrees Remaining degrees in field order; omit for a uniform degree.
       * @param data Initialized mutable Portable MatrixFree data.
       * @param form Expression to store by value; const forms are accepted.
       * @return An operator with statically selected field evaluations.
       */
      template <int dim,
                int fe_degree,
                int... OtherDegrees,
                typename Number,
                typename Form>
      auto
      make_portable_matrix_free_operator(
        std::shared_ptr<dealii::Portable::MatrixFree<dim, Number>> data,
        Form                                                       form)
      {
        using DefaultOperator =
          PortableMatrixFreeOperator<dim, fe_degree, Form, Number>;
        return PortableMatrixFreeOperator<
          dim,
          fe_degree,
          Form,
          Number,
          typename DefaultOperator::Vector,
          FieldDegrees<fe_degree, OtherDegrees...>>(std::move(data),
                                                    std::move(form));
      }

      /** @brief Alias for a CPU elasticity operator using generic cell lowering. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number = double,
                typename VectorType =
                  dealii::LinearAlgebra::distributed::Vector<Number>>
      using MatrixFreeElasticityOperator =
        MatrixFreeOperator<dim, fe_degree, Form, Number, VectorType>;

      /** @brief Alias for a Portable elasticity operator using generic cell lowering. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number     = double,
                typename VectorType = dealii::LinearAlgebra::distributed::
                  Vector<Number, dealii::MemorySpace::Default>>
      using PortableMatrixFreeElasticityOperator =
        PortableMatrixFreeOperator<dim, fe_degree, Form, Number, VectorType>;
    } // namespace expression_templates
  } // namespace forms
} // namespace pmf

#endif
