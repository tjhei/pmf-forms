#ifndef PMF_FORM_ELASTICITY_MATRIX_FREE_H
#define PMF_FORM_ELASTICITY_MATRIX_FREE_H

#include <deal.II/base/config.h>

#include <deal.II/base/tensor.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <expression_templates.h>

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace pmf
{
  namespace forms
  {
    namespace expression_templates
    {
      /**
       * @brief Bind constant Lamé parameters for isotropic elasticity.
       * @tparam LambdaTag The tag type used by the form's lambda coefficient.
       * @tparam MuTag The tag type used by the form's mu coefficient.
       */
      /**
       * @brief Create constant coefficient bindings for an elasticity form.
       * @param lambda The first Lamé parameter.
       * @param mu The shear modulus.
       */
      template <typename LambdaTag, typename MuTag, typename Number = double>
      struct ElasticityCoefficients
      {
        using value_type = Number;
        using lambda_tag = LambdaTag;
        using mu_tag     = MuTag;

        Number lambda;
        Number mu;
      };

      /** @brief A value bound to one coefficient symbol in a form. */
      template <typename Tag, typename Number = double>
      struct CoefficientBinding
      {
        using value_type = Number;
        using tag_type   = Tag;
        Number value;
      };

      /** @brief Bind a numeric value to a tagged coefficient symbol. */
      template <typename Tag, typename Number>
      constexpr CoefficientBinding<Tag, Number>
      bind_coefficient(Number value)
      {
        return {value};
      }

      /** @brief Empty coefficient bindings for coefficient-free forms. */
      struct NoCoefficients
      {
        using value_type = double;
      };

      template <typename LambdaTag, typename MuTag, typename Number = double>
      constexpr ElasticityCoefficients<LambdaTag, MuTag, Number>
      elasticity_coefficients(Number lambda, Number mu)
      {
        return {lambda, mu};
      }


      namespace internal
      {
        using dealii::Tensor;
        using dealii::trace;

        template <typename TensorType>
        struct TensorDimension;

        template <int dim, typename Number>
        struct TensorDimension<Tensor<2, dim, Number>>
          : std::integral_constant<int, dim>
        {};

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename TestTag,
                  typename TrialTag>
        DEAL_II_HOST_DEVICE void
        apply_scalar_term(
          const Integral<Inner<Gradient<Test<TestTag, ValueShape::scalar>>,
                               Gradient<Trial<TrialTag, ValueShape::scalar>>>>
            &,
          const Tensor<1, dim, Number> &gradient,
          Tensor<1, dim, Number>       &flux,
          const Coefficients &,
          const Number sign)
        {
          flux += sign * gradient;
        }

        template <int dim,
                  typename Number,
                  typename TestTag,
                  typename TrialTag,
                  typename CoefficientTag,
                  typename BindingNumber>
        DEAL_II_HOST_DEVICE void
        apply_scalar_term(
          const Integral<
            Multiply<Coefficient<CoefficientTag>,
                     Inner<Gradient<Test<TestTag, ValueShape::scalar>>,
                           Gradient<Trial<TrialTag, ValueShape::scalar>>>>> &,
          const Tensor<1, dim, Number>                            &gradient,
          Tensor<1, dim, Number>                                  &flux,
          const CoefficientBinding<CoefficientTag, BindingNumber> &coefficient,
          const Number                                             sign)
        {
          flux += (sign * Number(coefficient.value)) * gradient;
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Left,
                  typename Right>
        DEAL_II_HOST_DEVICE void
        apply_scalar_form(const FormSum<Left, Right>   &form,
                          const Tensor<1, dim, Number> &gradient,
                          Tensor<1, dim, Number>       &flux,
                          const Coefficients           &coefficients,
                          const Number                  sign)
        {
          apply_scalar_form(form.left, gradient, flux, coefficients, sign);
          apply_scalar_form(form.right, gradient, flux, coefficients, sign);
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_scalar_form(const Integral<Expression>   &form,
                          const Tensor<1, dim, Number> &gradient,
                          Tensor<1, dim, Number>       &flux,
                          const Coefficients           &coefficients,
                          const Number                  sign)
        {
          apply_scalar_term(form, gradient, flux, coefficients, sign);
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Left,
                  typename Right>
        DEAL_II_HOST_DEVICE void
        apply_scalar_form(const FormDifference<Left, Right> &form,
                          const Tensor<1, dim, Number>      &gradient,
                          Tensor<1, dim, Number>            &flux,
                          const Coefficients                &coefficients,
                          const Number                       sign)
        {
          apply_scalar_form(form.left, gradient, flux, coefficients, sign);
          apply_scalar_form(form.right, gradient, flux, coefficients, -sign);
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(const Expression &,
                                const Number,
                                const Tensor<1, dim, Number> &,
                                Number &,
                                Tensor<1, dim, Number> &,
                                const Coefficients &,
                                const Number)
        {
          static_assert(sizeof(Expression) == 0,
                        "unsupported scalar bilinear expression");
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename TestTag,
                  typename TrialTag>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Inner<Gradient<Test<TestTag, ValueShape::scalar>>,
                      Gradient<Trial<TrialTag, ValueShape::scalar>>> &,
          const Number,
          const Tensor<1, dim, Number> &gradient,
          Number &,
          Tensor<1, dim, Number> &flux,
          const Coefficients &,
          const Number sign)
        {
          flux += sign * gradient;
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename TestTag,
                  typename TrialTag>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Multiply<Test<TestTag, ValueShape::scalar>,
                         Trial<TrialTag, ValueShape::scalar>> &,
          const Number value,
          const Tensor<1, dim, Number> &,
          Number &submitted_value,
          Tensor<1, dim, Number> &,
          const Coefficients &,
          const Number sign)
        {
          submitted_value += sign * value;
        }

        template <int dim,
                  typename Number,
                  typename TestTag,
                  typename TrialTag,
                  typename CoefficientTag,
                  typename BindingNumber>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Multiply<Coefficient<CoefficientTag>,
                         Multiply<Test<TestTag, ValueShape::scalar>,
                                  Trial<TrialTag, ValueShape::scalar>>> &,
          const Number value,
          const Tensor<1, dim, Number> &,
          Number &submitted_value,
          Tensor<1, dim, Number> &,
          const CoefficientBinding<CoefficientTag, BindingNumber> &binding,
          const Number                                             sign)
        {
          submitted_value += sign * Number(binding.value) * value;
        }

        template <int dim,
                  typename Number,
                  typename TestTag,
                  typename TrialTag,
                  typename CoefficientTag,
                  typename BindingNumber>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Multiply<Multiply<Coefficient<CoefficientTag>,
                                  Test<TestTag, ValueShape::scalar>>,
                         Trial<TrialTag, ValueShape::scalar>> &,
          const Number value,
          const Tensor<1, dim, Number> &,
          Number &submitted_value,
          Tensor<1, dim, Number> &,
          const CoefficientBinding<CoefficientTag, BindingNumber> &binding,
          const Number                                             sign)
        {
          submitted_value += sign * Number(binding.value) * value;
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Left,
                  typename Right>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(const Add<Left, Right>       &expression,
                                const Number                  value,
                                const Tensor<1, dim, Number> &gradient,
                                Number                       &submitted_value,
                                Tensor<1, dim, Number>       &flux,
                                const Coefficients           &coefficients,
                                const Number                  sign)
        {
          apply_scalar_expression(expression.left,
                                  value,
                                  gradient,
                                  submitted_value,
                                  flux,
                                  coefficients,
                                  sign);
          apply_scalar_expression(expression.right,
                                  value,
                                  gradient,
                                  submitted_value,
                                  flux,
                                  coefficients,
                                  sign);
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Left,
                  typename Right>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(const Subtract<Left, Right>  &expression,
                                const Number                  value,
                                const Tensor<1, dim, Number> &gradient,
                                Number                       &submitted_value,
                                Tensor<1, dim, Number>       &flux,
                                const Coefficients           &coefficients,
                                const Number                  sign)
        {
          apply_scalar_expression(expression.left,
                                  value,
                                  gradient,
                                  submitted_value,
                                  flux,
                                  coefficients,
                                  sign);
          apply_scalar_expression(expression.right,
                                  value,
                                  gradient,
                                  submitted_value,
                                  flux,
                                  coefficients,
                                  -sign);
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_scalar_form(const Integral<Expression>   &form,
                          const Number                  value,
                          const Tensor<1, dim, Number> &gradient,
                          Number                       &submitted_value,
                          Tensor<1, dim, Number>       &flux,
                          const Coefficients           &coefficients,
                          const Number                  sign)
        {
          apply_scalar_expression(form.expression,
                                  value,
                                  gradient,
                                  submitted_value,
                                  flux,
                                  coefficients,
                                  sign);
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Left,
                  typename Right>
        DEAL_II_HOST_DEVICE void
        apply_scalar_form(const FormSum<Left, Right>   &form,
                          const Number                  value,
                          const Tensor<1, dim, Number> &gradient,
                          Number                       &submitted_value,
                          Tensor<1, dim, Number>       &flux,
                          const Coefficients           &coefficients,
                          const Number                  sign)
        {
          apply_scalar_form(form.left,
                            value,
                            gradient,
                            submitted_value,
                            flux,
                            coefficients,
                            sign);
          apply_scalar_form(form.right,
                            value,
                            gradient,
                            submitted_value,
                            flux,
                            coefficients,
                            sign);
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Left,
                  typename Right>
        DEAL_II_HOST_DEVICE void
        apply_scalar_form(const FormDifference<Left, Right> &form,
                          const Number                       value,
                          const Tensor<1, dim, Number>      &gradient,
                          Number                            &submitted_value,
                          Tensor<1, dim, Number>            &flux,
                          const Coefficients                &coefficients,
                          const Number                       sign)
        {
          apply_scalar_form(form.left,
                            value,
                            gradient,
                            submitted_value,
                            flux,
                            coefficients,
                            sign);
          apply_scalar_form(form.right,
                            value,
                            gradient,
                            submitted_value,
                            flux,
                            coefficients,
                            -sign);
        }

        template <int dim, typename Form, typename Coefficients>
        class ScalarLaplaceQuadratureKernel
        {
        public:
          ScalarLaplaceQuadratureKernel(Form form, Coefficients coefficients)
            : form(std::move(form))
            , coefficients(std::move(coefficients))
          {}

          template <typename FEEvaluationType>
          DEAL_II_HOST_DEVICE void
          operator()(FEEvaluationType &phi, const unsigned int q) const
          {
            const auto value    = phi.get_value(q);
            const auto gradient = phi.get_gradient(q);
            using Number = typename std::decay<decltype(gradient[0])>::type;
            dealii::Tensor<1, dim, Number> flux;
            flux                   = Number();
            Number submitted_value = Number();
            apply_scalar_form(form,
                              Number(value),
                              gradient,
                              submitted_value,
                              flux,
                              coefficients,
                              Number(1));
            phi.submit_value(submitted_value, q);
            phi.submit_gradient(flux, q);
          }

        private:
          Form         form;
          Coefficients coefficients;
        };

        template <int dim, typename Number>
        DEAL_II_HOST_DEVICE void
        add_symmetric_gradient_term(Tensor<2, dim, Number>       &stress,
                                    const Tensor<2, dim, Number> &gradient,
                                    const Number                  mu,
                                    const Number                  factor)
        {
          stress +=
            (factor * mu * Number(0.5)) * (gradient + transpose(gradient));
        }

        template <int dim, typename Number>
        DEAL_II_HOST_DEVICE void
        add_divergence_term(Tensor<2, dim, Number>       &stress,
                            const Tensor<2, dim, Number> &gradient,
                            const Number                  lambda)
        {
          const Number div_u = trace(gradient);
          for (unsigned int d = 0; d < dim; ++d)
            stress[d][d] += lambda * div_u;
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename TestTag,
                  typename TrialTag>
        DEAL_II_HOST_DEVICE void
        apply_term(
          const Integral<
            Inner<Symmetrize<Gradient<Test<TestTag, ValueShape::vector>>>,
                  Symmetrize<Gradient<Trial<TrialTag, ValueShape::vector>>>>> &,
          const Tensor<2, dim, Number> &gradient,
          Tensor<2, dim, Number>       &stress,
          const Coefficients &,
          const double sign)
        {
          add_symmetric_gradient_term(stress,
                                      gradient,
                                      Number(1),
                                      Number(sign));
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_term(const Integral<Expression> &,
                   const Tensor<2, dim, Number> &,
                   Tensor<2, dim, Number> &,
                   const Coefficients &)
        {
          static_assert(sizeof(Expression) == 0,
                        "This MatrixFree backend currently supports the "
                        "isotropic elasticity expression patterns only");
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_form(const Integral<Expression> &,
                   const Tensor<2, dim, Number> &,
                   Tensor<2, dim, Number> &,
                   const Coefficients &,
                   double);

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_term(const Integral<Expression> &,
                   const Tensor<2, dim, Number> &,
                   Tensor<2, dim, Number> &,
                   const Coefficients &,
                   double);

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename ConstantNumber,
                  typename MuTag,
                  typename TestTag,
                  typename TrialTag>
        DEAL_II_HOST_DEVICE void
        apply_term(
          const Integral<Multiply<
            Multiply<Constant<ConstantNumber>, Coefficient<MuTag>>,
            Inner<Symmetrize<Gradient<Test<TestTag, ValueShape::vector>>>,
                  Symmetrize<Gradient<Trial<TrialTag, ValueShape::vector>>>>>>
            &,
          const Tensor<2, dim, Number> &,
          Tensor<2, dim, Number> &,
          const Coefficients &,
          double);

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename LambdaTag,
                  typename TestTag,
                  typename TrialTag>
        DEAL_II_HOST_DEVICE void
        apply_term(
          const Integral<
            Multiply<Multiply<Coefficient<LambdaTag>,
                              Divergence<Test<TestTag, ValueShape::vector>>>,
                     Divergence<Trial<TrialTag, ValueShape::vector>>>> &,
          const Tensor<2, dim, Number> &,
          Tensor<2, dim, Number> &,
          const Coefficients &,
          double);

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Left,
                  typename Right>
        DEAL_II_HOST_DEVICE void
        apply_form(const FormSum<Left, Right>   &form,
                   const Tensor<2, dim, Number> &gradient,
                   Tensor<2, dim, Number>       &stress,
                   const Coefficients           &coefficients,
                   const double                  sign = 1.0)
        {
          apply_form(form.left, gradient, stress, coefficients, sign);
          apply_form(form.right, gradient, stress, coefficients, sign);
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Left,
                  typename Right>
        DEAL_II_HOST_DEVICE void
        apply_form(const FormDifference<Left, Right> &form,
                   const Tensor<2, dim, Number>      &gradient,
                   Tensor<2, dim, Number>            &stress,
                   const Coefficients                &coefficients,
                   const double                       sign = 1.0)
        {
          apply_form(form.left, gradient, stress, coefficients, sign);
          apply_form(form.right, gradient, stress, coefficients, -sign);
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_form(const Integral<Expression>   &form,
                   const Tensor<2, dim, Number> &gradient,
                   Tensor<2, dim, Number>       &stress,
                   const Coefficients           &coefficients,
                   const double                  sign)
        {
          apply_term(form, gradient, stress, coefficients, sign);
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_term(const Integral<Expression> &,
                   const Tensor<2, dim, Number> &,
                   Tensor<2, dim, Number> &,
                   const Coefficients &,
                   const double)
        {
          static_assert(sizeof(Expression) == 0,
                        "This MatrixFree backend currently supports the "
                        "isotropic elasticity expression patterns only");
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename ConstantNumber,
                  typename MuTag,
                  typename TestTag,
                  typename TrialTag>
        DEAL_II_HOST_DEVICE void
        apply_term(
          const Integral<Multiply<
            Multiply<Constant<ConstantNumber>, Coefficient<MuTag>>,
            Inner<Symmetrize<Gradient<Test<TestTag, ValueShape::vector>>>,
                  Symmetrize<Gradient<Trial<TrialTag, ValueShape::vector>>>>>>
                                       &form,
          const Tensor<2, dim, Number> &gradient,
          Tensor<2, dim, Number>       &stress,
          const Coefficients           &coefficients,
          const double                  sign)
        {
          static_assert(
            std::is_same<MuTag, typename Coefficients::mu_tag>::value,
            "The shear coefficient binding must match the form");
          add_symmetric_gradient_term(
            stress,
            gradient,
            static_cast<Number>(coefficients.mu),
            static_cast<Number>(sign * form.expression.left.left.value));
        }

        template <int dim,
                  typename Number,
                  typename Coefficients,
                  typename LambdaTag,
                  typename TestTag,
                  typename TrialTag>
        DEAL_II_HOST_DEVICE void
        apply_term(
          const Integral<
            Multiply<Multiply<Coefficient<LambdaTag>,
                              Divergence<Test<TestTag, ValueShape::vector>>>,
                     Divergence<Trial<TrialTag, ValueShape::vector>>>> &,
          const Tensor<2, dim, Number> &gradient,
          Tensor<2, dim, Number>       &stress,
          const Coefficients           &coefficients,
          const double                  sign)
        {
          static_assert(
            std::is_same<LambdaTag, typename Coefficients::lambda_tag>::value,
            "The volumetric coefficient binding must match the form");
          add_divergence_term(stress,
                              gradient,
                              static_cast<Number>(sign * coefficients.lambda));
        }
      } // namespace internal

      /** @brief Quadrature kernel for an expression-template form. */
      template <typename Form, typename Coefficients>
      class ElasticityQuadratureKernel
      {
      public:
        ElasticityQuadratureKernel(Form form, Coefficients coefficients)
          : form(std::move(form))
          , coefficients(std::move(coefficients))
        {}

        template <typename FEEvaluationType>
        DEAL_II_HOST_DEVICE void
        operator()(FEEvaluationType &phi, const unsigned int q_point) const
        {
          const auto gradient = phi.get_gradient(q_point);
          using Number = typename std::decay<decltype(gradient[0][0])>::type;
          constexpr int dim = internal::TensorDimension<
            typename std::remove_cv<decltype(gradient)>::type>::value;
          dealii::Tensor<2, dim, Number> stress;
          stress = Number();
          internal::apply_form(form, gradient, stress, coefficients, 1.0);
          phi.submit_gradient(stress, q_point);
        }

      private:
        Form         form;
        Coefficients coefficients;
      };

      /**
       * @brief Apply a static isotropic-elasticity form with CPU MatrixFree.
       *
       * The form expression type selects the quadrature operations at compile
       * time. This first lowering supports the canonical isotropic-elasticity
       * expression with constant Lamé parameters. The quadrature kernel is
       * shared with the Portable::MatrixFree implementation below.
       */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Coefficients = NoCoefficients,
                typename VectorType   = dealii::LinearAlgebra::distributed::
                  Vector<typename Coefficients::value_type>>
      class MatrixFreeFormOperator
        : public dealii::MatrixFreeOperators::Base<dim, VectorType>
      {
      public:
        using Number = typename Coefficients::value_type;
        using Data   = dealii::MatrixFree<dim, Number>;
        using Vector = VectorType;

        MatrixFreeFormOperator(std::shared_ptr<const Data> data,
                               Form                        form,
                               Coefficients                coefficients = {})
          : kernel(std::move(form), std::move(coefficients))
        {
          this->initialize(std::move(data));
        }

        /** @brief Compute the diagonal, which is not implemented yet. */
        void
        compute_diagonal() override
        {
          throw std::logic_error(
            "diagonal computation is not implemented for this operator");
        }

      private:
        void
        apply_add(VectorType       &destination,
                  const VectorType &source) const override
        {
          this->data->cell_loop(&MatrixFreeFormOperator::local_apply,
                                this,
                                destination,
                                source);
        }

        void
        local_apply(
          const Data                                  &matrix_free_data,
          VectorType                                  &destination,
          const VectorType                            &source,
          const std::pair<unsigned int, unsigned int> &cell_range) const
        {
          dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> phi(
            matrix_free_data, 0);
          for (unsigned int cell = cell_range.first; cell < cell_range.second;
               ++cell)
            {
              phi.reinit(cell);
              phi.read_dof_values(source);
              phi.evaluate(dealii::EvaluationFlags::gradients);
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                kernel(phi, q);
              phi.integrate(dealii::EvaluationFlags::gradients);
              phi.distribute_local_to_global(destination);
            }
        }

        ElasticityQuadratureKernel<Form, Coefficients> kernel;
      };

      /**
       * @brief Apply the same static isotropic-elasticity form with
       * Portable::MatrixFree.
       */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Coefficients = NoCoefficients,
                typename VectorType   = dealii::LinearAlgebra::distributed::
                  Vector<typename Coefficients::value_type,
                         dealii::MemorySpace::Default>>
      class PortableMatrixFreeFormOperator
      {
      public:
        using Number = typename Coefficients::value_type;
        using Data   = dealii::Portable::MatrixFree<dim, Number>;
        using Vector = VectorType;

        PortableMatrixFreeFormOperator(std::shared_ptr<Data> data,
                                       Form                  form,
                                       Coefficients          coefficients = {})
          : data(std::move(data))
          , cell_operation{ElasticityQuadratureKernel<Form, Coefficients>(
              std::move(form),
              std::move(coefficients))}
        {}

        void
        initialize_dof_vector(VectorType &vector) const
        {
          data->initialize_dof_vector(vector, 0);
        }

        void
        vmult(VectorType &destination, const VectorType &source) const
        {
          destination = Number();
          data->cell_loop(cell_operation, source, destination);
          data->copy_constrained_values(source, destination, 0);
        }

      private:
        struct CellOperation
        {
          static constexpr unsigned int n_q_points =
            dealii::Utilities::pow(fe_degree + 1, dim);

          ElasticityQuadratureKernel<Form, Coefficients> kernel;

          DEAL_II_HOST_DEVICE void
          operator()(const typename Data::Data                    *cell_data,
                     const dealii::Portable::DeviceVector<Number> &source,
                     dealii::Portable::DeviceVector<Number> &destination) const
          {
            dealii::Portable::
              FEEvaluation<dim, fe_degree, fe_degree + 1, dim, Number>
                phi(cell_data, 0);
            phi.read_dof_values(source);
            phi.evaluate(dealii::EvaluationFlags::gradients);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              kernel(phi, q);
            phi.integrate(dealii::EvaluationFlags::gradients);
            phi.distribute_local_to_global(destination);
          }
        };

        std::shared_ptr<Data> data;
        CellOperation         cell_operation;
      };

      /** @brief MatrixFree operator for scalar gradient inner-product forms. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Coefficients = NoCoefficients,
                typename Number       = typename Coefficients::value_type,
                typename VectorType =
                  dealii::LinearAlgebra::distributed::Vector<Number>>
      class MatrixFreeScalarFormOperator
        : public dealii::MatrixFreeOperators::Base<dim, VectorType>
      {
      public:
        using Data   = dealii::MatrixFree<dim, Number>;
        using Vector = VectorType;

        MatrixFreeScalarFormOperator(std::shared_ptr<const Data> data,
                                     Form                        form,
                                     Coefficients coefficients = {})
          : kernel(std::move(form), std::move(coefficients))
        {
          this->initialize(std::move(data));
        }

        void
        compute_diagonal() override
        {
          throw std::logic_error("diagonal computation is not implemented");
        }

      private:
        void
        apply_add(VectorType &dst, const VectorType &src) const override
        {
          this->data->cell_loop(&MatrixFreeScalarFormOperator::local_apply,
                                this,
                                dst,
                                src);
        }

        void
        local_apply(const Data                                  &mf,
                    VectorType                                  &dst,
                    const VectorType                            &src,
                    const std::pair<unsigned int, unsigned int> &range) const
        {
          dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1> phi(mf, 0);
          for (unsigned int cell = range.first; cell < range.second; ++cell)
            {
              phi.reinit(cell);
              phi.read_dof_values(src);
              phi.evaluate(dealii::EvaluationFlags::values |
                           dealii::EvaluationFlags::gradients);
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                kernel(phi, q);
              phi.integrate(dealii::EvaluationFlags::values |
                            dealii::EvaluationFlags::gradients);
              phi.distribute_local_to_global(dst);
            }
        }

        internal::ScalarLaplaceQuadratureKernel<dim, Form, Coefficients> kernel;
      };

      /** @brief Portable::MatrixFree operator for scalar gradient forms. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Coefficients = NoCoefficients,
                typename Number       = typename Coefficients::value_type,
                typename VectorType   = dealii::LinearAlgebra::distributed::
                  Vector<Number, dealii::MemorySpace::Default>>
      class PortableMatrixFreeScalarFormOperator
      {
      public:
        using Data   = dealii::Portable::MatrixFree<dim, Number>;
        using Vector = VectorType;

        PortableMatrixFreeScalarFormOperator(std::shared_ptr<Data> data,
                                             Form                  form,
                                             Coefficients coefficients = {})
          : data(std::move(data))
          , cell_operation{
              internal::ScalarLaplaceQuadratureKernel<dim, Form, Coefficients>(
                std::move(form),
                std::move(coefficients))}
        {}

        void
        initialize_dof_vector(VectorType &vector) const
        {
          data->initialize_dof_vector(vector, 0);
        }

        void
        vmult(VectorType &dst, const VectorType &src) const
        {
          dst = Number();
          data->cell_loop(cell_operation, src, dst);
          data->copy_constrained_values(src, dst, 0);
        }

      private:
        struct CellOperation
        {
          static constexpr unsigned int n_q_points =
            dealii::Utilities::pow(fe_degree + 1, dim);
          internal::ScalarLaplaceQuadratureKernel<dim, Form, Coefficients>
            kernel;

          DEAL_II_HOST_DEVICE void
          operator()(const typename Data::Data                    *cell_data,
                     const dealii::Portable::DeviceVector<Number> &src,
                     dealii::Portable::DeviceVector<Number>       &dst) const
          {
            dealii::Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1>
              phi(cell_data, 0);
            phi.read_dof_values(src);
            phi.evaluate(dealii::EvaluationFlags::values |
                         dealii::EvaluationFlags::gradients);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              kernel(phi, q);
            phi.integrate(dealii::EvaluationFlags::values |
                          dealii::EvaluationFlags::gradients);
            phi.distribute_local_to_global(dst);
          }
        };

        std::shared_ptr<Data> data;
        CellOperation         cell_operation;
      };

      namespace internal
      {
        template <typename Expression>
        struct FormFieldInfo
        {
          static constexpr bool       found = false;
          static constexpr ValueShape shape = ValueShape::scalar;
        };

        template <typename Tag, ValueShape Shape>
        struct FormFieldInfo<Trial<Tag, Shape>>
        {
          static constexpr bool       found = true;
          static constexpr ValueShape shape = Shape;
        };

        template <typename Tag, ValueShape Shape>
        struct FormFieldInfo<Test<Tag, Shape>>
        {
          static constexpr bool       found = true;
          static constexpr ValueShape shape = Shape;
        };

        template <typename Expression>
        struct FormFieldInfo<Gradient<Expression>> : FormFieldInfo<Expression>
        {};
        template <typename Expression>
        struct FormFieldInfo<Divergence<Expression>> : FormFieldInfo<Expression>
        {};
        template <typename Expression>
        struct FormFieldInfo<Symmetrize<Expression>> : FormFieldInfo<Expression>
        {};

#define PMF_FORM_FIELD_INFO_BINARY(Node)           \
  template <typename Left, typename Right>         \
  struct FormFieldInfo<Node<Left, Right>>          \
    : std::conditional<FormFieldInfo<Left>::found, \
                       FormFieldInfo<Left>,        \
                       FormFieldInfo<Right>>::type \
  {}

        PMF_FORM_FIELD_INFO_BINARY(Add);
        PMF_FORM_FIELD_INFO_BINARY(Subtract);
        PMF_FORM_FIELD_INFO_BINARY(Multiply);
        PMF_FORM_FIELD_INFO_BINARY(Inner);
        PMF_FORM_FIELD_INFO_BINARY(FormSum);
        PMF_FORM_FIELD_INFO_BINARY(FormDifference);
#undef PMF_FORM_FIELD_INFO_BINARY

        template <typename Expression>
        struct FormFieldInfo<Integral<Expression>> : FormFieldInfo<Expression>
        {};
      } // namespace internal

      /**
       * @brief CPU MatrixFree operator selected from the form's field shape.
       *
       * Scalar forms use scalar FEEvaluation. Vector forms use dim components.
       */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Coefficients = NoCoefficients,
                typename VectorType   = dealii::LinearAlgebra::distributed::
                  Vector<typename Coefficients::value_type>>
      using MatrixFreeOperator = typename std::conditional<
        internal::FormFieldInfo<Form>::shape == ValueShape::scalar,
        MatrixFreeScalarFormOperator<dim,
                                     fe_degree,
                                     Form,
                                     Coefficients,
                                     typename Coefficients::value_type,
                                     VectorType>,
        MatrixFreeFormOperator<dim,
                               fe_degree,
                               Form,
                               Coefficients,
                               VectorType>>::type;

      /** @brief Portable::MatrixFree operator selected from the form shape. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Coefficients = NoCoefficients,
                typename VectorType   = dealii::LinearAlgebra::distributed::
                  Vector<typename Coefficients::value_type,
                         dealii::MemorySpace::Default>>
      using PortableMatrixFreeOperator = typename std::conditional<
        internal::FormFieldInfo<Form>::shape == ValueShape::scalar,
        PortableMatrixFreeScalarFormOperator<dim,
                                             fe_degree,
                                             Form,
                                             Coefficients,
                                             typename Coefficients::value_type,
                                             VectorType>,
        PortableMatrixFreeFormOperator<dim,
                                       fe_degree,
                                       Form,
                                       Coefficients,
                                       VectorType>>::type;

      /** @brief Backwards-compatible name for the form-driven CPU operator. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Coefficients,
                typename VectorType = dealii::LinearAlgebra::distributed::
                  Vector<typename Coefficients::value_type>>
      using MatrixFreeElasticityOperator =
        MatrixFreeFormOperator<dim, fe_degree, Form, Coefficients, VectorType>;

      /** @brief Backwards-compatible name for the Portable form operator. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Coefficients,
                typename VectorType = dealii::LinearAlgebra::distributed::
                  Vector<typename Coefficients::value_type,
                         dealii::MemorySpace::Default>>
      using PortableMatrixFreeElasticityOperator =
        PortableMatrixFreeFormOperator<dim,
                                       fe_degree,
                                       Form,
                                       Coefficients,
                                       VectorType>;
    } // namespace expression_templates
  }   // namespace forms
} // namespace pmf

#endif // PMF_FORM_ELASTICITY_MATRIX_FREE_H
