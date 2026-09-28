#ifndef PMF_FORM_MATRIX_FREE_OPERATOR_H
#define PMF_FORM_MATRIX_FREE_OPERATOR_H

#include <deal.II/base/config.h>

#include <deal.II/base/tensor.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <expression_templates.h>

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace pmf
{
  namespace forms
  {
    namespace expression_templates
    {
      namespace internal
      {
        template <typename Kernel>
        struct PortableDiagonalKernel
        {
          Kernel kernel;

          template <typename Evaluation>
          DEAL_II_HOST_DEVICE void
          operator()(Evaluation *evaluation, const unsigned int point) const
          {
            kernel(*evaluation, point);
          }
        };

        template <int dim,
                  int degree,
                  int components,
                  typename Number,
                  typename VectorType,
                  typename Kernel>
        void
        compute_diagonal(const dealii::MatrixFree<dim, Number> &data,
                         VectorType                            &diagonal,
                         const Kernel                          &kernel,
                         const dealii::EvaluationFlags::EvaluationFlags flags,
                         const unsigned int field = 0)
        {
          using Evaluation =
            dealii::FEEvaluation<dim, degree, degree + 1, components, Number>;
          const std::function<void(Evaluation &)> operation =
            [&kernel, flags](Evaluation &evaluation) {
              evaluation.evaluate(flags);
              for (unsigned int point = 0; point < evaluation.n_q_points;
                   ++point)
                kernel(evaluation, point);
              evaluation.integrate(flags);
            };
          dealii::MatrixFreeTools::compute_diagonal(data,
                                                    diagonal,
                                                    operation,
                                                    field);
          for (const auto index : data.get_constrained_dofs(field))
            diagonal.local_element(index) = Number(1);
        }

        template <int dim,
                  int degree,
                  int components,
                  typename Number,
                  typename VectorType,
                  typename Kernel>
        void
        compute_diagonal(const dealii::Portable::MatrixFree<dim, Number> &data,
                         VectorType   &diagonal,
                         const Kernel &kernel,
                         const dealii::EvaluationFlags::EvaluationFlags flags,
                         const unsigned int field = 0)
        {
          dealii::MatrixFreeTools::
            compute_diagonal<dim, degree, degree + 1, components, Number>(
              data,
              diagonal,
              PortableDiagonalKernel<Kernel>{kernel},
              flags,
              flags,
              field);
        }

        using dealii::Tensor;
        using dealii::trace;

        template <typename Expression>
        struct CoefficientValue;

        template <typename Number>
        struct CoefficientValue<Coefficient<Number>>
        {
          DEAL_II_HOST_DEVICE static Number
          get(const Coefficient<Number> &expression)
          {
            return expression.value;
          }
        };

        template <typename Expression>
        struct CoefficientValue<Integral<Expression>>
        {
          DEAL_II_HOST_DEVICE static auto
          get(const Integral<Expression> &form)
          {
            return CoefficientValue<Expression>::get(form.expression);
          }
        };

#define PMF_FORM_COEFFICIENT_VALUE_BINARY(Node)                \
  template <typename Left, typename Right>                     \
  struct CoefficientValue<Node<Left, Right>>                   \
  {                                                            \
    DEAL_II_HOST_DEVICE static auto                            \
    get(const Node<Left, Right> &expression)                   \
    {                                                          \
      if constexpr (FormFields<Left>::n_coefficients > 0)      \
        return CoefficientValue<Left>::get(expression.left);   \
      else                                                     \
        return CoefficientValue<Right>::get(expression.right); \
    }                                                          \
  }

        PMF_FORM_COEFFICIENT_VALUE_BINARY(Add);
        PMF_FORM_COEFFICIENT_VALUE_BINARY(Subtract);
        PMF_FORM_COEFFICIENT_VALUE_BINARY(Multiply);
        PMF_FORM_COEFFICIENT_VALUE_BINARY(FormSum);
        PMF_FORM_COEFFICIENT_VALUE_BINARY(FormDifference);
#undef PMF_FORM_COEFFICIENT_VALUE_BINARY

        template <typename Form>
        DEAL_II_HOST_DEVICE auto
        single_coefficient_value(const Form &form)
        {
          static_assert(FormFields<Form>::n_coefficients == 1,
                        "expected exactly one coefficient occurrence");
          return CoefficientValue<Form>::get(form);
        }

        template <typename TensorType>
        struct TensorDimension;

        template <int dim, typename Number>
        struct TensorDimension<Tensor<2, dim, Number>>
          : std::integral_constant<int, dim>
        {};

        template <int dim,
                  typename Number,
                  unsigned int TestIndex,
                  unsigned int TrialIndex>
        DEAL_II_HOST_DEVICE void
        apply_scalar_term(
          const Integral<Inner<Gradient<Test<TestIndex, ValueShape::scalar>>,
                               Gradient<Trial<TrialIndex, ValueShape::scalar>>>>
            &,
          const Tensor<1, dim, Number> &gradient,
          Tensor<1, dim, Number>       &flux,
          const Number                  sign)
        {
          flux += sign * gradient;
        }

        template <int dim,
                  typename Number,
                  unsigned int TestIndex,
                  unsigned int TrialIndex,
                  typename CoefficientNumber>
        DEAL_II_HOST_DEVICE void
        apply_scalar_term(
          const Integral<
            Multiply<Coefficient<CoefficientNumber>,
                     Inner<Gradient<Test<TestIndex, ValueShape::scalar>>,
                           Gradient<Trial<TrialIndex, ValueShape::scalar>>>>>
                                       &form,
          const Tensor<1, dim, Number> &gradient,
          Tensor<1, dim, Number>       &flux,
          const Number                  sign)
        {
          flux += (sign * Number(form.expression.left.value)) * gradient;
        }

        template <int dim, typename Number, typename Left, typename Right>
        DEAL_II_HOST_DEVICE void
        apply_scalar_form(const FormSum<Left, Right>   &form,
                          const Tensor<1, dim, Number> &gradient,
                          Tensor<1, dim, Number>       &flux,
                          const Number                  sign)
        {
          apply_scalar_form(form.left, gradient, flux, sign);
          apply_scalar_form(form.right, gradient, flux, sign);
        }

        template <int dim, typename Number, typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_scalar_form(const Integral<Expression>   &form,
                          const Tensor<1, dim, Number> &gradient,
                          Tensor<1, dim, Number>       &flux,
                          const Number                  sign)
        {
          apply_scalar_term(form, gradient, flux, sign);
        }

        template <int dim, typename Number, typename Left, typename Right>
        DEAL_II_HOST_DEVICE void
        apply_scalar_form(const FormDifference<Left, Right> &form,
                          const Tensor<1, dim, Number>      &gradient,
                          Tensor<1, dim, Number>            &flux,
                          const Number                       sign)
        {
          apply_scalar_form(form.left, gradient, flux, sign);
          apply_scalar_form(form.right, gradient, flux, -sign);
        }

        template <int dim, typename Number, typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(const Expression &,
                                const Number,
                                const Tensor<1, dim, Number> &,
                                Number &,
                                Tensor<1, dim, Number> &,
                                const Number)
        {
          static_assert(sizeof(Expression) == 0,
                        "unsupported scalar bilinear expression");
        }

        template <int dim,
                  typename Number,
                  unsigned int TestIndex,
                  unsigned int TrialIndex>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Inner<Gradient<Test<TestIndex, ValueShape::scalar>>,
                      Gradient<Trial<TrialIndex, ValueShape::scalar>>> &,
          const Number,
          const Tensor<1, dim, Number> &gradient,
          Number &,
          Tensor<1, dim, Number> &flux,
          const Number            sign)
        {
          flux += sign * gradient;
        }

        template <int dim,
                  typename Number,
                  unsigned int TestIndex,
                  unsigned int TrialIndex,
                  typename ConstantNumber>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Multiply<Constant<ConstantNumber>,
                         Inner<Gradient<Test<TestIndex, ValueShape::scalar>>,
                               Gradient<Trial<TrialIndex, ValueShape::scalar>>>>
            &expression,
          const Number,
          const Tensor<1, dim, Number> &gradient,
          Number &,
          Tensor<1, dim, Number> &flux,
          const Number            sign)
        {
          flux += (sign * Number(expression.left.value)) * gradient;
        }

        template <int dim,
                  typename Number,
                  unsigned int TestIndex,
                  unsigned int TrialIndex,
                  typename CoefficientNumber>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Multiply<Coefficient<CoefficientNumber>,
                         Inner<Gradient<Test<TestIndex, ValueShape::scalar>>,
                               Gradient<Trial<TrialIndex, ValueShape::scalar>>>>
            &expression,
          const Number,
          const Tensor<1, dim, Number> &gradient,
          Number &,
          Tensor<1, dim, Number> &flux,
          const Number            sign)
        {
          flux += (sign * Number(expression.left.value)) * gradient;
        }

        template <int dim,
                  typename Number,
                  unsigned int TestIndex,
                  unsigned int TrialIndex>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Multiply<Test<TestIndex, ValueShape::scalar>,
                         Trial<TrialIndex, ValueShape::scalar>> &,
          const Number value,
          const Tensor<1, dim, Number> &,
          Number &submitted_value,
          Tensor<1, dim, Number> &,
          const Number sign)
        {
          submitted_value += sign * value;
        }

        template <int dim,
                  typename Number,
                  unsigned int TestIndex,
                  unsigned int TrialIndex,
                  typename ConstantNumber>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Multiply<Constant<ConstantNumber>,
                         Multiply<Test<TestIndex, ValueShape::scalar>,
                                  Trial<TrialIndex, ValueShape::scalar>>>
                      &constant,
          const Number value,
          const Tensor<1, dim, Number> &,
          Number &submitted_value,
          Tensor<1, dim, Number> &,
          const Number sign)
        {
          submitted_value += (sign * Number(constant.left.value)) * value;
        }

        template <int dim,
                  typename Number,
                  unsigned int TestIndex,
                  unsigned int TrialIndex,
                  typename ConstantNumber>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Multiply<Test<TestIndex, ValueShape::scalar>,
                         Multiply<Constant<ConstantNumber>,
                                  Trial<TrialIndex, ValueShape::scalar>>>
                      &expression,
          const Number value,
          const Tensor<1, dim, Number> &,
          Number &submitted_value,
          Tensor<1, dim, Number> &,
          const Number sign)
        {
          submitted_value +=
            (sign * Number(expression.right.left.value)) * value;
        }

        template <int dim,
                  typename Number,
                  unsigned int TestIndex,
                  unsigned int TrialIndex,
                  typename ConstantNumber>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Multiply<Multiply<Constant<ConstantNumber>,
                                  Test<TestIndex, ValueShape::scalar>>,
                         Trial<TrialIndex, ValueShape::scalar>> &expression,
          const Number                                           value,
          const Tensor<1, dim, Number> &,
          Number &submitted_value,
          Tensor<1, dim, Number> &,
          const Number sign)
        {
          submitted_value +=
            (sign * Number(expression.left.left.value)) * value;
        }

        template <int dim,
                  typename Number,
                  unsigned int TestIndex,
                  unsigned int TrialIndex,
                  typename CoefficientNumber>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Multiply<Coefficient<CoefficientNumber>,
                         Multiply<Test<TestIndex, ValueShape::scalar>,
                                  Trial<TrialIndex, ValueShape::scalar>>>
                      &expression,
          const Number value,
          const Tensor<1, dim, Number> &,
          Number &submitted_value,
          Tensor<1, dim, Number> &,
          const Number sign)
        {
          submitted_value += sign * Number(expression.left.value) * value;
        }

        template <int dim,
                  typename Number,
                  unsigned int TestIndex,
                  unsigned int TrialIndex,
                  typename CoefficientNumber>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(
          const Multiply<Multiply<Coefficient<CoefficientNumber>,
                                  Test<TestIndex, ValueShape::scalar>>,
                         Trial<TrialIndex, ValueShape::scalar>> &expression,
          const Number                                           value,
          const Tensor<1, dim, Number> &,
          Number &submitted_value,
          Tensor<1, dim, Number> &,
          const Number sign)
        {
          submitted_value += sign * Number(expression.left.left.value) * value;
        }

        template <int dim, typename Number, typename Left, typename Right>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(const Add<Left, Right>       &expression,
                                const Number                  value,
                                const Tensor<1, dim, Number> &gradient,
                                Number                       &submitted_value,
                                Tensor<1, dim, Number>       &flux,
                                const Number                  sign)
        {
          apply_scalar_expression(
            expression.left, value, gradient, submitted_value, flux, sign);
          apply_scalar_expression(
            expression.right, value, gradient, submitted_value, flux, sign);
        }

        template <int dim, typename Number, typename Left, typename Right>
        DEAL_II_HOST_DEVICE void
        apply_scalar_expression(const Subtract<Left, Right>  &expression,
                                const Number                  value,
                                const Tensor<1, dim, Number> &gradient,
                                Number                       &submitted_value,
                                Tensor<1, dim, Number>       &flux,
                                const Number                  sign)
        {
          apply_scalar_expression(
            expression.left, value, gradient, submitted_value, flux, sign);
          apply_scalar_expression(
            expression.right, value, gradient, submitted_value, flux, -sign);
        }

        template <int dim, typename Number, typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_scalar_form(const Integral<Expression>   &form,
                          const Number                  value,
                          const Tensor<1, dim, Number> &gradient,
                          Number                       &submitted_value,
                          Tensor<1, dim, Number>       &flux,
                          const Number                  sign)
        {
          apply_scalar_expression(
            form.expression, value, gradient, submitted_value, flux, sign);
        }

        template <int dim, typename Number, typename Left, typename Right>
        DEAL_II_HOST_DEVICE void
        apply_scalar_form(const FormSum<Left, Right>   &form,
                          const Number                  value,
                          const Tensor<1, dim, Number> &gradient,
                          Number                       &submitted_value,
                          Tensor<1, dim, Number>       &flux,
                          const Number                  sign)
        {
          apply_scalar_form(
            form.left, value, gradient, submitted_value, flux, sign);
          apply_scalar_form(
            form.right, value, gradient, submitted_value, flux, sign);
        }

        template <int dim, typename Number, typename Left, typename Right>
        DEAL_II_HOST_DEVICE void
        apply_scalar_form(const FormDifference<Left, Right> &form,
                          const Number                       value,
                          const Tensor<1, dim, Number>      &gradient,
                          Number                            &submitted_value,
                          Tensor<1, dim, Number>            &flux,
                          const Number                       sign)
        {
          apply_scalar_form(
            form.left, value, gradient, submitted_value, flux, sign);
          apply_scalar_form(
            form.right, value, gradient, submitted_value, flux, -sign);
        }

        template <int dim, typename Form>
        class ScalarLaplaceQuadratureKernel
        {
        public:
          ScalarLaplaceQuadratureKernel(Form form)
            : form(std::move(form))
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
            apply_scalar_form(
              form, Number(value), gradient, submitted_value, flux, Number(1));
            phi.submit_value(submitted_value, q);
            phi.submit_gradient(flux, q);
          }

        private:
          Form form;
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
                  unsigned int TestIndex,
                  unsigned int TrialIndex>
        DEAL_II_HOST_DEVICE void
        apply_term(
          const Integral<
            Inner<Symmetrize<Gradient<Test<TestIndex, ValueShape::vector>>>,
                  Symmetrize<Gradient<Trial<TrialIndex, ValueShape::vector>>>>>
            &,
          const Tensor<2, dim, Number> &gradient,
          Tensor<2, dim, Number>       &stress,
          const double                  sign)
        {
          add_symmetric_gradient_term(stress,
                                      gradient,
                                      Number(1),
                                      Number(sign));
        }

        template <int dim, typename Number, typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_form(const Integral<Expression> &,
                   const Tensor<2, dim, Number> &,
                   Tensor<2, dim, Number> &,
                   double = 1.0);

        template <int dim, typename Number, typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_term(const Integral<Expression> &,
                   const Tensor<2, dim, Number> &,
                   Tensor<2, dim, Number> &,
                   double);

        template <int dim,
                  typename Number,
                  typename ConstantNumber,
                  typename MuNumber,
                  unsigned int TestIndex,
                  unsigned int TrialIndex>
        DEAL_II_HOST_DEVICE void
        apply_term(
          const Integral<Multiply<
            Multiply<Constant<ConstantNumber>, Coefficient<MuNumber>>,
            Inner<Symmetrize<Gradient<Test<TestIndex, ValueShape::vector>>>,
                  Symmetrize<Gradient<Trial<TrialIndex, ValueShape::vector>>>>>>
            &,
          const Tensor<2, dim, Number> &,
          Tensor<2, dim, Number> &,
          double);

        template <int dim,
                  typename Number,
                  typename LambdaNumber,
                  unsigned int TestIndex,
                  unsigned int TrialIndex>
        DEAL_II_HOST_DEVICE void
        apply_term(
          const Integral<
            Multiply<Multiply<Coefficient<LambdaNumber>,
                              Divergence<Test<TestIndex, ValueShape::vector>>>,
                     Divergence<Trial<TrialIndex, ValueShape::vector>>>> &form,
          const Tensor<2, dim, Number> &,
          Tensor<2, dim, Number> &,
          double);

        template <int dim, typename Number, typename Left, typename Right>
        DEAL_II_HOST_DEVICE void
        apply_form(const Integral<Add<Left, Right>> &form,
                   const Tensor<2, dim, Number>     &gradient,
                   Tensor<2, dim, Number>           &stress,
                   double                            sign = 1.0);

        template <int dim, typename Number, typename Left, typename Right>
        DEAL_II_HOST_DEVICE void
        apply_form(const Integral<Subtract<Left, Right>> &form,
                   const Tensor<2, dim, Number>          &gradient,
                   Tensor<2, dim, Number>                &stress,
                   double                                 sign = 1.0);

        template <int dim, typename Number, typename Left, typename Right>
        DEAL_II_HOST_DEVICE void
        apply_form(const FormSum<Left, Right>   &form,
                   const Tensor<2, dim, Number> &gradient,
                   Tensor<2, dim, Number>       &stress,
                   const double                  sign = 1.0)
        {
          apply_form(form.left, gradient, stress, sign);
          apply_form(form.right, gradient, stress, sign);
        }

        template <int dim, typename Number, typename Left, typename Right>
        DEAL_II_HOST_DEVICE void
        apply_form(const FormDifference<Left, Right> &form,
                   const Tensor<2, dim, Number>      &gradient,
                   Tensor<2, dim, Number>            &stress,
                   const double                       sign = 1.0)
        {
          apply_form(form.left, gradient, stress, sign);
          apply_form(form.right, gradient, stress, -sign);
        }

        template <int dim, typename Number, typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_form(const Integral<Expression>   &form,
                   const Tensor<2, dim, Number> &gradient,
                   Tensor<2, dim, Number>       &stress,
                   const double                  sign)
        {
          apply_term(form, gradient, stress, sign);
        }

        template <int dim, typename Number, typename Expression>
        DEAL_II_HOST_DEVICE void
        apply_term(const Integral<Expression> &,
                   const Tensor<2, dim, Number> &,
                   Tensor<2, dim, Number> &,
                   const double)
        {
          static_assert(sizeof(Expression) == 0,
                        "This MatrixFree backend currently supports the "
                        "isotropic elasticity expression patterns only");
        }

        template <int dim,
                  typename Number,
                  typename ConstantNumber,
                  typename MuNumber,
                  unsigned int TestIndex,
                  unsigned int TrialIndex>
        DEAL_II_HOST_DEVICE void
        apply_term(
          const Integral<Multiply<
            Multiply<Constant<ConstantNumber>, Coefficient<MuNumber>>,
            Inner<Symmetrize<Gradient<Test<TestIndex, ValueShape::vector>>>,
                  Symmetrize<Gradient<Trial<TrialIndex, ValueShape::vector>>>>>>
                                       &form,
          const Tensor<2, dim, Number> &gradient,
          Tensor<2, dim, Number>       &stress,
          const double                  sign)
        {
          add_symmetric_gradient_term(
            stress,
            gradient,
            static_cast<Number>(form.expression.left.right.value),
            static_cast<Number>(sign * form.expression.left.left.value));
        }

        template <int dim,
                  typename Number,
                  typename LambdaNumber,
                  unsigned int TestIndex,
                  unsigned int TrialIndex>
        DEAL_II_HOST_DEVICE void
        apply_term(
          const Integral<
            Multiply<Multiply<Coefficient<LambdaNumber>,
                              Divergence<Test<TestIndex, ValueShape::vector>>>,
                     Divergence<Trial<TrialIndex, ValueShape::vector>>>> &form,
          const Tensor<2, dim, Number> &gradient,
          Tensor<2, dim, Number>       &stress,
          const double                  sign)
        {
          add_divergence_term(stress,
                              gradient,
                              static_cast<Number>(
                                sign * form.expression.left.left.value));
        }

        template <int dim, typename Number, typename Left, typename Right>
        DEAL_II_HOST_DEVICE void
        apply_form(const Integral<Add<Left, Right>> &form,
                   const Tensor<2, dim, Number>     &gradient,
                   Tensor<2, dim, Number>           &stress,
                   const double                      sign)
        {
          apply_form(Integral<Left>{form.expression.left},
                     gradient,
                     stress,
                     sign);
          apply_form(Integral<Right>{form.expression.right},
                     gradient,
                     stress,
                     sign);
        }

        template <int dim, typename Number, typename Left, typename Right>
        DEAL_II_HOST_DEVICE void
        apply_form(const Integral<Subtract<Left, Right>> &form,
                   const Tensor<2, dim, Number>          &gradient,
                   Tensor<2, dim, Number>                &stress,
                   const double                           sign)
        {
          apply_form(Integral<Left>{form.expression.left},
                     gradient,
                     stress,
                     sign);
          apply_form(Integral<Right>{form.expression.right},
                     gradient,
                     stress,
                     -sign);
        }

      } // namespace internal

      /** @brief Quadrature kernel for an expression-template form. */
      template <typename Form>
      class ElasticityQuadratureKernel
      {
      public:
        ElasticityQuadratureKernel(Form form)
          : form(std::move(form))
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
          internal::apply_form(form, gradient, stress, 1.0);
          phi.submit_gradient(stress, q_point);
        }

      private:
        Form form;
      };

      namespace internal
      {
        template <typename Field, int dim>
        struct FieldComponentCount
          : std::integral_constant<unsigned int,
                                   Field::shape == ValueShape::scalar ? 1 : dim>
        {};

        template <typename Fields, unsigned int index, int dim>
        struct FieldComponentOffset;

        template <typename First, typename... Rest, int dim>
        struct FieldComponentOffset<TypeList<First, Rest...>, 0, dim>
          : std::integral_constant<unsigned int, 0>
        {};

        template <typename First, typename... Rest, unsigned int index, int dim>
        struct FieldComponentOffset<TypeList<First, Rest...>, index, dim>
          : std::integral_constant<
              unsigned int,
              FieldComponentCount<First, dim>::value +
                FieldComponentOffset<TypeList<Rest...>, index - 1, dim>::value>
        {};

        template <typename Fields, int dim>
        struct TotalFieldComponents;

        template <int dim>
        struct TotalFieldComponents<TypeList<>, dim>
          : std::integral_constant<unsigned int, 0>
        {};

        template <typename First, typename... Rest, int dim>
        struct TotalFieldComponents<TypeList<First, Rest...>, dim>
          : std::integral_constant<
              unsigned int,
              FieldComponentCount<First, dim>::value +
                TotalFieldComponents<TypeList<Rest...>, dim>::value>
        {};

      } // namespace internal

      /** @brief Quadrature kernel for the current two-field Stokes form. */
      template <int dim, typename Form>
      class StokesQuadratureKernel
      {
        using Analysis    = FormFields<Form>;
        using TrialFields = typename Analysis::trial_fields;
        using TestFields  = typename Analysis::test_fields;
        using Velocity    = typename internal::TypeListAt<0, TrialFields>::type;
        using Pressure    = typename internal::TypeListAt<1, TrialFields>::type;
        using TestVelocity = typename internal::TypeListAt<0, TestFields>::type;
        using TestPressure = typename internal::TypeListAt<1, TestFields>::type;

      public:
        StokesQuadratureKernel(Form form)
          : form(std::move(form))
        {
          static_assert(
            Analysis::n_trial_fields == 2 && Analysis::n_test_fields == 2,
            "the Stokes kernel expects two trial and two test fields");
          static_assert(
            Velocity::shape == ValueShape::vector &&
              TestVelocity::shape == ValueShape::vector &&
              Pressure::shape == ValueShape::scalar &&
              TestPressure::shape == ValueShape::scalar,
            "Stokes fields must be vector velocity and scalar pressure");
          static_assert(Velocity::index == 0 && TestVelocity::index == 0 &&
                          Pressure::index == 1 && TestPressure::index == 1,
                        "the Stokes kernel expects velocity at index 0 and "
                        "pressure at index 1");
          static_assert(
            Analysis::n_coefficients == 1,
            "the current Stokes kernel expects one viscosity symbol");
          static_assert(
            internal::FieldRequirements<Form, Velocity>::gradient &&
              internal::FieldRequirements<Form, Pressure>::value &&
              internal::FieldRequirements<Form, TestVelocity>::gradient &&
              internal::FieldRequirements<Form, TestPressure>::value,
            "Stokes evaluations must be derived as u:gradient, p:value, "
            "v:gradient, q:value");
        }

        template <typename VelocityEvaluation, typename PressureEvaluation>
        DEAL_II_HOST_DEVICE void
        operator()(VelocityEvaluation &velocity_phi,
                   PressureEvaluation &pressure_phi,
                   const unsigned int  q) const
        {
          const auto velocity_gradient = velocity_phi.get_gradient(q);
          using Number =
            typename std::decay<decltype(velocity_gradient[0][0])>::type;

          const Number mu =
            static_cast<Number>(internal::single_coefficient_value(form));
          const auto symmetric_gradient =
            Number(0.5) * (velocity_gradient + transpose(velocity_gradient));
          dealii::Tensor<2, dim, Number> stress =
            (Number(2) * mu) * symmetric_gradient;
          const Number pressure   = pressure_phi.get_value(q);
          const Number divergence = trace(velocity_gradient);
          for (unsigned int d = 0; d < dim; ++d)
            stress[d][d] -= pressure;

          velocity_phi.submit_gradient(stress, q);
          pressure_phi.submit_value(-divergence, q);
        }

        /**
         * @brief Apply the velocity diagonal block at one quadrature point.
         * @param velocity_phi The velocity evaluation with evaluated gradients.
         * @param point The quadrature-point index.
         */
        template <typename VelocityEvaluation>
        DEAL_II_HOST_DEVICE void
        operator()(VelocityEvaluation &velocity_phi,
                   const unsigned int  point) const
        {
          const auto gradient = velocity_phi.get_gradient(point);
          using Number        = std::decay_t<decltype(gradient[0][0])>;
          const Number viscosity =
            Number(internal::single_coefficient_value(form));
          velocity_phi.submit_gradient(viscosity *
                                         (gradient + transpose(gradient)),
                                       point);
        }

      private:
        Form form;
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
                typename Number = double,
                typename VectorType =
                  dealii::LinearAlgebra::distributed::Vector<Number>>
      class MatrixFreeFormOperator
        : public dealii::MatrixFreeOperators::Base<dim, VectorType>
      {
      public:
        using Data   = dealii::MatrixFree<dim, Number>;
        using Vector = VectorType;

        MatrixFreeFormOperator(std::shared_ptr<const Data> data, Form form)
          : kernel(std::move(form))
        {
          this->initialize(std::move(data));
        }

        /**
         * @brief Return the diagonal, computing and caching it on first use.
         * @return The owned diagonal vector (not its inverse).
         * Constrained entries are one. The first call is collective over the
         * operator's MPI communicator; all ranks must call it consistently.
         * The reference remains valid until recomputation or destruction.
         */
        const VectorType &
        get_diagonal() const
        {
          if (!cached_diagonal)
            cached_diagonal = build_diagonal();
          return cached_diagonal->get_vector();
        }

        /**
         * @brief Recompute the diagonal, replacing any cached values.
         * This collective operation invalidates earlier diagonal references.
         */
        void
        compute_diagonal() override
        {
          cached_diagonal        = build_diagonal();
          this->diagonal_entries = cached_diagonal;
        }

      private:
        std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
        build_diagonal() const
        {
          auto  result = std::make_shared<dealii::DiagonalMatrix<VectorType>>();
          auto &diagonal = result->get_vector();
          internal::compute_diagonal<dim, fe_degree, dim>(
            *this->data, diagonal, kernel, dealii::EvaluationFlags::gradients);
          return result;
        }

        mutable std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
          cached_diagonal;

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

        ElasticityQuadratureKernel<Form> kernel;
      };

      /**
       * @brief Apply the same static isotropic-elasticity form with
       * Portable::MatrixFree.
       */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number     = double,
                typename VectorType = dealii::LinearAlgebra::distributed::
                  Vector<Number, dealii::MemorySpace::Default>>
      class PortableMatrixFreeFormOperator
      {
      public:
        using Data   = dealii::Portable::MatrixFree<dim, Number>;
        using Vector = VectorType;

        PortableMatrixFreeFormOperator(std::shared_ptr<Data> data, Form form)
          : data(std::move(data))
          , cell_operation{ElasticityQuadratureKernel<Form>(std::move(form))}
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

        /**
         * @brief Return the diagonal, computing and caching it on first use.
         * @return The owned diagonal vector (not its inverse).
         * Constrained entries are one. The first call is collective over the
         * operator's MPI communicator; all ranks must call it consistently.
         * The reference remains valid until recomputation or destruction.
         */
        const VectorType &
        get_diagonal() const
        {
          if (!cached_diagonal)
            cached_diagonal = build_diagonal();
          return cached_diagonal->get_vector();
        }

        /**
         * @brief Recompute the diagonal, replacing any cached values.
         * This collective operation invalidates earlier diagonal references.
         */
        void
        compute_diagonal()
        {
          cached_diagonal = build_diagonal();
        }

      private:
        std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
        build_diagonal() const
        {
          auto  result = std::make_shared<dealii::DiagonalMatrix<VectorType>>();
          auto &diagonal = result->get_vector();
          internal::compute_diagonal<dim, fe_degree, dim>(
            *data,
            diagonal,
            cell_operation.kernel,
            dealii::EvaluationFlags::gradients);
          return result;
        }

        mutable std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
          cached_diagonal;

        struct CellOperation
        {
          static constexpr unsigned int n_q_points =
            dealii::Utilities::pow(fe_degree + 1, dim);

          ElasticityQuadratureKernel<Form> kernel;

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
                typename Number = double,
                typename VectorType =
                  dealii::LinearAlgebra::distributed::Vector<Number>>
      class MatrixFreeScalarFormOperator
        : public dealii::MatrixFreeOperators::Base<dim, VectorType>
      {
      public:
        using Data   = dealii::MatrixFree<dim, Number>;
        using Vector = VectorType;

        MatrixFreeScalarFormOperator(std::shared_ptr<const Data> data,
                                     Form                        form)
          : kernel(std::move(form))
        {
          this->initialize(std::move(data));
        }

        /**
         * @brief Return the diagonal, computing and caching it on first use.
         * @return The owned diagonal vector (not its inverse).
         * Constrained entries are one. The first call is collective over the
         * operator's MPI communicator; all ranks must call it consistently.
         * The reference remains valid until recomputation or destruction.
         */
        const VectorType &
        get_diagonal() const
        {
          if (!cached_diagonal)
            cached_diagonal = build_diagonal();
          return cached_diagonal->get_vector();
        }

        /**
         * @brief Recompute the diagonal, replacing any cached values.
         * This collective operation invalidates earlier diagonal references.
         */
        void
        compute_diagonal() override
        {
          cached_diagonal        = build_diagonal();
          this->diagonal_entries = cached_diagonal;
        }

      private:
        std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
        build_diagonal() const
        {
          auto  result = std::make_shared<dealii::DiagonalMatrix<VectorType>>();
          auto &diagonal = result->get_vector();
          internal::compute_diagonal<dim, fe_degree, 1>(
            *this->data,
            diagonal,
            kernel,
            dealii::EvaluationFlags::values |
              dealii::EvaluationFlags::gradients);
          return result;
        }

        mutable std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
          cached_diagonal;

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

        internal::ScalarLaplaceQuadratureKernel<dim, Form> kernel;
      };

      /** @brief Portable::MatrixFree operator for scalar gradient forms. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number     = double,
                typename VectorType = dealii::LinearAlgebra::distributed::
                  Vector<Number, dealii::MemorySpace::Default>>
      class PortableMatrixFreeScalarFormOperator
      {
      public:
        using Data   = dealii::Portable::MatrixFree<dim, Number>;
        using Vector = VectorType;

        PortableMatrixFreeScalarFormOperator(std::shared_ptr<Data> data,
                                             Form                  form)
          : data(std::move(data))
          , cell_operation{internal::ScalarLaplaceQuadratureKernel<dim, Form>(
              std::move(form))}
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

        /**
         * @brief Return the diagonal, computing and caching it on first use.
         * @return The owned diagonal vector (not its inverse).
         * Constrained entries are one. The first call is collective over the
         * operator's MPI communicator; all ranks must call it consistently.
         * The reference remains valid until recomputation or destruction.
         */
        const VectorType &
        get_diagonal() const
        {
          if (!cached_diagonal)
            cached_diagonal = build_diagonal();
          return cached_diagonal->get_vector();
        }

        /**
         * @brief Recompute the diagonal, replacing any cached values.
         * This collective operation invalidates earlier diagonal references.
         */
        void
        compute_diagonal()
        {
          cached_diagonal = build_diagonal();
        }

      private:
        std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
        build_diagonal() const
        {
          auto  result = std::make_shared<dealii::DiagonalMatrix<VectorType>>();
          auto &diagonal = result->get_vector();
          internal::compute_diagonal<dim, fe_degree, 1>(
            *data,
            diagonal,
            cell_operation.kernel,
            dealii::EvaluationFlags::values |
              dealii::EvaluationFlags::gradients);
          return result;
        }

        mutable std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
          cached_diagonal;

        struct CellOperation
        {
          static constexpr unsigned int n_q_points =
            dealii::Utilities::pow(fe_degree + 1, dim);
          internal::ScalarLaplaceQuadratureKernel<dim, Form> kernel;

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

      /** @brief MatrixFree operator for the two-field Stokes form. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number = double,
                typename VectorType =
                  dealii::LinearAlgebra::distributed::BlockVector<Number>>
      class MatrixFreeStokesOperator
        : public dealii::MatrixFreeOperators::Base<dim, VectorType>
      {
      public:
        using Data   = dealii::MatrixFree<dim, Number>;
        using Vector = VectorType;
        using Kernel = StokesQuadratureKernel<dim, Form>;

        MatrixFreeStokesOperator(std::shared_ptr<const Data> data, Form form)
          : kernel(std::move(form))
        {
          this->initialize(std::move(data));
        }

        void
        initialize_dof_vector(VectorType &vector) const
        {
          this->data->initialize_dof_vector(vector);
        }

        /**
         * @brief Return the diagonal, computing and caching it on first use.
         * @return The owned diagonal vector (not its inverse).
         * Constrained entries are one. The first call is collective over the
         * operator's MPI communicator; all ranks must call it consistently.
         * The reference remains valid until recomputation or destruction.
         */
        const VectorType &
        get_diagonal() const
        {
          if (!cached_diagonal)
            cached_diagonal = build_diagonal();
          return cached_diagonal->get_vector();
        }

        /**
         * @brief Recompute the diagonal, replacing any cached values.
         * This collective operation invalidates earlier diagonal references.
         */
        void
        compute_diagonal() override
        {
          cached_diagonal        = build_diagonal();
          this->diagonal_entries = cached_diagonal;
        }

      private:
        std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
        build_diagonal() const
        {
          auto  result = std::make_shared<dealii::DiagonalMatrix<VectorType>>();
          auto &diagonal = result->get_vector();
          initialize_dof_vector(diagonal);
          diagonal = Number();
          internal::compute_diagonal<dim, fe_degree, dim>(
            *this->data,
            diagonal.block(0),
            kernel,
            dealii::EvaluationFlags::gradients);
          for (const auto index : this->data->get_constrained_dofs(1))
            diagonal.block(1).local_element(index) = Number(1);
          return result;
        }

        mutable std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
          cached_diagonal;

        void
        apply_add(VectorType &dst, const VectorType &src) const override
        {
          this->data->cell_loop(&MatrixFreeStokesOperator::local_apply,
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
          dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, dim> velocity_phi(
            mf, 0);
          dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1> pressure_phi(
            mf, 1);
          for (unsigned int cell = range.first; cell < range.second; ++cell)
            {
              velocity_phi.reinit(cell);
              pressure_phi.reinit(cell);
              velocity_phi.read_dof_values(src, 0);
              pressure_phi.read_dof_values(src, 1);
              velocity_phi.evaluate(dealii::EvaluationFlags::gradients);
              pressure_phi.evaluate(dealii::EvaluationFlags::values);
              for (unsigned int q = 0; q < velocity_phi.n_q_points; ++q)
                kernel(velocity_phi, pressure_phi, q);
              velocity_phi.integrate(dealii::EvaluationFlags::gradients);
              pressure_phi.integrate(dealii::EvaluationFlags::values);
              velocity_phi.distribute_local_to_global(dst, 0);
              pressure_phi.distribute_local_to_global(dst, 1);
            }
        }

        Kernel kernel;
      };

      /** @brief Portable::MatrixFree operator for the two-field Stokes form. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number     = double,
                typename VectorType = dealii::LinearAlgebra::distributed::
                  BlockVector<Number, dealii::MemorySpace::Default>>
      class PortableMatrixFreeStokesOperator
      {
      public:
        using Data   = dealii::Portable::MatrixFree<dim, Number>;
        using Vector = VectorType;
        using Kernel = StokesQuadratureKernel<dim, Form>;

        PortableMatrixFreeStokesOperator(std::shared_ptr<Data> data, Form form)
          : data(std::move(data))
          , cell_operation{Kernel(std::move(form))}
        {}

        void
        initialize_dof_vector(VectorType &vector) const
        {
          data->initialize_dof_vector(vector);
        }

        void
        vmult(VectorType &dst, const VectorType &src) const
        {
          dst = Number();
          data->cell_loop(cell_operation, src, dst);
          data->copy_constrained_values(src, dst);
        }

        /**
         * @brief Return the diagonal, computing and caching it on first use.
         * @return The owned diagonal vector (not its inverse).
         * Constrained entries are one. The first call is collective over the
         * operator's MPI communicator; all ranks must call it consistently.
         * The reference remains valid until recomputation or destruction.
         */
        const VectorType &
        get_diagonal() const
        {
          if (!cached_diagonal)
            cached_diagonal = build_diagonal();
          return cached_diagonal->get_vector();
        }

        /**
         * @brief Recompute the diagonal, replacing any cached values.
         * This collective operation invalidates earlier diagonal references.
         */
        void
        compute_diagonal()
        {
          cached_diagonal = build_diagonal();
        }

      private:
        std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
        build_diagonal() const
        {
          auto  result = std::make_shared<dealii::DiagonalMatrix<VectorType>>();
          auto &diagonal = result->get_vector();
          initialize_dof_vector(diagonal);
          diagonal = Number();
          internal::compute_diagonal<dim, fe_degree, dim>(
            *data,
            diagonal.block(0),
            cell_operation.kernel,
            dealii::EvaluationFlags::gradients);
          data->set_constrained_values(Number(1), diagonal.block(1), 1);
          return result;
        }

        mutable std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
          cached_diagonal;

        struct CellOperation
        {
          static constexpr unsigned int n_q_points =
            dealii::Utilities::pow(fe_degree + 1, dim);
          Kernel kernel;

          DEAL_II_HOST_DEVICE void
          operator()(const typename Data::Data *cell_data,
                     const dealii::Portable::DeviceBlockVector<Number> &src,
                     dealii::Portable::DeviceBlockVector<Number> &dst) const
          {
            dealii::Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, dim>
              velocity_phi(cell_data, 0);
            dealii::Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1>
              pressure_phi(cell_data, 1);
            velocity_phi.read_dof_values(src.block(0));
            pressure_phi.read_dof_values(src.block(1));
            velocity_phi.evaluate(dealii::EvaluationFlags::gradients);
            pressure_phi.evaluate(dealii::EvaluationFlags::values);
            for (unsigned int q = 0; q < velocity_phi.n_q_points; ++q)
              kernel(velocity_phi, pressure_phi, q);
            velocity_phi.integrate(dealii::EvaluationFlags::gradients);
            pressure_phi.integrate(dealii::EvaluationFlags::values);
            velocity_phi.distribute_local_to_global(dst.block(0));
            pressure_phi.distribute_local_to_global(dst.block(1));
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

        template <unsigned int Index, ValueShape Shape>
        struct FormFieldInfo<Trial<Index, Shape>>
        {
          static constexpr bool       found = true;
          static constexpr ValueShape shape = Shape;
        };

        template <unsigned int Index, ValueShape Shape>
        struct FormFieldInfo<Test<Index, Shape>>
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
       * Const and reference qualifiers on forms are ignored.
       */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number     = double,
                typename VectorType = typename std::conditional<
                  (FormFields<Form>::n_trial_fields > 1),
                  dealii::LinearAlgebra::distributed::BlockVector<Number>,
                  dealii::LinearAlgebra::distributed::Vector<Number>>::type>
      using MatrixFreeOperator = typename std::conditional<
        (FormFields<Form>::n_trial_fields > 1),
        MatrixFreeStokesOperator<dim,
                                 fe_degree,
                                 std::decay_t<Form>,
                                 Number,
                                 VectorType>,
        typename std::conditional<
          internal::FormFieldInfo<std::decay_t<Form>>::shape ==
            ValueShape::scalar,
          MatrixFreeScalarFormOperator<dim,
                                       fe_degree,
                                       std::decay_t<Form>,
                                       Number,
                                       VectorType>,
          MatrixFreeFormOperator<dim,
                                 fe_degree,
                                 std::decay_t<Form>,
                                 Number,
                                 VectorType>>::type>::type;

      /**
       * @brief Portable::MatrixFree operator selected from the form shape.
       * Const and reference qualifiers on forms are ignored.
       */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number     = double,
                typename VectorType = typename std::conditional<
                  (FormFields<Form>::n_trial_fields > 1),
                  dealii::LinearAlgebra::distributed::
                    BlockVector<Number, dealii::MemorySpace::Default>,
                  dealii::LinearAlgebra::distributed::
                    Vector<Number, dealii::MemorySpace::Default>>::type>
      using PortableMatrixFreeOperator = typename std::conditional<
        (FormFields<Form>::n_trial_fields > 1),
        PortableMatrixFreeStokesOperator<dim,
                                         fe_degree,
                                         std::decay_t<Form>,
                                         Number,
                                         VectorType>,
        typename std::conditional<
          internal::FormFieldInfo<std::decay_t<Form>>::shape ==
            ValueShape::scalar,
          PortableMatrixFreeScalarFormOperator<dim,
                                               fe_degree,
                                               std::decay_t<Form>,
                                               Number,
                                               VectorType>,
          PortableMatrixFreeFormOperator<dim,
                                         fe_degree,
                                         std::decay_t<Form>,
                                         Number,
                                         VectorType>>::type>::type;

      /**
       * @brief Create a CPU operator from a form with embedded coefficients.
       * @tparam dim The spatial dimension.
       * @tparam fe_degree The finite-element degree.
       * @param data The MatrixFree data, which may be const.
       * @param form The form expression, copied or moved into the operator.
       * @return The operator selected from the form and data types.
       * Const forms are accepted and stored as unqualified values.
       */
      template <int dim, int fe_degree, typename Data, typename Form>
      auto
      make_matrix_free_operator(std::shared_ptr<Data> data, Form form)
      {
        return MatrixFreeOperator<dim,
                                  fe_degree,
                                  Form,
                                  typename Data::value_type>(std::move(data),
                                                             std::move(form));
      }

      /**
       * @brief Create a Portable operator from a form with embedded coefficients.
       * @tparam dim The spatial dimension.
       * @tparam fe_degree The finite-element degree.
       * @param data The mutable Portable::MatrixFree data.
       * @param form The form expression, copied or moved into the operator.
       * @return The portable operator selected from the form and data types.
       * Const forms are accepted and stored as unqualified values.
       */
      template <int dim, int fe_degree, typename Number, typename Form>
      auto
      make_portable_matrix_free_operator(
        std::shared_ptr<dealii::Portable::MatrixFree<dim, Number>> data,
        Form                                                       form)
      {
        return PortableMatrixFreeOperator<dim, fe_degree, Form, Number>(
          std::move(data), std::move(form));
      }

      /** @brief Backwards-compatible name for the form-driven CPU operator. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number = double,
                typename VectorType =
                  dealii::LinearAlgebra::distributed::Vector<Number>>
      using MatrixFreeElasticityOperator =
        MatrixFreeFormOperator<dim, fe_degree, Form, Number, VectorType>;

      /** @brief Backwards-compatible name for the Portable form operator. */
      template <int dim,
                int fe_degree,
                typename Form,
                typename Number     = double,
                typename VectorType = dealii::LinearAlgebra::distributed::
                  Vector<Number, dealii::MemorySpace::Default>>
      using PortableMatrixFreeElasticityOperator =
        PortableMatrixFreeFormOperator<dim,
                                       fe_degree,
                                       Form,
                                       Number,
                                       VectorType>;
    } // namespace expression_templates
  } // namespace forms
} // namespace pmf

#endif // PMF_FORM_ELASTICITY_MATRIX_FREE_H
