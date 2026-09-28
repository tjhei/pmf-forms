#ifndef PMF_FORM_CELL_FORM_LOWERING_H
#define PMF_FORM_CELL_FORM_LOWERING_H

#include <deal.II/base/tensor.h>

#include <expression_templates.h>

namespace pmf
{
  namespace forms
  {
    namespace expression_templates
    {
      namespace internal
      {
        template <typename Context>
        struct LoweringAlgebra
        {
          template <typename Number>
          DEAL_II_HOST_DEVICE static auto
          constant(const Context &, const Number value)
          {
            return typename Context::Number(value);
          }

          template <typename Number>
          DEAL_II_HOST_DEVICE static auto
          coefficient(const Context &context, const Number value)
          {
            return constant(context, value);
          }

          template <typename Tensor>
          DEAL_II_HOST_DEVICE static auto
          trace(const Tensor &tensor)
          {
            return dealii::trace(tensor);
          }

          template <typename Tensor>
          DEAL_II_HOST_DEVICE static auto
          symmetrize(const Tensor &tensor)
          {
            return typename Context::Number(0.5) *
                   (tensor + dealii::transpose(tensor));
          }

          template <typename Left, typename Right>
          DEAL_II_HOST_DEVICE static auto
          scalar_product(const Left &left, const Right &right)
          {
            return dealii::scalar_product(left, right);
          }
        };

        template <typename Expression>
        struct PolynomialDegree
        {
          static constexpr unsigned int trial = 0;
          static constexpr unsigned int test  = 0;
          static constexpr bool         valid = false;
        };

        template <unsigned int Index, ValueShape Shape>
        struct PolynomialDegree<Trial<Index, Shape>>
        {
          static constexpr unsigned int trial = 1;
          static constexpr unsigned int test  = 0;
          static constexpr bool         valid = true;
        };

        template <unsigned int Index, ValueShape Shape>
        struct PolynomialDegree<Test<Index, Shape>>
        {
          static constexpr unsigned int trial = 0;
          static constexpr unsigned int test  = 1;
          static constexpr bool         valid = true;
        };

        template <typename Number>
        struct PolynomialDegree<Constant<Number>>
        {
          static constexpr unsigned int trial = 0;
          static constexpr unsigned int test  = 0;
          static constexpr bool         valid = true;
        };

        template <typename Number>
        struct PolynomialDegree<Coefficient<Number>>
          : PolynomialDegree<Constant<Number>>
        {};

        template <typename Expression>
        struct PolynomialDegree<Gradient<Expression>>
          : PolynomialDegree<Expression>
        {};
        template <typename Expression>
        struct PolynomialDegree<Divergence<Expression>>
          : PolynomialDegree<Expression>
        {};
        template <typename Expression>
        struct PolynomialDegree<Symmetrize<Expression>>
          : PolynomialDegree<Expression>
        {};
        template <typename Expression>
        struct PolynomialDegree<Integral<Expression>>
          : PolynomialDegree<Expression>
        {};

        template <typename Left, typename Right>
        struct PolynomialDegree<Add<Left, Right>>
        {
          using LeftDegree                    = PolynomialDegree<Left>;
          using RightDegree                   = PolynomialDegree<Right>;
          static constexpr unsigned int trial = LeftDegree::trial;
          static constexpr unsigned int test  = LeftDegree::test;
          static constexpr bool         valid =
            LeftDegree::valid && RightDegree::valid &&
            trial == RightDegree::trial && test == RightDegree::test;
        };
        template <typename Left, typename Right>
        struct PolynomialDegree<Subtract<Left, Right>>
          : PolynomialDegree<Add<Left, Right>>
        {};
        template <typename Left, typename Right>
        struct PolynomialDegree<FormSum<Left, Right>>
          : PolynomialDegree<Add<Left, Right>>
        {};
        template <typename Left, typename Right>
        struct PolynomialDegree<FormDifference<Left, Right>>
          : PolynomialDegree<Add<Left, Right>>
        {};

        template <typename Left, typename Right>
        struct PolynomialDegree<Multiply<Left, Right>>
        {
          using LeftDegree  = PolynomialDegree<Left>;
          using RightDegree = PolynomialDegree<Right>;
          static constexpr unsigned int trial =
            LeftDegree::trial + RightDegree::trial;
          static constexpr unsigned int test =
            LeftDegree::test + RightDegree::test;
          static constexpr bool valid =
            LeftDegree::valid && RightDegree::valid && trial <= 1 && test <= 1;
        };
        template <typename Left, typename Right>
        struct PolynomialDegree<Inner<Left, Right>>
          : PolynomialDegree<Multiply<Left, Right>>
        {};

        template <typename Form>
        constexpr bool is_bilinear_cell_form =
          PolynomialDegree<Form>::valid && PolynomialDegree<Form>::trial == 1 &&
          PolynomialDegree<Form>::test == 1;

        template <typename Expression>
        struct Lower;

        template <unsigned int Index, ValueShape Shape>
        struct Lower<Trial<Index, Shape>>
        {
          template <typename Context>
          DEAL_II_HOST_DEVICE static auto
          evaluate(const Trial<Index, Shape> &, const Context &context)
          {
            return context.template value<Trial<Index, Shape>>();
          }
        };

        template <unsigned int Index, ValueShape Shape>
        struct Lower<Test<Index, Shape>>
        {
          template <typename Adjoint, typename Context>
          DEAL_II_HOST_DEVICE static void
          submit(const Test<Index, Shape> &,
                 const Adjoint &adjoint,
                 Context       &context)
          {
            context.template submit_value<Test<Index, Shape>>(adjoint);
          }
        };

        template <typename Number>
        struct Lower<Constant<Number>>
        {
          template <typename Context>
          DEAL_II_HOST_DEVICE static auto
          evaluate(const Constant<Number> &expression, const Context &context)
          {
            return LoweringAlgebra<Context>::constant(context,
                                                      expression.value);
          }
        };

        template <typename Number>
        struct Lower<Coefficient<Number>>
        {
          template <typename Context>
          DEAL_II_HOST_DEVICE static auto
          evaluate(const Coefficient<Number> &expression,
                   const Context             &context)
          {
            return LoweringAlgebra<Context>::coefficient(context,
                                                         expression.value);
          }
        };

        template <unsigned int Index, ValueShape Shape>
        struct Lower<Gradient<Trial<Index, Shape>>>
        {
          template <typename Context>
          DEAL_II_HOST_DEVICE static auto
          evaluate(const Gradient<Trial<Index, Shape>> &,
                   const Context &context)
          {
            return context.template gradient<Trial<Index, Shape>>();
          }
        };

        template <unsigned int Index, ValueShape Shape>
        struct Lower<Gradient<Test<Index, Shape>>>
        {
          template <typename Adjoint, typename Context>
          DEAL_II_HOST_DEVICE static void
          submit(const Gradient<Test<Index, Shape>> &,
                 const Adjoint &adjoint,
                 Context       &context)
          {
            context.template submit_gradient<Test<Index, Shape>>(adjoint);
          }
        };

        template <unsigned int Index>
        struct Lower<Divergence<Trial<Index, ValueShape::vector>>>
        {
          template <typename Context>
          DEAL_II_HOST_DEVICE static auto
          evaluate(const Divergence<Trial<Index, ValueShape::vector>> &,
                   const Context &context)
          {
            return LoweringAlgebra<Context>::trace(
              context.template gradient<Trial<Index, ValueShape::vector>>());
          }
        };

        template <unsigned int Index>
        struct Lower<Divergence<Test<Index, ValueShape::vector>>>
        {
          template <typename Adjoint, typename Context>
          DEAL_II_HOST_DEVICE static void
          submit(const Divergence<Test<Index, ValueShape::vector>> &,
                 const Adjoint &adjoint,
                 Context       &context)
          {
            context.template submit_divergence<Test<Index, ValueShape::vector>>(
              adjoint);
          }
        };

        template <typename Expression>
        struct Lower<Symmetrize<Expression>>
        {
          template <typename Context>
          DEAL_II_HOST_DEVICE static auto
          evaluate(const Symmetrize<Expression> &expression,
                   const Context                &context)
          {
            const auto tensor =
              Lower<Expression>::evaluate(expression.expression, context);
            return LoweringAlgebra<Context>::symmetrize(tensor);
          }

          template <typename Adjoint, typename Context>
          DEAL_II_HOST_DEVICE static void
          submit(const Symmetrize<Expression> &expression,
                 const Adjoint                &adjoint,
                 Context                      &context)
          {
            Lower<Expression>::submit(expression.expression,
                                      LoweringAlgebra<Context>::symmetrize(
                                        adjoint),
                                      context);
          }
        };

        template <typename Left, typename Right, bool subtract>
        struct LowerSum
        {
          template <typename Expression, typename Context>
          DEAL_II_HOST_DEVICE static auto
          evaluate(const Expression &expression, const Context &context)
          {
            if constexpr (subtract)
              return Lower<Left>::evaluate(expression.left, context) -
                     Lower<Right>::evaluate(expression.right, context);
            else
              return Lower<Left>::evaluate(expression.left, context) +
                     Lower<Right>::evaluate(expression.right, context);
          }

          template <typename Expression, typename Adjoint, typename Context>
          DEAL_II_HOST_DEVICE static void
          submit(const Expression &expression,
                 const Adjoint    &adjoint,
                 Context          &context)
          {
            Lower<Left>::submit(expression.left, adjoint, context);
            if constexpr (subtract)
              Lower<Right>::submit(expression.right, -adjoint, context);
            else
              Lower<Right>::submit(expression.right, adjoint, context);
          }
        };

        template <typename Left, typename Right>
        struct Lower<Add<Left, Right>> : LowerSum<Left, Right, false>
        {};
        template <typename Left, typename Right>
        struct Lower<Subtract<Left, Right>> : LowerSum<Left, Right, true>
        {};
        template <typename Left, typename Right>
        struct Lower<FormSum<Left, Right>> : LowerSum<Left, Right, false>
        {};
        template <typename Left, typename Right>
        struct Lower<FormDifference<Left, Right>> : LowerSum<Left, Right, true>
        {};

        template <typename Left, typename Right>
        struct Lower<Multiply<Left, Right>>
        {
          template <typename Context>
          DEAL_II_HOST_DEVICE static auto
          evaluate(const Multiply<Left, Right> &expression,
                   const Context               &context)
          {
            return Lower<Left>::evaluate(expression.left, context) *
                   Lower<Right>::evaluate(expression.right, context);
          }

          template <typename TestExpression,
                    typename OtherExpression,
                    typename Adjoint,
                    typename Context>
          DEAL_II_HOST_DEVICE static void
          propagate(const TestExpression  &test_expression,
                    const OtherExpression &other,
                    const Adjoint         &adjoint,
                    Context               &context)
          {
            const auto value = Lower<OtherExpression>::evaluate(other, context);
            if constexpr (TestExpression::shape == ValueShape::scalar &&
                          OtherExpression::shape != ValueShape::scalar)
              Lower<TestExpression>::submit(
                test_expression,
                LoweringAlgebra<Context>::scalar_product(adjoint, value),
                context);
            else
              Lower<TestExpression>::submit(test_expression,
                                            adjoint * value,
                                            context);
          }

          template <typename Adjoint, typename Context>
          DEAL_II_HOST_DEVICE static void
          submit(const Multiply<Left, Right> &expression,
                 const Adjoint               &adjoint,
                 Context                     &context)
          {
            if constexpr (PolynomialDegree<Left>::test == 1)
              propagate(expression.left, expression.right, adjoint, context);
            else
              propagate(expression.right, expression.left, adjoint, context);
          }
        };

        template <typename Left, typename Right>
        struct Lower<Inner<Left, Right>>
        {
          template <typename Context>
          DEAL_II_HOST_DEVICE static auto
          evaluate(const Inner<Left, Right> &expression, const Context &context)
          {
            return LoweringAlgebra<Context>::scalar_product(
              Lower<Left>::evaluate(expression.left, context),
              Lower<Right>::evaluate(expression.right, context));
          }

          template <typename Adjoint, typename Context>
          DEAL_II_HOST_DEVICE static void
          submit(const Inner<Left, Right> &expression,
                 const Adjoint            &adjoint,
                 Context                  &context)
          {
            if constexpr (PolynomialDegree<Left>::test == 1)
              Lower<Left>::submit(expression.left,
                                  adjoint *
                                    Lower<Right>::evaluate(expression.right,
                                                           context),
                                  context);
            else
              Lower<Right>::submit(expression.right,
                                   adjoint *
                                     Lower<Left>::evaluate(expression.left,
                                                           context),
                                   context);
          }
        };

        template <typename Expression>
        struct Lower<Integral<Expression>>
        {
          template <typename Adjoint, typename Context>
          DEAL_II_HOST_DEVICE static void
          submit(const Integral<Expression> &form,
                 const Adjoint              &adjoint,
                 Context                    &context)
          {
            Lower<Expression>::submit(form.expression, adjoint, context);
          }
        };
      } // namespace internal

      /**
       * @brief Compile-time lowering of a bilinear cell form to test submissions.
       * @tparam Form A scalar cell integral or sum/difference of cell integrals.
       * Each term must be linear in trial fields and linear in test fields.
       * Supported operations are values, field gradients/divergences,
       * symmetrization, scalar products, scaling, addition and subtraction.
       */
      template <typename Form>
      class BilinearCellKernel
      {
        static_assert(
          IsForm<Form>::value,
          "expected a cell integral or sum/difference of integrals");
        static_assert(
          internal::is_bilinear_cell_form<Form>,
          "cell forms must be homogeneous and linear in trial and test fields");

      public:
        /** @brief Store an owned form expression. @param form The cell form. */
        explicit BilinearCellKernel(Form form)
          : form(std::move(form))
        {}

        /**
         * @brief Accumulate test values and gradients at one quadrature point.
         * @param context Field-indexed trial reads and test accumulation buffers.
         * The caller submits the buffers after all terms have been accumulated.
         */
        template <typename Context>
        DEAL_II_HOST_DEVICE void
        operator()(Context &context) const
        {
          internal::Lower<Form>::submit(
            form,
            internal::LoweringAlgebra<Context>::constant(context, 1),
            context);
        }

      private:
        Form form;
      };
    } // namespace expression_templates
  } // namespace forms
} // namespace pmf

#endif
