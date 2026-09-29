#ifndef PMF_FORM_COEFFICIENT_EVALUATION_H
#define PMF_FORM_COEFFICIENT_EVALUATION_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <type_traits>

namespace pmf
{
  namespace forms
  {
    namespace expression_templates
    {
      namespace internal
      {
        template <int dim, typename Number, typename Provider>
        DEAL_II_HOST_DEVICE Number
        evaluate_coefficient_at_point(const Provider           &provider,
                                      const dealii::Point<dim> &point)
        {
          if constexpr (std::is_invocable_v<Provider, dealii::Point<dim>>)
            return static_cast<Number>(provider(point));
          else
            return static_cast<Number>(provider.value(point, 0));
        }
      } // namespace internal

      /**
       * @brief Evaluate a deal.II function as a spatially varying coefficient.
       * @tparam dim The spatial dimension of the coefficient domain.
       * @tparam Number The scalar type returned at quadrature points.
       */
      template <int dim, typename Number = double>
      class FunctionCoefficient
      {
      public:
        FunctionCoefficient() = default;

        /** @brief Bind a function without taking ownership. */
        explicit FunctionCoefficient(const dealii::Function<dim> &function)
          : function(&function)
        {}

        /** @brief Evaluate the bound function at a physical point. */
        Number
        operator()(const dealii::Point<dim> &point) const
        {
          return function->value(point);
        }

      private:
        const dealii::Function<dim> *function = nullptr;
      };
    } // namespace expression_templates
  } // namespace forms
} // namespace pmf

#endif
