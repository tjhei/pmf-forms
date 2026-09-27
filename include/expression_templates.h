#ifndef PMF_FORM_EXPRESSION_TEMPLATES_H
#define PMF_FORM_EXPRESSION_TEMPLATES_H

#include <form_types.h>

#include <type_traits>
#include <utility>

namespace pmf
{
  namespace forms
  {
    /** @brief Compile-time expression types used to build form descriptions. */
    namespace expression_templates
    {
      namespace internal
      {
        struct DefaultTrialTag
        {};
        struct DefaultTestTag
        {};
      } // namespace internal

      /**
       * @brief A formal trial field identified by a user-provided tag type.
       * @tparam Tag A unique C++ type identifying the field.
       * @tparam Shape The compile-time value shape of the field.
       */
      template <typename Tag, ValueShape Shape>
      struct Trial
      {
        static constexpr ValueShape shape = Shape;
      };

      /**
       * @brief A formal test field identified by a user-provided tag type.
       * @tparam Tag A unique C++ type identifying the field.
       * @tparam Shape The compile-time value shape of the field.
       */
      template <typename Tag, ValueShape Shape>
      struct Test
      {
        static constexpr ValueShape shape = Shape;
      };

      /** @brief A scalar coefficient identified by a user-provided tag type. */
      template <typename Tag>
      struct Coefficient
      {
        static constexpr ValueShape shape = ValueShape::scalar;
      };

      /** @brief A scalar constant expression. */
      template <typename Number>
      struct Constant
      {
        static constexpr ValueShape shape = ValueShape::scalar;
        Number                      value;
      };

      /** @brief Compile-time gradient operation node. */
      template <typename Expression>
      struct Gradient
      {
        static constexpr ValueShape shape =
          Expression::shape == ValueShape::scalar ? ValueShape::vector :
          Expression::shape == ValueShape::vector ? ValueShape::tensor :
                                                    ValueShape::scalar;
        Expression expression;
      };

      /** @brief Compile-time divergence operation node. */
      template <typename Expression>
      struct Divergence
      {
        static constexpr ValueShape shape = ValueShape::scalar;
        Expression                  expression;
      };

      /** @brief Compile-time symmetrization operation node. */
      template <typename Expression>
      struct Symmetrize
      {
        static constexpr ValueShape shape = ValueShape::symmetric_tensor;
        Expression                  expression;
      };

      /** @brief Compile-time sum of same-shaped expression nodes. */
      template <typename Left, typename Right>
      struct Add
      {
        static constexpr ValueShape shape = Left::shape;
        Left                        left;
        Right                       right;
      };

      /** @brief Compile-time difference of same-shaped expression nodes. */
      template <typename Left, typename Right>
      struct Subtract
      {
        static constexpr ValueShape shape = Left::shape;
        Left                        left;
        Right                       right;
      };

      /** @brief Compile-time product with a scalar operand. */
      template <typename Left, typename Right>
      struct Multiply
      {
        static constexpr ValueShape shape =
          Left::shape == ValueShape::scalar ? Right::shape : Left::shape;
        Left  left;
        Right right;
      };

      /** @brief Compile-time contraction of matching non-scalar shapes. */
      template <typename Left, typename Right>
      struct Inner
      {
        static constexpr ValueShape shape = ValueShape::scalar;
        Left                        left;
        Right                       right;
      };

      /** @brief Compile-time cell integral node. */
      template <typename Expression>
      struct Integral
      {
        Expression expression;
      };

      /** @brief Compile-time sum of integral forms. */
      template <typename Left, typename Right>
      struct FormSum
      {
        Left  left;
        Right right;
      };

      /** @brief Compile-time difference of integral forms. */
      template <typename Left, typename Right>
      struct FormDifference
      {
        Left  left;
        Right right;
      };

      template <typename T>
      struct IsExpression : std::false_type
      {};

      template <typename Tag, ValueShape Shape>
      struct IsExpression<Trial<Tag, Shape>> : std::true_type
      {};
      template <typename Tag, ValueShape Shape>
      struct IsExpression<Test<Tag, Shape>> : std::true_type
      {};
      template <typename Tag>
      struct IsExpression<Coefficient<Tag>> : std::true_type
      {};
      template <typename Number>
      struct IsExpression<Constant<Number>> : std::true_type
      {};
      template <typename Expression>
      struct IsExpression<Gradient<Expression>> : std::true_type
      {};
      template <typename Expression>
      struct IsExpression<Divergence<Expression>> : std::true_type
      {};
      template <typename Expression>
      struct IsExpression<Symmetrize<Expression>> : std::true_type
      {};
      template <typename Left, typename Right>
      struct IsExpression<Add<Left, Right>> : std::true_type
      {};
      template <typename Left, typename Right>
      struct IsExpression<Subtract<Left, Right>> : std::true_type
      {};
      template <typename Left, typename Right>
      struct IsExpression<Multiply<Left, Right>> : std::true_type
      {};
      template <typename Left, typename Right>
      struct IsExpression<Inner<Left, Right>> : std::true_type
      {};

      template <typename T>
      struct IsForm : std::false_type
      {};
      template <typename Expression>
      struct IsForm<Integral<Expression>> : std::true_type
      {};
      template <typename Left, typename Right>
      struct IsForm<FormSum<Left, Right>> : std::true_type
      {};
      template <typename Left, typename Right>
      struct IsForm<FormDifference<Left, Right>> : std::true_type
      {};

      /** @brief Type-level list used by compile-time form analysis. */
      template <typename... Types>
      struct TypeList
      {};

      namespace internal
      {
        template <typename List>
        struct TypeListSize;

        template <typename... Types>
        struct TypeListSize<TypeList<Types...>>
          : std::integral_constant<unsigned int, sizeof...(Types)>
        {};

        template <unsigned int index, typename List>
        struct TypeListAt;

        template <typename First, typename... Rest>
        struct TypeListAt<0, TypeList<First, Rest...>>
        {
          using type = First;
        };

        template <unsigned int index, typename First, typename... Rest>
        struct TypeListAt<index, TypeList<First, Rest...>>
          : TypeListAt<index - 1, TypeList<Rest...>>
        {};

        template <typename Type, typename List>
        struct TypeListContains;

        template <typename Type>
        struct TypeListContains<Type, TypeList<>> : std::false_type
        {};

        template <typename Type, typename First, typename... Rest>
        struct TypeListContains<Type, TypeList<First, Rest...>>
          : std::conditional<std::is_same<Type, First>::value,
                             std::true_type,
                             TypeListContains<Type, TypeList<Rest...>>>::type
        {};

        template <typename List, typename Type>
        struct TypeListAppendUnique;

        template <typename... Types, typename Type>
        struct TypeListAppendUnique<TypeList<Types...>, Type>
        {
          using type = typename std::conditional<
            TypeListContains<Type, TypeList<Types...>>::value,
            TypeList<Types...>,
            TypeList<Types..., Type>>::type;
        };

        template <typename Left, typename Right>
        struct TypeListMergeUnique;

        template <typename Left>
        struct TypeListMergeUnique<Left, TypeList<>>
        {
          using type = Left;
        };

        template <typename Left, typename First, typename... Rest>
        struct TypeListMergeUnique<Left, TypeList<First, Rest...>>
        {
          using appended = typename TypeListAppendUnique<Left, First>::type;
          using type =
            typename TypeListMergeUnique<appended, TypeList<Rest...>>::type;
        };

        template <typename Expression>
        struct FieldAnalysis
        {
          using trial_fields     = TypeList<>;
          using test_fields      = TypeList<>;
          using coefficient_tags = TypeList<>;
        };

        template <typename Tag, ValueShape Shape>
        struct FieldAnalysis<Trial<Tag, Shape>>
        {
          using trial_fields     = TypeList<Trial<Tag, Shape>>;
          using test_fields      = TypeList<>;
          using coefficient_tags = TypeList<>;
        };

        template <typename Tag, ValueShape Shape>
        struct FieldAnalysis<Test<Tag, Shape>>
        {
          using trial_fields     = TypeList<>;
          using test_fields      = TypeList<Test<Tag, Shape>>;
          using coefficient_tags = TypeList<>;
        };

        template <typename Tag>
        struct FieldAnalysis<Coefficient<Tag>>
        {
          using trial_fields     = TypeList<>;
          using test_fields      = TypeList<>;
          using coefficient_tags = TypeList<Coefficient<Tag>>;
        };

        template <typename Expression>
        struct FieldAnalysis<Gradient<Expression>> : FieldAnalysis<Expression>
        {};
        template <typename Expression>
        struct FieldAnalysis<Divergence<Expression>> : FieldAnalysis<Expression>
        {};
        template <typename Expression>
        struct FieldAnalysis<Symmetrize<Expression>> : FieldAnalysis<Expression>
        {};
        template <typename Expression>
        struct FieldAnalysis<Integral<Expression>> : FieldAnalysis<Expression>
        {};

#define PMF_FORM_ANALYZE_BINARY(Node)                         \
  template <typename Left, typename Right>                    \
  struct FieldAnalysis<Node<Left, Right>>                     \
  {                                                           \
    using trial_fields = typename TypeListMergeUnique<        \
      typename FieldAnalysis<Left>::trial_fields,             \
      typename FieldAnalysis<Right>::trial_fields>::type;     \
    using test_fields = typename TypeListMergeUnique<         \
      typename FieldAnalysis<Left>::test_fields,              \
      typename FieldAnalysis<Right>::test_fields>::type;      \
    using coefficient_tags = typename TypeListMergeUnique<    \
      typename FieldAnalysis<Left>::coefficient_tags,         \
      typename FieldAnalysis<Right>::coefficient_tags>::type; \
  }

        PMF_FORM_ANALYZE_BINARY(Add);
        PMF_FORM_ANALYZE_BINARY(Subtract);
        PMF_FORM_ANALYZE_BINARY(Multiply);
        PMF_FORM_ANALYZE_BINARY(Inner);
        PMF_FORM_ANALYZE_BINARY(FormSum);
        PMF_FORM_ANALYZE_BINARY(FormDifference);
#undef PMF_FORM_ANALYZE_BINARY

        template <typename Expression, typename Field>
        struct FieldRequirementsImpl
        {
          static constexpr bool value    = false;
          static constexpr bool gradient = false;
        };

        template <typename Tag, ValueShape Shape>
        struct FieldRequirementsImpl<Trial<Tag, Shape>, Trial<Tag, Shape>>
        {
          static constexpr bool value    = true;
          static constexpr bool gradient = false;
        };

        template <typename Tag, ValueShape Shape>
        struct FieldRequirementsImpl<Test<Tag, Shape>, Test<Tag, Shape>>
        {
          static constexpr bool value    = true;
          static constexpr bool gradient = false;
        };

        template <typename Expression, typename Field>
        struct FieldRequirementsImpl<Gradient<Expression>, Field>
        {
          static constexpr bool value = false;
          static constexpr bool gradient =
            TypeListContains<
              Field,
              typename FieldAnalysis<Expression>::trial_fields>::value ||
            TypeListContains<
              Field,
              typename FieldAnalysis<Expression>::test_fields>::value;
        };

        template <typename Expression, typename Field>
        struct FieldRequirementsImpl<Divergence<Expression>, Field>
          : FieldRequirementsImpl<Gradient<Expression>, Field>
        {};

        template <typename Expression, typename Field>
        struct FieldRequirementsImpl<Symmetrize<Expression>, Field>
          : FieldRequirementsImpl<Expression, Field>
        {};

        template <typename Expression, typename Field>
        struct FieldRequirementsImpl<Integral<Expression>, Field>
          : FieldRequirementsImpl<Expression, Field>
        {};

#define PMF_FORM_REQUIREMENTS_BINARY(Node)                                     \
  template <typename Left, typename Right, typename Field>                     \
  struct FieldRequirementsImpl<Node<Left, Right>, Field>                       \
  {                                                                            \
    static constexpr bool value = FieldRequirementsImpl<Left, Field>::value || \
                                  FieldRequirementsImpl<Right, Field>::value;  \
    static constexpr bool gradient =                                           \
      FieldRequirementsImpl<Left, Field>::gradient ||                          \
      FieldRequirementsImpl<Right, Field>::gradient;                           \
  }

        PMF_FORM_REQUIREMENTS_BINARY(Add);
        PMF_FORM_REQUIREMENTS_BINARY(Subtract);
        PMF_FORM_REQUIREMENTS_BINARY(Multiply);
        PMF_FORM_REQUIREMENTS_BINARY(Inner);
        PMF_FORM_REQUIREMENTS_BINARY(FormSum);
        PMF_FORM_REQUIREMENTS_BINARY(FormDifference);
#undef PMF_FORM_REQUIREMENTS_BINARY

        template <typename Expression, typename Field>
        struct FieldRequirements
          : FieldRequirementsImpl<Expression, typename std::decay<Field>::type>
        {};
      } // namespace internal

      /**
       * @brief Compile-time trial/test fields and evaluation needs of a form.
       *
       * `trial_fields` and `test_fields` are unique type lists. Query
       * `internal::FieldRequirements<Form, Field>` for value/gradient use.
       */
      template <typename Form>
      struct FormFields : internal::FieldAnalysis<Form>
      {
        static constexpr unsigned int n_trial_fields = internal::TypeListSize<
          typename internal::FieldAnalysis<Form>::trial_fields>::value;
        static constexpr unsigned int n_test_fields = internal::TypeListSize<
          typename internal::FieldAnalysis<Form>::test_fields>::value;
        using coefficient_tags =
          typename internal::FieldAnalysis<Form>::coefficient_tags;
        static constexpr unsigned int n_coefficients =
          internal::TypeListSize<coefficient_tags>::value;
      };

      /** @brief Create a typed formal trial field. */
      template <typename Tag, ValueShape Shape>
      constexpr Trial<Tag, Shape>
      trial()
      {
        return {};
      }

      /**
       * @brief Create the default formal trial field without a user tag.
       * @tparam Shape The compile-time value shape, scalar by default.
       *
       * This shorthand is intended for forms with one trial field. Give fields
       * explicit tags when a form contains multiple trial arguments.
       */
      template <ValueShape Shape = ValueShape::scalar>
      constexpr Trial<internal::DefaultTrialTag, Shape>
      trial()
      {
        return {};
      }

      /** @brief Create a typed formal test field. */
      template <typename Tag, ValueShape Shape>
      constexpr Test<Tag, Shape>
      test()
      {
        return {};
      }

      /**
       * @brief Create the default formal test field without a user tag.
       * @tparam Shape The compile-time value shape, scalar by default.
       *
       * This shorthand is intended for forms with one test field. Give fields
       * explicit tags when a form contains multiple test arguments.
       */
      template <ValueShape Shape = ValueShape::scalar>
      constexpr Test<internal::DefaultTestTag, Shape>
      test()
      {
        return {};
      }

      /** @brief Create a typed scalar coefficient field. */
      template <typename Tag>
      constexpr Coefficient<Tag>
      coefficient()
      {
        return {};
      }

      /** @brief Create a scalar constant expression. */
      template <typename Number>
      constexpr Constant<Number>
      constant(Number value)
      {
        return {value};
      }

      /** @brief Apply the compile-time gradient operation. */
      template <typename Expression>
      constexpr auto
      grad(Expression expression)
      {
        static_assert(Expression::shape == ValueShape::scalar ||
                        Expression::shape == ValueShape::vector,
                      "grad requires a scalar or vector expression");
        return Gradient<Expression>{std::move(expression)};
      }

      /** @brief Apply the compile-time divergence operation. */
      template <typename Expression>
      constexpr auto
      div(Expression expression)
      {
        static_assert(Expression::shape == ValueShape::vector,
                      "div requires a vector expression");
        return Divergence<Expression>{std::move(expression)};
      }

      /** @brief Apply the compile-time symmetrization operation. */
      template <typename Expression>
      constexpr auto
      sym(Expression expression)
      {
        static_assert(Expression::shape == ValueShape::tensor,
                      "sym requires a rank-2 tensor expression");
        return Symmetrize<Expression>{std::move(expression)};
      }

      /** @brief Contract matching vector or tensor expression nodes. */
      template <typename Left, typename Right>
      constexpr auto
      inner(Left left, Right right)
      {
        static_assert(Left::shape == Right::shape,
                      "inner requires matching value shapes");
        static_assert(Left::shape != ValueShape::scalar,
                      "inner requires vector or tensor operands");
        return Inner<Left, Right>{std::move(left), std::move(right)};
      }

      /** @brief Add two same-shaped expression nodes. */
      template <typename Left,
                typename Right,
                typename std::enable_if<IsExpression<Left>::value &&
                                          IsExpression<Right>::value,
                                        int>::type = 0>
      constexpr auto
      operator+(Left left, Right right)
      {
        static_assert(Left::shape == Right::shape,
                      "addition requires matching value shapes");
        return Add<Left, Right>{std::move(left), std::move(right)};
      }

      /** @brief Subtract two same-shaped expression nodes. */
      template <typename Left,
                typename Right,
                typename std::enable_if<IsExpression<Left>::value &&
                                          IsExpression<Right>::value,
                                        int>::type = 0>
      constexpr auto
      operator-(Left left, Right right)
      {
        static_assert(Left::shape == Right::shape,
                      "subtraction requires matching value shapes");
        return Subtract<Left, Right>{std::move(left), std::move(right)};
      }

      /** @brief Multiply expression nodes when at least one is scalar. */
      template <typename Left,
                typename Right,
                typename std::enable_if<IsExpression<Left>::value &&
                                          IsExpression<Right>::value,
                                        int>::type = 0>
      constexpr auto
      operator*(Left left, Right right)
      {
        static_assert(Left::shape == ValueShape::scalar ||
                        Right::shape == ValueShape::scalar,
                      "multiplication requires a scalar operand");
        return Multiply<Left, Right>{std::move(left), std::move(right)};
      }

      /** @brief Scale an expression node by a scalar on the right. */
      template <typename Expression,
                typename Number,
                typename std::enable_if<IsExpression<Expression>::value &&
                                          std::is_arithmetic<Number>::value,
                                        int>::type = 0>
      constexpr auto
      operator*(Expression expression, Number value)
      {
        return std::move(expression) * constant(value);
      }

      /** @brief Scale an expression node by a scalar on the left. */
      template <typename Number,
                typename Expression,
                typename std::enable_if<std::is_arithmetic<Number>::value &&
                                          IsExpression<Expression>::value,
                                        int>::type = 0>
      constexpr auto
      operator*(Number value, Expression expression)
      {
        return constant(value) * std::move(expression);
      }

      /** @brief Integrate a scalar expression over cells. */
      template <
        typename Expression,
        typename std::enable_if<IsExpression<Expression>::value, int>::type = 0>
      constexpr auto
      integral(Expression expression, CellMeasure)
      {
        static_assert(Expression::shape == ValueShape::scalar,
                      "integral requires a scalar expression");
        return Integral<Expression>{std::move(expression)};
      }

      /** @brief Add two integral forms. */
      template <
        typename Left,
        typename Right,
        typename std::enable_if<IsForm<Left>::value && IsForm<Right>::value,
                                int>::type = 0>
      constexpr auto
      operator+(Left left, Right right)
      {
        return FormSum<Left, Right>{std::move(left), std::move(right)};
      }

      /** @brief Subtract one integral form from another. */
      template <
        typename Left,
        typename Right,
        typename std::enable_if<IsForm<Left>::value && IsForm<Right>::value,
                                int>::type = 0>
      constexpr auto
      operator-(Left left, Right right)
      {
        return FormDifference<Left, Right>{std::move(left), std::move(right)};
      }
    } // namespace expression_templates
  }   // namespace forms
} // namespace pmf

#endif // PMF_FORM_EXPRESSION_TEMPLATES_H
