#ifndef PMF_FORM_FORMS_H
#define PMF_FORM_FORMS_H

#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pmf
{
  namespace forms
  {
    enum class ValueShape
    {
      scalar,
      vector,
      tensor,
      symmetric_tensor
    };

    enum class SymbolKind
    {
      trial,
      test,
      coefficient
    };

    namespace internal
    {
      enum class NodeKind
      {
        symbol,
        constant,
        add,
        subtract,
        multiply,
        gradient,
        divergence,
        symmetrize,
        inner
      };

      struct Node
      {
        Node(NodeKind kind, ValueShape shape)
          : kind(kind)
          , shape(shape)
        {}

        NodeKind                    kind;
        ValueShape                  shape;
        std::string                 name;
        SymbolKind                  symbol_kind = SymbolKind::coefficient;
        double                      value       = 0.0;
        std::shared_ptr<const Node> left;
        std::shared_ptr<const Node> right;
      };
    } // namespace internal

    class Expression
    {
    public:
      Expression() = default;

      ValueShape
      shape() const
      {
        ensure_valid();
        return node->shape;
      }

      bool
      valid() const
      {
        return static_cast<bool>(node);
      }

      std::string
      str() const;

    private:
      explicit Expression(std::shared_ptr<const internal::Node> node)
        : node(std::move(node))
      {}

      void
      ensure_valid() const
      {
        if (!node)
          throw std::logic_error("use of an empty form expression");
      }

      std::shared_ptr<const internal::Node> node;

      friend Expression
      trial(const std::string &, ValueShape);
      friend Expression
      test(const std::string &, ValueShape);
      friend Expression
      coefficient(const std::string &);
      friend Expression
      constant(double);
      friend Expression
      operator+(const Expression &, const Expression &);
      friend Expression
      operator-(const Expression &, const Expression &);
      friend Expression
      operator*(const Expression &, const Expression &);
      friend Expression
      operator-(const Expression &);
      friend Expression
      grad(const Expression &);
      friend Expression
      div(const Expression &);
      friend Expression
      sym(const Expression &);
      friend Expression
      inner(const Expression &, const Expression &);
      friend class Form;
      friend Expression
      make_symbol(const std::string &, SymbolKind, ValueShape);
      friend Expression
      make_binary(internal::NodeKind,
                  const Expression &,
                  const Expression &,
                  ValueShape);
    };

    inline Expression
    make_symbol(const std::string &name,
                const SymbolKind   kind,
                const ValueShape   shape)
    {
      auto node =
        std::make_shared<internal::Node>(internal::NodeKind::symbol, shape);
      node->name        = name;
      node->symbol_kind = kind;
      return Expression(std::move(node));
    }

    inline Expression
    trial(const std::string &name, const ValueShape shape)
    {
      return make_symbol(name, SymbolKind::trial, shape);
    }

    inline Expression
    test(const std::string &name, const ValueShape shape)
    {
      return make_symbol(name, SymbolKind::test, shape);
    }

    inline Expression
    coefficient(const std::string &name)
    {
      return make_symbol(name, SymbolKind::coefficient, ValueShape::scalar);
    }

    inline Expression
    constant(const double value)
    {
      auto node = std::make_shared<internal::Node>(internal::NodeKind::constant,
                                                   ValueShape::scalar);
      node->value = value;
      return Expression(std::move(node));
    }

    inline void
    require_shape(const Expression &expression,
                  const ValueShape  expected,
                  const char       *operation)
    {
      if (expression.shape() != expected)
        throw std::invalid_argument(std::string(operation) +
                                    " received an expression with an invalid "
                                    "value shape");
    }

    inline Expression
    make_binary(const internal::NodeKind kind,
                const Expression        &left,
                const Expression        &right,
                const ValueShape         shape)
    {
      left.ensure_valid();
      right.ensure_valid();
      auto node   = std::make_shared<internal::Node>(kind, shape);
      node->left  = left.node;
      node->right = right.node;
      return Expression(std::move(node));
    }

    inline Expression
    operator+(const Expression &left, const Expression &right)
    {
      if (left.shape() != right.shape())
        throw std::invalid_argument("addition requires matching value shapes");
      return make_binary(internal::NodeKind::add, left, right, left.shape());
    }

    inline Expression
    operator-(const Expression &left, const Expression &right)
    {
      if (left.shape() != right.shape())
        throw std::invalid_argument(
          "subtraction requires matching value shapes");
      return make_binary(internal::NodeKind::subtract,
                         left,
                         right,
                         left.shape());
    }

    inline Expression
    operator*(const Expression &left, const Expression &right)
    {
      const auto shape =
        left.shape() == ValueShape::scalar  ? right.shape() :
        right.shape() == ValueShape::scalar ? left.shape() :
                                              ValueShape::scalar;
      if (left.shape() != ValueShape::scalar &&
          right.shape() != ValueShape::scalar)
        throw std::invalid_argument(
          "multiplication requires at least one scalar operand");
      return make_binary(internal::NodeKind::multiply, left, right, shape);
    }

    inline Expression
    operator*(const Expression &left, const double right)
    {
      return left * constant(right);
    }

    inline Expression
    operator*(const double left, const Expression &right)
    {
      return constant(left) * right;
    }

    inline Expression
    operator-(const Expression &expression)
    {
      return constant(-1.0) * expression;
    }

    inline Expression
    grad(const Expression &expression)
    {
      ValueShape result;
      switch (expression.shape())
        {
          case ValueShape::scalar:
            result = ValueShape::vector;
            break;
          case ValueShape::vector:
            result = ValueShape::tensor;
            break;
          default:
            throw std::invalid_argument(
              "grad is only defined for scalar and vector expressions");
        }
      auto node =
        std::make_shared<internal::Node>(internal::NodeKind::gradient, result);
      node->left = expression.node;
      return Expression(std::move(node));
    }

    inline Expression
    div(const Expression &expression)
    {
      require_shape(expression, ValueShape::vector, "div");
      auto node =
        std::make_shared<internal::Node>(internal::NodeKind::divergence,
                                         ValueShape::scalar);
      node->left = expression.node;
      return Expression(std::move(node));
    }

    inline Expression
    sym(const Expression &expression)
    {
      require_shape(expression, ValueShape::tensor, "sym");
      auto node =
        std::make_shared<internal::Node>(internal::NodeKind::symmetrize,
                                         ValueShape::symmetric_tensor);
      node->left = expression.node;
      return Expression(std::move(node));
    }

    inline Expression
    inner(const Expression &left, const Expression &right)
    {
      if (left.shape() != right.shape() || left.shape() == ValueShape::scalar)
        throw std::invalid_argument(
          "inner requires matching vector or tensor value shapes");
      return make_binary(internal::NodeKind::inner,
                         left,
                         right,
                         ValueShape::scalar);
    }

    struct CellMeasure
    {};

    constexpr CellMeasure dx{};

    class Form
    {
    public:
      struct IntegralTerm
      {
        Expression expression;
        int        sign;
      };

      const std::vector<IntegralTerm> &
      integrals() const
      {
        return terms;
      }

      std::string
      str() const;

    private:
      explicit Form(Expression expression)
      {
        terms.push_back({std::move(expression), 1});
      }

      std::vector<IntegralTerm> terms;

      friend Form
      integral(const Expression &, CellMeasure);
      friend Form
      operator+(const Form &, const Form &);
      friend Form
      operator-(const Form &, const Form &);
    };

    inline Form
    integral(const Expression &expression, CellMeasure)
    {
      require_shape(expression, ValueShape::scalar, "integral");
      return Form(expression);
    }

    inline Form
    operator+(const Form &left, const Form &right)
    {
      Form result = left;
      result.terms.insert(result.terms.end(),
                          right.terms.begin(),
                          right.terms.end());
      return result;
    }

    inline Form
    operator-(const Form &left, const Form &right)
    {
      Form result = left;
      for (const auto &term : right.terms)
        result.terms.push_back({term.expression, -term.sign});
      return result;
    }

    inline std::string
    shape_name(const ValueShape shape)
    {
      switch (shape)
        {
          case ValueShape::scalar:
            return "scalar";
          case ValueShape::vector:
            return "vector";
          case ValueShape::tensor:
            return "tensor";
          case ValueShape::symmetric_tensor:
            return "symmetric_tensor";
        }
      return "unknown";
    }

    inline std::string
    print_node(const std::shared_ptr<const internal::Node> &node)
    {
      using internal::NodeKind;
      switch (node->kind)
        {
          case NodeKind::symbol:
            if (node->symbol_kind == SymbolKind::trial)
              return "trial(" + node->name + ")";
            if (node->symbol_kind == SymbolKind::test)
              return "test(" + node->name + ")";
            return "coefficient(" + node->name + ")";
          case NodeKind::constant:
            {
              std::ostringstream out;
              out << node->value;
              return out.str();
            }
          case NodeKind::add:
            return "(" + print_node(node->left) + " + " +
                   print_node(node->right) + ")";
          case NodeKind::subtract:
            return "(" + print_node(node->left) + " - " +
                   print_node(node->right) + ")";
          case NodeKind::multiply:
            return "(" + print_node(node->left) + " * " +
                   print_node(node->right) + ")";
          case NodeKind::gradient:
            return "grad(" + print_node(node->left) + ")";
          case NodeKind::divergence:
            return "div(" + print_node(node->left) + ")";
          case NodeKind::symmetrize:
            return "sym(" + print_node(node->left) + ")";
          case NodeKind::inner:
            return "inner(" + print_node(node->left) + ", " +
                   print_node(node->right) + ")";
        }
      return "<invalid expression>";
    }

    inline std::string
    Expression::str() const
    {
      ensure_valid();
      return print_node(node);
    }

    inline std::string
    Form::str() const
    {
      std::ostringstream out;
      for (std::size_t i = 0; i < terms.size(); ++i)
        {
          if (i != 0)
            out << (terms[i].sign < 0 ? " - " : " + ");
          else if (terms[i].sign < 0)
            out << "-";
          out << "integral(" << terms[i].expression.str() << ", dx)";
        }
      return out.str();
    }

    inline std::ostream &
    operator<<(std::ostream &out, const Expression &expression)
    {
      return out << expression.str();
    }

    inline std::ostream &
    operator<<(std::ostream &out, const Form &form)
    {
      return out << form.str();
    }
  } // namespace forms
} // namespace pmf

#endif // PMF_FORM_FORMS_H
