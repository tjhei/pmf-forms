#ifndef PMF_FORM_INSPECT_LOWERING_H
#define PMF_FORM_INSPECT_LOWERING_H

#include <cell_form_lowering.h>
#include <lowering_ir.h>

#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <sstream>
#include <tuple>

namespace pmf
{
  namespace forms
  {
    namespace expression_templates
    {
      namespace internal
      {
        struct InspectionNode;
        using InspectionValue = std::shared_ptr<const InspectionNode>;

        struct InspectionNode
        {
          LoweringOpcode               opcode;
          std::vector<InspectionValue> operands;
          unsigned int                 field = 0;
          std::string                  literal;
        };

        inline InspectionValue
        inspection_node(const LoweringOpcode         opcode,
                        std::vector<InspectionValue> operands = {},
                        const unsigned int           field    = 0,
                        std::string                  literal  = {})
        {
          return std::make_shared<InspectionNode>(InspectionNode{
            opcode, std::move(operands), field, std::move(literal)});
        }

        inline bool
        inspection_unit(const InspectionValue &value, const char *literal)
        {
          return value->opcode == LoweringOpcode::constant &&
                 value->literal == literal;
        }

        inline InspectionValue
        operator-(const InspectionValue &value)
        {
          if (inspection_unit(value, "1"))
            return inspection_node(LoweringOpcode::constant, {}, 0, "-1");
          if (value->opcode == LoweringOpcode::negate)
            return value->operands[0];
          return inspection_node(LoweringOpcode::negate, {value});
        }

        inline InspectionValue
        operator+(const InspectionValue &left, const InspectionValue &right)
        {
          return inspection_node(LoweringOpcode::add, {left, right});
        }

        inline InspectionValue
        operator-(const InspectionValue &left, const InspectionValue &right)
        {
          return inspection_node(LoweringOpcode::subtract, {left, right});
        }

        inline InspectionValue
        operator*(const InspectionValue &left, const InspectionValue &right)
        {
          if (inspection_unit(left, "1"))
            return right;
          if (inspection_unit(right, "1"))
            return left;
          if (inspection_unit(left, "-1"))
            return -right;
          if (inspection_unit(right, "-1"))
            return -left;
          return inspection_node(LoweringOpcode::multiply, {left, right});
        }

        struct InspectionContext
        {
          std::map<std::pair<unsigned int, bool>, InspectionValue> submitted;

          template <typename Field>
          InspectionValue
          value() const
          {
            return inspection_node(LoweringOpcode::get_value, {}, Field::index);
          }

          template <typename Field>
          InspectionValue
          gradient() const
          {
            return inspection_node(LoweringOpcode::get_gradient,
                                   {},
                                   Field::index);
          }

          void
          accumulate(const unsigned int     field,
                     const bool             gradient,
                     const InspectionValue &value)
          {
            auto &sum = submitted[{field, gradient}];
            sum       = sum ? sum + value : value;
          }

          template <typename Field>
          void
          submit_value(const InspectionValue &value)
          {
            accumulate(Field::index, false, value);
          }

          template <typename Field>
          void
          submit_gradient(const InspectionValue &value)
          {
            accumulate(Field::index, true, value);
          }
        };

        template <>
        struct LoweringAlgebra<InspectionContext>
        {
          template <typename Number>
          static InspectionValue
          scalar(const LoweringOpcode opcode, const Number value)
          {
            std::ostringstream stream;
            stream.imbue(std::locale::classic());
            stream << std::setprecision(
                        std::numeric_limits<Number>::max_digits10)
                   << +value;
            return inspection_node(opcode, {}, 0, stream.str());
          }

          template <typename Number>
          static InspectionValue
          constant(const InspectionContext &, const Number value)
          {
            return scalar(LoweringOpcode::constant, value);
          }

          template <typename Number>
          static InspectionValue
          coefficient(const InspectionContext &, const Number value)
          {
            return scalar(LoweringOpcode::coefficient, value);
          }

          static InspectionValue
          trace(const InspectionValue &value)
          {
            return inspection_node(LoweringOpcode::trace, {value});
          }

          static InspectionValue
          symmetrize(const InspectionValue &value)
          {
            return inspection_node(LoweringOpcode::symmetrize, {value});
          }

          static InspectionValue
          identity(const InspectionValue &value)
          {
            return inspection_node(LoweringOpcode::identity, {value});
          }

          static InspectionValue
          scalar_product(const InspectionValue &left,
                         const InspectionValue &right)
          {
            return inspection_node(LoweringOpcode::scalar_product,
                                   {left, right});
          }
        };

        template <typename Form, typename... Fields>
        std::vector<LoweringField>
        inspection_fields(TypeList<Fields...>)
        {
          return {{Fields::index,
                   Fields::shape,
                   FieldRequirements<Form, Fields>::value,
                   FieldRequirements<Form, Fields>::gradient}...};
        }

        struct InspectionExporter
        {
          KernelIR &kernel;
          using Key = std::tuple<LoweringOpcode,
                                 std::vector<std::size_t>,
                                 unsigned int,
                                 std::string>;
          std::map<Key, std::size_t>                    ids;
          std::map<const InspectionNode *, std::size_t> visited;

          std::size_t
          append(const InspectionValue &value)
          {
            const auto existing = visited.find(value.get());
            if (existing != visited.end())
              return existing->second;
            std::vector<std::size_t> operands;
            for (const auto &operand : value->operands)
              operands.push_back(append(operand));
            const Key  key{value->opcode,
                          operands,
                          value->field,
                          value->literal};
            const auto inserted = ids.emplace(key, kernel.operations.size());
            if (inserted.second)
              kernel.operations.push_back({value->opcode,
                                           std::move(operands),
                                           value->field,
                                           value->literal});
            const auto id = inserted.first->second;
            visited.emplace(value.get(), id);
            return id;
          }
        };
      } // namespace internal

      /**
       * @brief Inspect the static lowering of a bilinear cell form.
       * @param form A form supported by BilinearCellKernel.
       * @return An owned, backend-independent quadrature-point operation DAG.
       * Uses the execution lowering and its field analysis with symbolic
       * values. Equal operations (including equal coefficient values) share
       * temporary IDs; coefficient variable names are not retained by the form
       * API. Unit scaling and initial zero accumulation are omitted for
       * readability. Other arithmetic, including test-adjoint symmetrization,
       * is preserved. Inspection allocates only when explicitly called and
       * requires no mesh, MatrixFree object, spatial dimension, or initialized
       * execution backend.
       */
      template <typename Form>
      KernelIR
      inspect_lowering(const Form &form)
      {
        using FormType = std::decay_t<Form>;
        KernelIR kernel;
        kernel.inputs = internal::inspection_fields<FormType>(
          typename FormFields<FormType>::trial_fields{});
        kernel.outputs = internal::inspection_fields<FormType>(
          typename FormFields<FormType>::test_fields{});
        internal::InspectionContext        context;
        const BilinearCellKernel<FormType> cell_kernel(form);
        cell_kernel(context);
        internal::InspectionExporter exporter{kernel, {}, {}};
        for (const auto &entry : context.submitted)
          kernel.submissions.push_back({entry.first.first,
                                        entry.first.second,
                                        exporter.append(entry.second)});
        return kernel;
      }
    } // namespace expression_templates
  } // namespace forms
} // namespace pmf

#endif
