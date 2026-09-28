#ifndef PMF_FORM_LOWERING_IR_H
#define PMF_FORM_LOWERING_IR_H

#include <form_types.h>

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace pmf
{
  namespace forms
  {
    namespace expression_templates
    {
      /** @brief Backend-independent operations at one quadrature point. */
      enum class LoweringOpcode
      {
        constant,       ///< A scalar literal.
        coefficient,    ///< An owned scalar coefficient value.
        get_value,      ///< Read a trial field value.
        get_gradient,   ///< Read a trial field gradient.
        add,            ///< Add two values.
        subtract,       ///< Subtract two values.
        multiply,       ///< Multiply, with at least one scalar operand.
        negate,         ///< Negate a value.
        scalar_product, ///< Contract two vectors or tensors.
        symmetrize,     ///< Compute (tensor + transpose(tensor)) / 2.
        trace,          ///< Take a tensor trace.
        identity,       ///< Multiply a scalar by the identity tensor.
        zero_tensor,    ///< A zero rank-2 gradient accumulator.
        add_diagonal ///< Add operand 1 (scalar) to the diagonal of operand 0.
      };

      /** @brief Value and gradient requirements for one field. */
      struct LoweringField
      {
        unsigned int index;    ///< Zero-based field index.
        ValueShape   shape;    ///< Shape of the field value.
        bool         value;    ///< Whether values are needed.
        bool         gradient; ///< Whether gradients are needed.
      };

      /**
       * @brief A pure operation producing one temporary.
       * Its position in KernelIR::operations is its temporary ID. Operands
       * reference earlier IDs. Equal expressions share an ID for inspection;
       * this does not imply common-subexpression elimination during execution.
       * An add_diagonal result represents the accumulator after an in-place
       * diagonal update; it does not imply a tensor copy during execution.
       */
      struct LoweringOperation
      {
        LoweringOpcode           opcode;    ///< Operation kind.
        std::vector<std::size_t> operands;  ///< Input temporary IDs.
        unsigned int             field = 0; ///< Field index for reads only.
        std::string              literal; ///< Value for literals/coefficients.
      };

      /** @brief One final submission of accumulated test contributions. */
      struct LoweringSubmission
      {
        unsigned int field;    ///< Zero-based test field index.
        bool         gradient; ///< True for submit_gradient, false for value.
        std::size_t  operand;  ///< Temporary containing the accumulated value.
      };

      /**
       * @brief Inspectable, non-executable description of a cell kernel.
       * Describes one quadrature point, without gather/scatter, geometry,
       * constraints, or backend scheduling. Field lists are sorted by index.
       */
      struct KernelIR
      {
        std::vector<LoweringField>     inputs;     ///< Trial evaluation needs.
        std::vector<LoweringField>     outputs;    ///< Test integration needs.
        std::vector<LoweringOperation> operations; ///< Topological value DAG.
        std::vector<LoweringSubmission>
          submissions; ///< Final test submissions.
      };

      /**
       * @brief Print field requirements and a readable quadrature kernel.
       * @param stream Destination stream.
       * @param kernel Inspected lowering to print.
       * @return The destination stream.
       */
      inline std::ostream &
      operator<<(std::ostream &stream, const KernelIR &kernel)
      {
        const auto fields = [&stream](const char *title, const auto &entries) {
          stream << title << ":\n";
          for (const auto &field : entries)
            {
              stream << "  field " << field.index << ": ";
              if (field.value)
                stream << "value";
              if (field.value && field.gradient)
                stream << ", ";
              if (field.gradient)
                stream << "gradient";
              stream << '\n';
            }
        };
        fields("Inputs", kernel.inputs);
        stream << '\n';
        fields("Outputs", kernel.outputs);
        stream << "\nKernel:\n";
        for (std::size_t index = 0; index < kernel.operations.size(); ++index)
          {
            const auto &operation = kernel.operations[index];
            stream << "  %" << index << " = ";
            switch (operation.opcode)
              {
                case LoweringOpcode::constant:
                  stream << operation.literal;
                  break;
                case LoweringOpcode::coefficient:
                  stream << "coefficient(" << operation.literal << ')';
                  break;
                case LoweringOpcode::get_value:
                case LoweringOpcode::get_gradient:
                  stream << (operation.opcode == LoweringOpcode::get_value ?
                               "get_value" :
                               "get_gradient")
                         << "(field " << operation.field << ')';
                  break;
                case LoweringOpcode::add:
                case LoweringOpcode::subtract:
                case LoweringOpcode::multiply:
                  stream << '%' << operation.operands[0]
                         << (operation.opcode == LoweringOpcode::add ? " + " :
                             operation.opcode == LoweringOpcode::subtract ?
                                                                       " - " :
                                                                       " * ")
                         << '%' << operation.operands[1];
                  break;
                case LoweringOpcode::negate:
                  stream << "-%" << operation.operands[0];
                  break;
                case LoweringOpcode::identity:
                  stream << '%' << operation.operands[0] << " * identity";
                  break;
                case LoweringOpcode::zero_tensor:
                  stream << "zero_tensor";
                  break;
                case LoweringOpcode::add_diagonal:
                  stream << "add_diagonal(%" << operation.operands[0] << ", %"
                         << operation.operands[1] << ')';
                  break;
                case LoweringOpcode::scalar_product:
                  stream << "scalar_product(%" << operation.operands[0] << ", %"
                         << operation.operands[1] << ')';
                  break;
                case LoweringOpcode::symmetrize:
                case LoweringOpcode::trace:
                  stream << (operation.opcode == LoweringOpcode::symmetrize ?
                               "symmetrize" :
                               "trace")
                         << "(%" << operation.operands[0] << ')';
                  break;
              }
            stream << '\n';
          }
        for (const auto &submission : kernel.submissions)
          stream << "  "
                 << (submission.gradient ? "submit_gradient" : "submit_value")
                 << "(field " << submission.field << ", %" << submission.operand
                 << ")\n";
        return stream;
      }
    } // namespace expression_templates
  } // namespace forms
} // namespace pmf

#endif
