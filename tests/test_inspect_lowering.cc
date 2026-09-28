#include <catch2/catch_test_macros.hpp>
#include <forms.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <tuple>

using namespace pmf::forms;
using namespace pmf::forms::expression_templates;

namespace
{
  std::size_t
  count(const KernelIR &kernel, const LoweringOpcode opcode)
  {
    return std::count_if(kernel.operations.begin(),
                         kernel.operations.end(),
                         [opcode](const auto &operation) {
                           return operation.opcode == opcode;
                         });
  }

  void
  check_read(const KernelIR      &kernel,
             const LoweringOpcode opcode,
             const unsigned int   field)
  {
    CHECK(std::count_if(kernel.operations.begin(),
                        kernel.operations.end(),
                        [opcode, field](const auto &operation) {
                          return operation.opcode == opcode &&
                                 operation.field == field;
                        }) == 1);
  }

  void
  check_submission(const KernelIR    &kernel,
                   const unsigned int field,
                   const bool         gradient)
  {
    CHECK(std::count_if(kernel.submissions.begin(),
                        kernel.submissions.end(),
                        [field, gradient](const auto &submission) {
                          return submission.field == field &&
                                 submission.gradient == gradient;
                        }) == 1);
  }

  auto
  requirements(const std::vector<LoweringField> &fields)
  {
    std::vector<std::tuple<unsigned int, ValueShape, bool, bool>> result;
    for (const auto &field : fields)
      result.emplace_back(field.index,
                          field.shape,
                          field.value,
                          field.gradient);
    return result;
  }

  auto
  essential_program(const KernelIR &kernel)
  {
    std::vector<std::string> expressions;
    for (std::size_t index = 0; index < kernel.operations.size(); ++index)
      {
        const auto              &operation = kernel.operations[index];
        std::vector<std::string> operands;
        for (const auto operand : operation.operands)
          {
            REQUIRE(operand < index);
            operands.push_back(expressions[operand]);
          }
        if (operation.opcode == LoweringOpcode::add ||
            operation.opcode == LoweringOpcode::multiply ||
            operation.opcode == LoweringOpcode::scalar_product)
          std::sort(operands.begin(), operands.end());
        std::string expression =
          std::to_string(static_cast<int>(operation.opcode)) + "[" +
          std::to_string(operation.field) + ":" + operation.literal;
        for (const auto &operand : operands)
          expression += ";" + operand;
        expressions.push_back(expression + "]");
      }
    std::map<std::pair<unsigned int, bool>, std::string> result;
    for (const auto &submission : kernel.submissions)
      {
        REQUIRE(submission.operand < expressions.size());
        REQUIRE(
          result
            .emplace(std::make_pair(submission.field, submission.gradient),
                     expressions[submission.operand])
            .second);
      }
    return result;
  }

  void
  check_equivalent(const KernelIR &left, const KernelIR &right)
  {
    CHECK(requirements(left.inputs) == requirements(right.inputs));
    CHECK(requirements(left.outputs) == requirements(right.outputs));
    CHECK(essential_program(left) == essential_program(right));
  }
} // namespace

TEST_CASE("Laplace inspection reads and submits one gradient",
          "[forms][inspection]")
{
  const auto trial_field = trial();
  const auto test_field  = test();
  const auto form   = integral(inner(grad(test_field), grad(trial_field)), dx);
  const auto kernel = inspect_lowering(form);
  REQUIRE(kernel.inputs.size() == 1);
  CHECK(kernel.inputs[0].index == 0);
  CHECK(kernel.inputs[0].shape == ValueShape::scalar);
  CHECK_FALSE(kernel.inputs[0].value);
  CHECK(kernel.inputs[0].gradient);
  CHECK(requirements(kernel.inputs) == requirements(kernel.outputs));
  check_read(kernel, LoweringOpcode::get_gradient, 0);
  CHECK(count(kernel, LoweringOpcode::get_value) == 0);
  REQUIRE(kernel.submissions.size() == 1);
  check_submission(kernel, 0, true);
  const auto &submitted = kernel.operations.at(kernel.submissions[0].operand);
  CHECK(submitted.opcode == LoweringOpcode::get_gradient);
  CHECK(submitted.field == 0);
  check_equivalent(kernel,
                   inspect_lowering(
                     integral(inner(grad(trial_field), grad(test_field)), dx)));
  std::ostringstream stream;
  stream << kernel;
  CHECK(stream.str().find("get_gradient(field 0)") != std::string::npos);
  CHECK(stream.str().find("submit_gradient(field 0,") != std::string::npos);
}

TEST_CASE(
  "Helmholtz inspection keeps values, gradients, and distinct coefficients",
  "[forms][inspection]")
{
  const auto trial_field = trial();
  const auto test_field  = test();
  const auto diffusion   = coefficient(2.5);
  const auto reaction    = coefficient(0.75);
  const auto kernel      = inspect_lowering(
    integral(diffusion * inner(grad(test_field), grad(trial_field)) +
               reaction * (test_field * trial_field),
             dx));
  REQUIRE(kernel.inputs.size() == 1);
  CHECK(kernel.inputs[0].value);
  CHECK(kernel.inputs[0].gradient);
  CHECK(requirements(kernel.inputs) == requirements(kernel.outputs));
  check_read(kernel, LoweringOpcode::get_value, 0);
  check_read(kernel, LoweringOpcode::get_gradient, 0);
  REQUIRE(kernel.submissions.size() == 2);
  check_submission(kernel, 0, false);
  check_submission(kernel, 0, true);
  CHECK(count(kernel, LoweringOpcode::coefficient) == 2);
  check_equivalent(
    kernel,
    inspect_lowering(
      integral(reaction * (trial_field * test_field), dx) +
      integral(diffusion * inner(grad(trial_field), grad(test_field)), dx)));
}

TEST_CASE("Elasticity inspection reuses gradients and accumulates stress",
          "[forms][inspection]")
{
  const auto velocity      = trial<ValueShape::vector>();
  const auto test_velocity = test<ValueShape::vector>();
  const auto mu            = coefficient(1.5);
  const auto lambda        = coefficient(3.25);
  const auto kernel        = inspect_lowering(
    integral(2 * mu * inner(sym(grad(test_velocity)), sym(grad(velocity))) +
               lambda * div(test_velocity) * div(velocity),
             dx));
  REQUIRE(kernel.inputs.size() == 1);
  CHECK(kernel.inputs[0].shape == ValueShape::vector);
  CHECK_FALSE(kernel.inputs[0].value);
  CHECK(kernel.inputs[0].gradient);
  CHECK(requirements(kernel.inputs) == requirements(kernel.outputs));
  check_read(kernel, LoweringOpcode::get_gradient, 0);
  CHECK(count(kernel, LoweringOpcode::get_value) == 0);
  CHECK(count(kernel, LoweringOpcode::trace) == 1);
  CHECK(count(kernel, LoweringOpcode::identity) == 1);
  CHECK(count(kernel, LoweringOpcode::symmetrize) == 2);
  REQUIRE(kernel.submissions.size() == 1);
  check_submission(kernel, 0, true);
  CHECK(kernel.operations.at(kernel.submissions[0].operand).opcode ==
        LoweringOpcode::add);
  check_equivalent(
    kernel,
    inspect_lowering(
      integral(lambda * div(velocity) * div(test_velocity) +
                 2 * mu * inner(sym(grad(velocity)), sym(grad(test_velocity))),
               dx)));
}

TEST_CASE("Stokes inspection reads fields once and submits both blocks",
          "[forms][inspection]")
{
  const auto [velocity, pressure] =
    trial_functions<ValueShape::vector, ValueShape::scalar>();
  const auto [test_velocity, test_pressure] =
    test_functions<ValueShape::vector, ValueShape::scalar>();
  const auto mu     = coefficient(1.5);
  const auto kernel = inspect_lowering(
    integral(2 * mu * inner(sym(grad(test_velocity)), sym(grad(velocity))),
             dx) -
    integral(div(test_velocity) * pressure, dx) -
    integral(test_pressure * div(velocity), dx));
  REQUIRE(kernel.inputs.size() == 2);
  CHECK(kernel.inputs[0].index == 0);
  CHECK_FALSE(kernel.inputs[0].value);
  CHECK(kernel.inputs[0].gradient);
  CHECK(kernel.inputs[1].index == 1);
  CHECK(kernel.inputs[1].value);
  CHECK_FALSE(kernel.inputs[1].gradient);
  CHECK(requirements(kernel.inputs) == requirements(kernel.outputs));
  check_read(kernel, LoweringOpcode::get_gradient, 0);
  check_read(kernel, LoweringOpcode::get_value, 1);
  CHECK(count(kernel, LoweringOpcode::get_gradient) == 1);
  CHECK(count(kernel, LoweringOpcode::get_value) == 1);
  REQUIRE(kernel.submissions.size() == 2);
  check_submission(kernel, 0, true);
  check_submission(kernel, 1, false);
  const auto &pressure_submission =
    kernel.operations.at(kernel.submissions[1].operand);
  REQUIRE(pressure_submission.opcode == LoweringOpcode::negate);
  const auto &divergence =
    kernel.operations.at(pressure_submission.operands.at(0));
  REQUIRE(divergence.opcode == LoweringOpcode::trace);
  CHECK(kernel.operations.at(divergence.operands.at(0)).opcode ==
        LoweringOpcode::get_gradient);
  check_equivalent(
    kernel,
    inspect_lowering(
      integral(2 * mu * inner(sym(grad(velocity)), sym(grad(test_velocity))) -
                 test_pressure * div(velocity) - pressure * div(test_velocity),
               dx)));
}

TEST_CASE(
  "Inspection shares repeated expressions without merging distinct fields",
  "[forms][inspection]")
{
  const auto [first, second] =
    trial_functions<ValueShape::scalar, ValueShape::scalar>();
  const auto [test_first, test_second] =
    test_functions<ValueShape::scalar, ValueShape::scalar>();
  const auto flux =
    coefficient(2.0) * grad(first) + coefficient(3.0) * grad(second);
  const auto kernel = inspect_lowering(
    integral(inner(grad(test_first), flux) + inner(grad(test_second), flux),
             dx));
  check_read(kernel, LoweringOpcode::get_gradient, 0);
  check_read(kernel, LoweringOpcode::get_gradient, 1);
  CHECK(count(kernel, LoweringOpcode::add) == 1);
  REQUIRE(kernel.submissions.size() == 2);
  CHECK(kernel.submissions[0].operand == kernel.submissions[1].operand);
  CHECK(kernel.submissions[0].field != kernel.submissions[1].field);
  CHECK(count(kernel, LoweringOpcode::coefficient) == 2);
  essential_program(kernel);
}

TEST_CASE("Stokes benchmark inspection has no symmetric-gradient operations",
          "[forms][inspection]")
{
  const auto [velocity, pressure] =
    trial_functions<ValueShape::vector, ValueShape::scalar>();
  const auto [test_velocity, test_pressure] =
    test_functions<ValueShape::vector, ValueShape::scalar>();
  const auto kernel =
    inspect_lowering(integral(inner(grad(test_velocity), grad(velocity)), dx) -
                     integral(div(test_velocity) * pressure, dx) -
                     integral(test_pressure * div(velocity), dx));
  check_read(kernel, LoweringOpcode::get_gradient, 0);
  check_read(kernel, LoweringOpcode::get_value, 1);
  check_submission(kernel, 0, true);
  check_submission(kernel, 1, false);
  CHECK(count(kernel, LoweringOpcode::symmetrize) == 0);
  CHECK(count(kernel, LoweringOpcode::trace) == 1);
  CHECK(count(kernel, LoweringOpcode::identity) == 1);
  REQUIRE(kernel.submissions.size() == 2);
  const auto &stress = kernel.operations.at(kernel.submissions[0].operand);
  REQUIRE(stress.opcode == LoweringOpcode::add);
  CHECK(kernel.operations.at(stress.operands.at(0)).opcode ==
        LoweringOpcode::get_gradient);
  const auto &pressure_tensor = kernel.operations.at(stress.operands.at(1));
  REQUIRE(pressure_tensor.opcode == LoweringOpcode::identity);
  const auto &negative_pressure =
    kernel.operations.at(pressure_tensor.operands.at(0));
  REQUIRE(negative_pressure.opcode == LoweringOpcode::negate);
  const auto &pressure_read =
    kernel.operations.at(negative_pressure.operands.at(0));
  CHECK(pressure_read.opcode == LoweringOpcode::get_value);
  CHECK(pressure_read.field == 1);
  essential_program(kernel);
}

TEST_CASE("Inspection distinguishes trial requirements from test requirements",
          "[forms][inspection]")
{
  const auto [velocity, pressure] =
    trial_functions<ValueShape::vector, ValueShape::scalar>();
  const auto [test_velocity, test_pressure] =
    test_functions<ValueShape::vector, ValueShape::scalar>();
  const auto kernel = inspect_lowering(integral(
    inner(test_velocity, grad(pressure)) + test_pressure * div(velocity), dx));
  REQUIRE(kernel.inputs.size() == 2);
  REQUIRE(kernel.outputs.size() == 2);
  for (const auto &field : kernel.inputs)
    {
      CHECK_FALSE(field.value);
      CHECK(field.gradient);
    }
  for (const auto &field : kernel.outputs)
    {
      CHECK(field.value);
      CHECK_FALSE(field.gradient);
    }
  check_submission(kernel, 0, false);
  check_submission(kernel, 1, false);
  essential_program(kernel);
}
