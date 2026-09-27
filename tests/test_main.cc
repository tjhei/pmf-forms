#include <catch2/catch_test_macros.hpp>
#include <forms.h>

#include <stdexcept>

using namespace pmf::forms;

TEST_CASE("Expression operators track value shapes", "[forms][shapes]")
{
  const auto scalar = trial("s", ValueShape::scalar);
  const auto vector = trial("u", ValueShape::vector);
  const auto tensor = grad(vector);
  const auto strain = sym(tensor);

  REQUIRE(grad(scalar).shape() == ValueShape::vector);
  REQUIRE(tensor.shape() == ValueShape::tensor);
  REQUIRE(strain.shape() == ValueShape::symmetric_tensor);
  REQUIRE(div(vector).shape() == ValueShape::scalar);
  REQUIRE(inner(vector, vector).shape() == ValueShape::scalar);
  REQUIRE(inner(strain, strain).shape() == ValueShape::scalar);

  REQUIRE_THROWS_AS(div(scalar), std::invalid_argument);
  REQUIRE_THROWS_AS(sym(vector), std::invalid_argument);
  REQUIRE_THROWS_AS(inner(vector, tensor), std::invalid_argument);
  REQUIRE_THROWS_AS(vector + scalar, std::invalid_argument);
  REQUIRE_THROWS_AS(integral(vector, dx), std::invalid_argument);
}

TEST_CASE("Stokes form can be constructed and printed", "[forms][stokes]")
{
  const auto u  = trial("u", ValueShape::vector);
  const auto p  = trial("p", ValueShape::scalar);
  const auto v  = test("v", ValueShape::vector);
  const auto q  = test("q", ValueShape::scalar);
  const auto mu = coefficient("mu");

  const auto form = integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))), dx) -
                    integral(div(v) * p, dx) - integral(q * div(u), dx);
  const auto text = form.str();

  INFO(text);
  REQUIRE(form.integrals().size() == 3);
  REQUIRE(text.find("trial(u)") != std::string::npos);
  REQUIRE(text.find("trial(p)") != std::string::npos);
  REQUIRE(text.find("test(v)") != std::string::npos);
  REQUIRE(text.find("test(q)") != std::string::npos);
  REQUIRE(text.find("coefficient(mu)") != std::string::npos);
  REQUIRE(text.find("sym(grad(test(v)))") != std::string::npos);
  REQUIRE(text.find("div(trial(u))") != std::string::npos);
}

TEST_CASE("Isotropic elasticity form can be constructed and printed",
          "[forms][elasticity]")
{
  const auto u      = trial("u", ValueShape::vector);
  const auto v      = test("v", ValueShape::vector);
  const auto lambda = coefficient("lambda");
  const auto mu     = coefficient("mu");

  const auto form = integral(2.0 * mu * inner(sym(grad(v)), sym(grad(u))), dx) +
                    integral(lambda * div(v) * div(u), dx);
  const auto text = form.str();

  INFO(text);
  REQUIRE(form.integrals().size() == 2);
  REQUIRE(text.find("coefficient(lambda)") != std::string::npos);
  REQUIRE(text.find("coefficient(mu)") != std::string::npos);
  REQUIRE(text.find("sym(grad(test(v)))") != std::string::npos);
  REQUIRE(text.find("sym(grad(trial(u)))") != std::string::npos);
  REQUIRE(text.find("div(test(v))") != std::string::npos);
  REQUIRE(text.find("div(trial(u))") != std::string::npos);
}
