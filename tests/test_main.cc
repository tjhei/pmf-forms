#include <catch2/catch_test_macros.hpp>

TEST_CASE("Catch2 test harness is available", "[infrastructure]")
{
  REQUIRE(1 + 1 == 2);
}
