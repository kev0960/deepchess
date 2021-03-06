#include "bit_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace chess {
namespace {

TEST(BitTest, OnBitAt) {
  uint64_t bit = 0;
  bit = OnBitAt(bit, 3);
  EXPECT_EQ(bit, 0b1000);

  bit = OnBitAt(bit, 7);
  EXPECT_EQ(bit, 0b10001000);

  bit = OnBitAt(bit, 20);
  EXPECT_EQ(bit, 0b100000000000010001000);

  // Do nothing since it is already off.
  bit = OffBitAt(bit, 19);
  EXPECT_EQ(bit, 0b100000000000010001000);

  bit = OffBitAt(bit, 20);
  EXPECT_EQ(bit, 0b10001000);

  bit = OffBitAt(bit, 3);
  EXPECT_EQ(bit, 0b10000000);

  bit = OffBitAt(bit, 7);
  EXPECT_EQ(bit, 0);
}

}  // namespace
}  // namespace chess
