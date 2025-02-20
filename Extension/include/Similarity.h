#pragma once

#include "Common.h"

namespace ExtensionTest {

at::Tensor NormalisedCrossCorrelation(const at::Tensor &a, const at::Tensor &b);

} // namespace ExtensionTest