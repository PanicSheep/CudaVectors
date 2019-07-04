#pragma once

#define __host__
#define __device__

#include "gtest/gtest.h"
#include "Chronosity.h"
#include "DeviceVector.cuh"
#include "HostVector.cuh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
