// Copyright 2025-2026 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#ifdef USE_BOOST_RANDOM
#include <boost/random.hpp>
namespace rnd = boost::random;
#else
#include <random>
namespace rnd = std;
#endif
