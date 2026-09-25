#pragma once

#ifdef USE_BOOST_RANDOM
#include <boost/random.hpp>
namespace rnd = boost::random;
#else
#include <random>
namespace rnd = std;
#endif
