#include "newtype/util/Rand.h"

namespace newtype::util {
	std::mt19937 Rand::sBase(310u);
	std::uniform_real_distribution<float> Rand::sFloatGen;
} // newtype::util