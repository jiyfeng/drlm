#ifndef TEST_HPP
#define TEST_HPP

#include "output.hpp"
#include "hidden.hpp"
#include "baseline.hpp"
#include "util.hpp"

int test(char* ftst, char* prefix, string flag = "output",
	 unsigned nlatvar = 4, unsigned nlayers = 1);

#endif
