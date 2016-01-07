#pragma once

#include <cmath>

#include <iostream>

namespace MSRAAI
{
	inline double inner(const double* x, const double* y, int dim)
	{
		double res = 0;
		for (int i = 0; i < dim; ++i)
			res += x[i] * y[i];
		return res;
	};
}