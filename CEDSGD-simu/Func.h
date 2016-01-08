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
	}

	inline void axpby(const double a, const double* x, const double b, const double* y, const int n, /*out*/double* out)
	{
		for (int i = 0; i < n; ++i)
			out[i] = a*x[i] + b*y[i];
	}

	inline void cpax(/*out*/double* c, const double a, const double* x, int n)
	{
		for (int i = 0; i < n; ++i)
			c[i] += a * x[i];
	}
}