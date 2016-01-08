#pragma once

#include <random>
#include <cassert>

#include "Func.h"

using namespace std;

namespace MSRAAI
{
	class SGDParams
	{
	public:
		SGDParams()
		{
			m_dimension = 300;
			m_learningrate = 0.1;
			m_lradjust = 100.;
			m_nsamples = 10000;

			m_T = 2000;
			m_m = 10;

			m_umin = 0.;
			m_umax = 1.;

			m_xmean = 0.;
			m_xstd = 1.;

			m_emean = 0.;
			m_estd = 1.;

			m_show = 100;
		};
		~SGDParams(){}

		int m_dimension;
		double m_learningrate;
		double m_lradjust;
		int m_nsamples;

		int m_T;
		int m_m;

		double m_umin, m_umax;
		double m_xmean, m_xstd;
		double m_emean, m_estd;

		int m_show;
	};

	class DataGenerator
	{
	public:
		DataGenerator(int n)
		{
			_sampler = new uniform_int_distribution<int>(0, n - 1);
		};
		~DataGenerator()
		{
			delete _sampler;
		};

		int SampleOne(){ return (*_sampler)(_generator); };
		void GenerateUniform(double min, double max, int dim, /*out*/double *out)
		{
			uniform_real_distribution<double> ud(min, max);
			for (int i = 0; i < dim; ++i)
				out[i] = ud(_generator);
		}
		void GenerateNormal(double mean, double std, int dim, /*out*/double *out)
		{
			normal_distribution<double> nd(mean, std);
			for (int i = 0; i < dim; ++i)
				out[i] = nd(_generator);
		}
	private:
		default_random_engine _generator;
		uniform_int_distribution<int>* _sampler;
	};

	class DataController
	{
	public:
		DataController(SGDParams* param)
		{
			_n = param->m_nsamples;
			_dim = param->m_dimension;

			_pGenerator = new DataGenerator(_n);

			_x = new double*[_n];
			for (int i = 0; i < _n; ++i)
			{
				_x[i] = new double[_dim];
				_pGenerator->GenerateNormal(param->m_xmean, param->m_xstd, _dim, _x[i]);
			}

			_e = new double[_n];
			_pGenerator->GenerateNormal(param->m_emean, param->m_estd, _n, _e);

			_u = new double[_dim];
			_pGenerator->GenerateUniform(param->m_umin, param->m_umax, _dim, _u);

			_y = new double[_n];
			for (int i = 0; i < _n; ++i)
				_y[i] = inner(_u, _x[i], _dim) + _e[i];
		};

		~DataController()
		{
			for (int i = 0; i < _n; i++)
				delete _x[i];
			delete _x, _y, _e, _u;
			delete _pGenerator;
		};

		double* GetX(int idx){ return _x[idx]; };
		double GetY(int idx){ return _y[idx]; };
		double* GetU(){ return _u; };

		DataGenerator* _pGenerator;

	private:
		double** _x;
		double* _y;
		double* _e;
		int _dim;
		int _n;

		double* _u;
			
	};
}