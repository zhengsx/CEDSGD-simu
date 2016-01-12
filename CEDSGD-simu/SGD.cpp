#pragma once

#include "SGD.h"
#include <algorithm>
#include <functional>

namespace MSRAAI
{
	CEDSGD::CEDSGD()
	{
		pParam = new SGDParams();
		_isInit = false;
	}
	CEDSGD::~CEDSGD()
	{
		delete pParam, _pController;
		if (_isInit)
		{
			for (int i = 0; i < pParam->m_nclients; i++)
			{
				delete _localW[i];
			}
			delete _localW, _w;
		}
	}

	void CEDSGD::Init()
	{
		if (!_isInit)
		{
			_pController = new DataController(pParam);
			_w = new double[pParam->m_dimension];
			_pController->_pGenerator->GenerateUniform(pParam->m_umin, pParam->m_umax, pParam->m_dimension, _w);
			if (pParam->m_nclients > 1)
			{
				_localW = new double*[pParam->m_nclients];
				for (int i = 0; i < pParam->m_nclients; i++)
				{
					_localW[i] = new double[pParam->m_dimension];
					memcpy(_localW[i], _w, sizeof(double) * pParam->m_dimension);
				}
			}
		}
		_isInit = true;
	}

	void CEDSGD::Train()
	{
		double loss = 0., losssincelast = 0.;
		int ntrainedsincelast = 0;
		for (int iter = 0; iter < pParam->m_T; ++iter)
		{
			double lr = pParam->m_dimension / (pParam->m_e * (pParam->m_dimension + iter));
			loss = TrainOneSample(_w, lr);
			losssincelast += loss;
			ntrainedsincelast++;

			if (ntrainedsincelast % pParam->m_show == 0 || iter == pParam->m_T - 1)
			{
				fprintf(stderr, "Train [%d - %d of %d], loss = %f \n", iter - ntrainedsincelast + 1, iter, pParam->m_T, losssincelast / ntrainedsincelast);
				fprintf(stderr, "Test loss = %f\n", Test());
				ntrainedsincelast = 0;
				losssincelast = 0.;
			}
		}
		fprintf(stderr, "Train Completed.\n");
	}

	double CEDSGD::TrainOneSample(double* weight, double learningrate)
	{
		double loss = 0.;
		//double lr = pParam->m_dimension / (pParam->m_lradjust * (pParam->m_dimension + iter));
		int idx = _pController->_pGenerator->SampleOne();
		double* x = _pController->GetX(idx);
		double y = _pController->GetY(idx);

		loss = inner(x, weight, pParam->m_dimension) - y;

		cpax(weight, -loss * learningrate, x, pParam->m_dimension);

		loss *= 0.5 * loss;

		return loss;
	}

	double CEDSGD::Test()
	{
		double loss = 0.;
		for (int i = 0; i < pParam->m_nsamples; ++i)
			loss += pow(_pController->GetY(i) - inner(_pController->GetX(i), _w, pParam->m_dimension), 2);
		loss /= 2 * pParam->m_nsamples;
		return loss;
	}

	void CEDSGD::ParallelTrain()
	{
		double loss = 0., losssincelast = 0.;
		int ntrainedsincelastaverage = 0, ntrainedsincelastshow = 0;
		for (int iter = 0; iter < pParam->m_T; ++iter)
		{
			double lr = pParam->m_dimension / (pParam->m_e * (pParam->m_dimension + iter));

			for (int client = 0; client < pParam->m_nclients; ++client)
			{
				loss += TrainOneSample(_localW[client], lr);
			}
			losssincelast += loss / pParam->m_nclients;
			loss = 0;
			ntrainedsincelastaverage++;
			ntrainedsincelastshow++;

			if (ntrainedsincelastaverage % pParam->m_m == 0)
			{
				ModelAverage();
				ntrainedsincelastaverage = 0;
			}

			if (ntrainedsincelastshow % pParam->m_show == 0 || iter == pParam->m_T - 1)
			{
				fprintf(stderr, "Train [%d - %d of %d], loss = %f \n", iter - ntrainedsincelastshow + 1, iter, pParam->m_T, losssincelast / ntrainedsincelastshow);
				fprintf(stderr, "Test loss = %f\n", Test());
				ntrainedsincelastshow = 0;
				losssincelast = 0.;
			}
		}
		fprintf(stderr, "Parallel Train Completed.\n");
	}

	void CEDSGD::ModelAverage()
	{
		memset(_w, 0, sizeof(double) * pParam->m_dimension);
		for (int d = 0; d < pParam->m_dimension; ++d)
		{
			for (int c = 0; c < pParam->m_nclients; ++c)
				_w[d] += _localW[c][d];
			_w[d] /= pParam->m_nclients;
			for (int c = 0; c < pParam->m_nclients; ++c)
				_localW[c][d] = _w[d];
		}
	}
}