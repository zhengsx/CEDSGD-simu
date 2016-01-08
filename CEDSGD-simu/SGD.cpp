#pragma once

#include "SGD.h"

namespace MSRAAI
{
	CEDSGD::CEDSGD()
	{
		pParam = new SGDParams();
	}
	CEDSGD::~CEDSGD()
	{
		delete pParam, _pController;
	}

	void CEDSGD::Init()
	{
		_pController = new DataController(pParam);
		_w = new double[pParam->m_dimension];
		_pController->_pGenerator->GenerateUniform(pParam->m_umin, pParam->m_umax, pParam->m_dimension, _w);
	}

	void CEDSGD::Train()
	{
		fprintf(stderr, "Training Start.......\n");
		double loss = 0., losssincelast = 0.;
		int ntrainedsincelast = 0;
		for (int iter = 0; iter < pParam->m_T; ++iter)
		{
			loss = TrainOneSample(iter);
			losssincelast += loss;
			ntrainedsincelast++;

			if (ntrainedsincelast % pParam->m_show == 0 || iter == pParam->m_T - 1)
			{
				fprintf(stderr, "Train [%d - %d of %d], loss = %f \n", iter - ntrainedsincelast + 1, iter, pParam->m_T, losssincelast / ntrainedsincelast);
				Test();
				ntrainedsincelast = 0;
				losssincelast = 0.;
			}
		}
	}

	double CEDSGD::TrainOneSample(int iter)
	{
		double loss = 0.;
		double lr = pParam->m_dimension / (pParam->m_lradjust * (pParam->m_dimension + iter));
		int idx = _pController->_pGenerator->SampleOne();
		double* x = _pController->GetX(idx);
		double y = _pController->GetY(idx);
		
		loss = inner(x, _w, pParam->m_dimension) - y;

		cpax(_w, -loss * lr, x, pParam->m_dimension);

		loss *= 0.5 * loss;

		return loss;
	}

	void CEDSGD::ParallelTrain(int nbclients){}

	double CEDSGD::Test()
	{
		fprintf(stderr, "Testing.......\n");
		double loss = 0.;
		for (int i = 0; i < pParam->m_nsamples; ++i)
			loss += pow(_pController->GetY(i) - inner(_pController->GetX(i), _w, pParam->m_dimension), 2);
		loss /= 2 * pParam->m_nsamples;
		fprintf(stderr, "Test loss = %f\n", loss);
		return loss;
	}
}