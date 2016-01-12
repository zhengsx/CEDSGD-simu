#pragma once

#include <thread>

#include "DataController.h"

namespace MSRAAI
{
	class CEDSGD
	{
	public:
		CEDSGD();
		~CEDSGD();

		void Init();
		void Train();
		double Test();

		void ParallelTrain();

		SGDParams * pParam;

	private:
		bool _isInit;
		double TrainOneSample(double* weight, double learningrate);
		void ModelAverage();

		double* _w;

		double** _localW;

		DataController * _pController;
	};
}