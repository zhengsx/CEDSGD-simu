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
		void ParallelTrain(int nbclients = 4);
		double Test();

		SGDParams * pParam;

	private:
		double TrainOneSample(int iter);

		double* _w;

		DataController * _pController;
	};
}