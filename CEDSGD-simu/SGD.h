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
		void Test();

		SGDParams * pParam;

	private:
		double* _w;

		DataController * _pController;
	};
}