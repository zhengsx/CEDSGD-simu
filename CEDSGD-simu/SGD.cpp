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

		_pController->_pGenerator->GenerateUniform(pParam->m_umin, pParam->m_umax, pParam->m_dimension, _w);
	}

	void CEDSGD::Train()
	{
		
	};
	void CEDSGD::ParallelTrain(int nbclients){};
	void CEDSGD::Test(){};
}