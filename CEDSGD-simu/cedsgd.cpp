#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>

#include "io.h"

#include "Timer.h"
#include "SGD.h"

using namespace std;
using namespace MSRAAI;

namespace MSRAAI
{
	void RedirectStdErr(string logpath)
	{
		fprintf(stderr, "Redirecting stderr to file %s\n", logpath.c_str());
		FILE* f = fopen(logpath.c_str(), "w");
		if (_dup2(_fileno(f), 2) == -1)
		{
			fprintf(stderr, "unexpected failure to redirect stderr to log file.\n");
			abort();
		}
		setvbuf(stderr, NULL, _IONBF, 16384);

		static auto fKept = f;
	}

	string WCharToString(const wchar_t* wst)
	{
		std::wstring ws(wst);
		std::string s(ws.begin(), ws.end());
		s.assign(ws.begin(), ws.end());
		return s;
	}

	string parsecommandline(int argc, char* argv[])
	{
		if (argc < 2)
			return "";
		string configfilepath = argv[1];
		return configfilepath;
	}

	void parseconfig(const string & configfilepath, SGDParams* param)
	{
		ifstream configfilestream(configfilepath);
		if (!configfilestream)
			fprintf(stderr, "Config File Read Failed.\n");
		string strline, strcontext;
		bool isRedirectStderr = false;
		while (getline(configfilestream, strline))
		{
			if (strline.compare(0, 1, "#") == 0)
			{
				continue;
			}
			else if (strline.compare(0, 7, "stderr=") == 0)
			{
				assert(!isRedirectStderr);
				strcontext = strline.substr(7, strline.length() - 7);
				RedirectStdErr(strcontext);
				isRedirectStderr = !isRedirectStderr;
			}
			else if (strline.compare(0, 10, "nparallel=") == 0)
			{
				strcontext = strline.substr(10, strline.length() - 10);
				param->m_nclients = stoi(strcontext);
			}
			else if (strline.compare(0, 2, "d=") == 0)
			{
				strcontext = strline.substr(2, strline.length() - 2);
				param->m_dimension = stoi(strcontext);
			}
			else if (strline.compare(0, 3, "lr=") == 0)
			{
				strcontext = strline.substr(3, strline.length() - 3);
				param->m_learningrate = stod(strcontext);
			}
			else if (strline.compare(0, 2, "e=") == 0)
			{
				strcontext = strline.substr(2, strline.length() - 2);
				param->m_e = stod(strcontext);
			}
			else if (strline.compare(0, 2, "n=") == 0)
			{
				strcontext = strline.substr(2, strline.length() - 2);
				param->m_nsamples = stoi(strcontext);
			}
			else if (strline.compare(0, 2, "T=") == 0)
			{
				strcontext = strline.substr(2, strline.length() - 2);
				param->m_T = stoi(strcontext);
			}
			else if (strline.compare(0, 2, "m=") == 0)
			{
				strcontext = strline.substr(2, strline.length() - 2);
				param->m_m = stoi(strcontext);
			}
			else if (strline.compare(0, 5, "show=") == 0)
			{
				strcontext = strline.substr(5, strline.length() - 5);
				param->m_show = stoi(strcontext);
			}
			else
			{
				continue;
			}
		}
		configfilestream.close();
	}
}

int main(int argc, char** argv)
{
	CEDSGD* sim = new CEDSGD();
	string configpath = parsecommandline(argc, argv);
	if (configpath != "")
		parseconfig(configpath, sim->pParam);

	sim->Init();
	if (sim->pParam->m_nclients == 1)
		sim->Train();
	else if (sim->pParam->m_nclients > 1)
		sim->ParallelTrain();
	fprintf(stderr, "Test loss = %f\n", sim->Test());
	fclose(stderr);
	return 0;
}