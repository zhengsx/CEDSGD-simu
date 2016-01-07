#include "Timer.h"
#include <assert.h>
#include <Windows.h>
static LARGE_INTEGER s_ticksPerSecond;
static BOOL s_setFreq = QueryPerformanceFrequency(&s_ticksPerSecond);

namespace MSRAAI
{
	long long Timer::GetStamp()
	{
		LARGE_INTEGER li;
		QueryPerformanceCounter(&li);
		return li.QuadPart;
	}

	void Timer::Start()
	{
		m_start = GetStamp();
	}

	void Timer::Restart()
	{
		m_start = m_end = 0;
		Start();
	}

	void Timer::Stop()
	{
		m_end = GetStamp();
	}

	long long Timer::ElapsedMicroseconds()
	{
		assert(m_start != 0);
		long long diff = 0;

		if (m_end != 0)
		{
			diff = m_end - m_start;
		}
		else
		{
			diff = GetStamp() - m_start;
		}

		if (diff < 0)
		{
			diff = 0;
		}

		assert(s_setFreq == TRUE);
		return (diff * MICRO_PER_SEC) / s_ticksPerSecond.QuadPart;
	}
}