#include<limits.h>
#include<cstring>

#include "GridDevice.h"

USING_QPANDA

BasicGridDevice::BasicGridDevice(int m, int n)
    : m(m), n(n)
{
    assert(m > 0);
    assert(n > 0);
    qubits = new PhysicalQubit[m * n];
    clear();
}

void BasicGridDevice::clear()
{
    for(int i = 0; i < m * n; i++)
	{
        qubits[i].map_to = -1;
    }
    current_time = 0;
    qubit_max_time = 0;
    resetTime();
}

BasicGridDevice::~BasicGridDevice()
{
    delete[] qubits;
}


ExGridDevice::ExGridDevice(int m, int n, std::vector<std::pair<int, int>> &lines)
	: BasicGridDevice(m, n)
{
	is_simple_grid_device = false;
	qcount = m * n;
	dist_mat = new int[qcount * qcount];
	for (int i = 0; i < qcount * qcount; i++)
	{
		dist_mat[i] = INT_MAX >> 2;
	}
	for (auto line : lines)
	{
		int i1 = line.first / n;
		int j1 = line.first % n;
		int i2 = line.second / n;
		int j2 = line.second % n;
		getQubit(i1, j1).nearbyQubits.emplace_back(i2, j2);
		dist_mat[line.first * qcount + line.second] = 1;
		dist_mat[line.second * qcount + line.first] = 1;
	}
	// floyd algorithm
	for (int k = 0; k < qcount; k++)
	{
		for (int j = 0; j < qcount; j++)
		{
			for (int i = 0; i < qcount; i++)
			{
				int dist = dist_mat[i * qcount + k] + dist_mat[k * qcount + j];
				int &dist0 = dist_mat[i * qcount + j];
				if (dist0 > dist)
				{
					dist0 = dist;
				}
			}
		}
	}
}


ExGridDevice::~ExGridDevice() 
{
	delete[] dist_mat;
}

bool ExGridDevice::isNearBy(int i1, int j1, int i2, int j2) 
{
	int q1 = i1 * n + j1;
	int q2 = i2 * n + j2;
	return dist_mat[q1 * qcount + q2] == 1;
}

int ExGridDevice::getDistance(int i1, int j1, int i2, int j2)
{
	int q1 = i1 * n + j1;
	int q2 = i2 * n + j2;
	return dist_mat[q1 * qcount + q2];
}



SimpleGridDevice::~SimpleGridDevice()
{}

bool SimpleGridDevice::isNearBy(int i1, int j1, int i2, int j2) 
{
	switch (i1 - i2) 
	{
	case -1:
	case 1:
		return j1 == j2;
	case 0:
		return j1 == j2 - 1 || j1 == j2 + 1;
	default:
		return false;
	}

    return true;
}

int SimpleGridDevice::getDistance(int i1, int j1, int i2, int j2) 
{
	return std::abs(i1 - i2) + std::abs(j1 - j2);
}



void UncompletedGridDevice::resetAvailableQubits(const bool *available_qubits) 
{
	memcpy(this->available_qubits, available_qubits, m * n);
	for (int i = 0; i < m; i++) 
	{
		for (int j = 0; j < n; j++) 
		{
			getQubit(i, j).nearbyQubits.clear();
			if (isQubitAvailable(i, j))
			{
				if (j > 0 && isQubitAvailable(i, j - 1))
					getQubit(i, j).nearbyQubits.emplace_back(i, j - 1);
				if (j < n - 1 && isQubitAvailable(i, j + 1))
					getQubit(i, j).nearbyQubits.emplace_back(i, j + 1);
				if (i > 0 && isQubitAvailable(i - 1, j))
					getQubit(i, j).nearbyQubits.emplace_back(i - 1, j);
				if (i < m - 1 && isQubitAvailable(i + 1, j))
					getQubit(i, j).nearbyQubits.emplace_back(i + 1, j);
			}
		}
	}
}

UncompletedGridDevice::~UncompletedGridDevice()
{
	delete[] available_qubits;
}

