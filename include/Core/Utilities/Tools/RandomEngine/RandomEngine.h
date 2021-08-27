/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef RANDOM_ENGINE_H
#define RANDOM_ENGINE_H
#include <random>
#include <chrono>
#include <vector>


/**
* @brief  Random Engine
* @ingroup VirtualQuantumProcessor
*/
class RandomEngine {
public:
	virtual double operator()() = 0;
	virtual inline std::vector<double> operator()(size_t n) {
		std::vector<double> ret;
		ret.reserve(n);
		for (size_t i = 0u; i < n; ++i) {
			ret.push_back((*this)());
		}
		return ret;
	}
};

/**
* @brief  Default Random Engine
* @ingroup VirtualQuantumProcessor
*/
class DefaultRandomEngine : public RandomEngine {
private:
	std::default_random_engine engine;
public:
	DefaultRandomEngine()
		:engine(std::chrono::system_clock::now().time_since_epoch().count()) 
	{ }

	DefaultRandomEngine(long long seed)
		:engine(seed) {	}

	inline double operator()() {
		return engine();
	}
};

/**
* @brief  XC Random Engine
* @ingroup VirtualQuantumProcessor
*/
class XC_RandomEngine16807 : public RandomEngine {
	int irandseed = 0;
	int ia = 16807;
	int im = 2147483647;
	int iq = 127773;
	int ir = 2836;
	int irandnewseed;
public:
	XC_RandomEngine16807() {
		irandseed = (int)std::chrono::system_clock::now().time_since_epoch().count();
	}
	XC_RandomEngine16807(long long _seed) 
		: irandseed((int)_seed) 
	{ }

	inline double operator()() {
		if (ia * (irandseed % iq) - ir * (irandseed / iq) >= 0)
		{
			irandnewseed = ia * (irandseed % iq) - ir * (irandseed / iq);
		}
		else
		{
			irandnewseed = ia * (irandseed % iq) - ir * (irandseed / iq) + im;
		}
		irandseed = irandnewseed;
		return (double)irandnewseed / im;
	}
};

inline double _default_random_generator() {
	static XC_RandomEngine16807 engine;
	return engine();
}

class RandomEngine19937
{
public:
    RandomEngine19937()
    {
        set_random_seed();
    }

    void set_random_seed()
    {
        m_mt.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }

    void set_random_seed(size_t seed)
    {
        m_mt.seed(seed);
    }

    inline double random_double(double a = 0., double b = 1.)
    {
        return std::uniform_real_distribution<double>(a, b)(m_mt);
    }

    template<typename Float = double>
    inline int random_discrete(const std::vector<Float> &probs)
    {
        return std::discrete_distribution<size_t>(probs.begin(), probs.end())(m_mt);
    }
private:
    std::mt19937_64 m_mt;
};

inline double random_generator19937(double begine = 0, double end = 1)
{
    static RandomEngine19937 rng;
    return rng.random_double(begine, end);
}

#endif // RANDOM_ENGINE_H
