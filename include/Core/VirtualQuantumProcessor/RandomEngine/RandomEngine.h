/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

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

#endif RANDOM_ENGINE