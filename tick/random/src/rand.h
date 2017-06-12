#ifndef TICK_RANDOM_SRC_RAND_H_
#define TICK_RANDOM_SRC_RAND_H_

#include <mutex>
#include <cmath>
#include <ctime>
#include <iostream>
#include <random>

#include "array.h"

#if defined(_MSC_VER)
template struct DLL_PUBLIC std::uniform_int<int>::param_type;
template struct DLL_PUBLIC std::uniform_int<ulong>::param_type;
template struct DLL_PUBLIC std::uniform_real<double>::param_type;
template struct DLL_PUBLIC std::normal_distribution<double>::param_type;
template struct DLL_PUBLIC std::exponential_distribution<double>::param_type;
template struct DLL_PUBLIC std::poisson_distribution<int>::param_type;
template struct DLL_PUBLIC std::discrete_distribution<ulong>::param_type;

template class DLL_PUBLIC std::uniform_int_distribution<int>;
template class DLL_PUBLIC std::uniform_int_distribution<ulong>;
template class DLL_PUBLIC std::uniform_real_distribution<double>;
template class DLL_PUBLIC std::normal_distribution<double>;
template class DLL_PUBLIC std::exponential_distribution<double>;
template class DLL_PUBLIC std::poisson_distribution<int>;
template class DLL_PUBLIC std::discrete_distribution<ulong>;

template class DLL_PUBLIC std::mersenne_twister_engine<uint_fast64_t,
  64,312,156,31,0xb5026f5aa96619e9,
  29,0x5555555555555555,
  17,0x71d67fffeda60000,
  37,0xfff7eee000000000,
  43,6364136223846793005>;
#endif

/**
 * @class Rand
 * @brief Instance of random generators
 *
 * Each instance wraps a Mersenne Twister random number generator and generate random probability
 * distributions from it.
 */
class DLL_PUBLIC Rand {
 private:
    int seed;
    std::mt19937_64 generator;

    std::uniform_int_distribution<int> uniform_int_dist;
    std::uniform_int_distribution<ulong> uniform_ulong_dist;
    std::uniform_real_distribution<double> uniform_dist;
    std::normal_distribution<double> normal_dist;
    std::exponential_distribution<double> expon_dist;
    std::poisson_distribution<int> poisson_dist;
    std::discrete_distribution<ulong> discrete_dist;

 public :
    /**
     * @brief Constructor of Rand object
     * \param seed : seed of the Rand object, if it is negative, a random seed will be chosen
     * (depending on timestamp and other variables)
     */
    explicit Rand(int seed);

    /**
     * @brief Constructor of Rand object with random seed
     */
    Rand() : Rand(-1) {}

    /**
     * @brief Constructor of Rand object with determined seed
     */
    explicit Rand(unsigned int seed) : Rand(static_cast<int>(seed)) {}

    /**
     * @brief Constructor of Rand object with predefined generator
     */
    explicit Rand(const std::mt19937_64 &generator);

 private:
    /**
     * @brief Some distributions might be kept, we init them there
     */
    void init_reusable_distributions();

 public:
    /**
     * @brief Returns a random integer between two number (both can be reached)
     * \param a : lower bound
     * \param b : upper bound
     */
    int uniform_int(int a, int b);

    /**
     * @brief Returns a random integer between two number (both can be reached)
     * \param a : lower bound
     * \param b : upper bound
     */
    ulong uniform_int(ulong a, ulong b);

    /**
     * @brief Returns a random real between 0 and 1
     */
    double uniform();

    /**
     * @brief Returns a random real between two numbers
     * \param a : lower bound
     * \param b : upper bound
     */
    double uniform(double a, double b);

    /**
     * @brief Returns a realization of a centered gaussian
     */
    double gaussian();

    /**
     * @brief Returns a realization of a gaussian with given mean and standard deviation
     * \param mu : mean
     * \param sigma : standard deviation
     */
    double gaussian(double mean, double std);

    /**
     * @brief Returns a realization of an exponential distribution with given intensity
     * \param intensity : given intensity
     */
    double exponential(double intensity);

    /**
     * @brief Returns a realization of a poisson distribution with given rate
     * \param rate : given rate
     */
    int poisson(double rate);

    /**
     * @brief Set probabilities discrete distribution for discrete distribution
     * \param probabilities: probabilities of each event
     */
    void set_discrete_dist(ArrayDouble probabilities);

    /**
     * @brief Returns a realization of a the set discrete distribution
     */
    ulong discrete();

    /**
     * @brief Returns a realization of a new discrete distribution
     * \param probabilities: probabilities of each event
     */
    ulong discrete(ArrayDouble probabilities);

    /**
     * @brief Getter for seed variable
     * @return Seed used to construct the random generator
     */
    int get_seed() const;

    /**
    * @brief Re-seed the generator
    * \param seed A new seed for the random generator
    */
    void reseed(const int seed);
};

#endif  // TICK_RANDOM_SRC_RAND_H_
