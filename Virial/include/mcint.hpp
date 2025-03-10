#ifndef MONTE_CARLO_HPP
#define MONTE_CARLO_HPP

#include <vector>
#include <random>
#include <functional>
#include <utility>

namespace MC {

    // A simple Monte Carlo integrator that uses importance sampling.
    // The caller must supply a “sampler” lambda that returns a pair: 
    //   (sample configuration, probability density at that sample)
    // and an integrand function that computes f(x) for a given configuration.
    // Then the integral is estimated as the average of f(x)/p(x) over the samples.
    inline double integrate(
        int numSamples,
        std::function<std::pair<std::vector<double>, double>()> sampler,
        std::function<double(const std::vector<double>&)> integrand
    ) {
        double sum = 0.0;
        for (int i = 0; i < numSamples; ++i) {
            auto [x, p] = sampler();
            sum += integrand(x) / p;
        }
        return sum / numSamples;
    }
}

#endif // MONTE_CARLO_HPP
