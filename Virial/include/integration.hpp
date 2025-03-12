#ifndef INTEGRATION_HPP
#define INTEGRATION_HPP

#include <cmath>
#include <vector>
#include <functional>
#include <random>
#include <stdexcept>

// Computes the Lennard-Jones potential: U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
inline double LennardJonesPotential(double r, double epsilon, double sigma) {
    double sr6 = std::pow(sigma / r, 6);
    return 4.0 * epsilon * (std::pow(sr6, 2) - sr6);
}

// Computes the Mayer function: f(r) = exp(-U(r)/(k_B * T)) - 1
inline double computeMayerFunction(double r, double epsilon, double sigma,
                                   double kb, double T) {
    double U = LennardJonesPotential(r, epsilon, sigma);
    return std::exp(-U / (kb * T)) - 1.0;
}

// Define a generic integrand type: function from a vector of variables to a double.
using Integrand = std::function<double(const std::vector<double>&)>;

// Performs Monte Carlo integration over a hyper-rectangular domain.
// 'limits' is a vector of pairs defining the lower and upper limit for each integration variable.
// 'numSamples' sets the number of Monte Carlo samples.
inline double monteCarloIntegration(const Integrand& f,
                                    const std::vector<std::pair<double, double>>& limits,
                                    int numSamples = 1000000) {
    const int dimensions = limits.size();
    double volume = 1.0;
    for (const auto& lim : limits) {
        volume *= (lim.second - lim.first);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::uniform_real_distribution<double>> distributions;
    for (const auto& lim : limits) {
        distributions.emplace_back(lim.first, lim.second);
    }

    double sum = 0.0;
    std::vector<double> point(dimensions);
    for (int i = 0; i < numSamples; ++i) {
        for (int j = 0; j < dimensions; ++j) {
            point[j] = distributions[j](gen);
        }
        sum += f(point);
    }
    return volume * sum / numSamples;
}

#endif // INTEGRATION_HPP

