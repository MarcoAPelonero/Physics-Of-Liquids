#include "mayerSampling.hpp"
#include "MonteCarlo.hpp"
#include "potentials.hpp"
#include "graphUtils.hpp"

#include <iostream>
#include <fstream>
#include <cmath>

#include <vector>

#define M_PI 3.14159265358979323846

int main() {
    int n = 2;
    int nSamples = 1000000;
    int dimension = 3;
    double sigma = 1.0;
    double epsilon = 1.0;
    double T = 1.0;

    double beta = 1.0 / T;

    // First, test everything is working as expected
    std::vector<NDGraph> graphs = GraphUtils::generateBiconnectedGraphsNoIsomorphism(n);
        
    double integral = 0.0;

    PotentialFunction potHS = [](double r, double sigma, double epsilon) {
        // If r < sigma => "infinite" => exp(-U)=0 => f(r)=-1
      return HS(r, sigma, epsilon);
    };
    PotentialFunction potLJ = [](double r, double sigma, double epsilon) {
        // If r < sigma => "infinite" => exp(-U)=0 => f(r)=-1
      return LJ(r, sigma, epsilon);
    };

    double v0 = (M_PI / 6.0) * sigma * sigma * sigma;

    for (auto &g : graphs) {
        auto integrand = graphToIntegrand(g, potHS, sigma, epsilon, dimension, beta, 5.0*sigma);
        double estimate = monteCarloHitOrMiss(integrand, dimension, n-1, sigma, nSamples);
        double deg = GraphUtils::computeDegeneracy(g);
        integral = estimate * deg;
        std::cout << "Estimation of HS coefficient: " << integral << std::endl;
        HardSpheresCoefficients hsCoeffs;
        hsCoeffs.getGraphIntegral(sigma);
        std::cout << "HS Coefficient: " << hsCoeffs[n] << std::endl;
        double factor = -(n-1) / ( std::tgamma(n+1) );
        double finalContribution = factor * integral;
        std::cout << "Final contribution: " << finalContribution/pow(v0, n-1) << std::endl;
    }

    // Now, test the Mayer Metropolis algorithm

    for (auto &g : graphs) {
        auto [integrandFull, integrandRef, config] = createIntegrandsAndConfig(
            g,
            potLJ,
            potLJ,
            sigma,
            epsilon,
            dimension,
            beta
        );
        
        double estimate = MonteCarloMayerMetropolis(
            config,
            integrandFull,
            integrandRef,
            nSamples
        );
        std::cout << "Estimation on 2 HS potentials (expected 1): " << estimate << std::endl;
    }

    // Try to compute LJ integrals (should be n=2: -2.38656)
    for (auto &g : graphs) {
        auto [integrandFull, integrandRef, config] = createIntegrandsAndConfig(
            g,
            potLJ,
            potHS,
            sigma,
            epsilon,
            dimension,
            beta
        );
        
        double estimate = MonteCarloMayerMetropolis(
            config,
            integrandFull,
            integrandRef,
            nSamples
        );

        auto refIntegrand = graphToIntegrand(g, potHS, sigma, epsilon, dimension, beta, 5.0*sigma);
        double refEstimate = monteCarloHitOrMiss(refIntegrand, dimension, n-1, sigma, nSamples);

        double val = estimate * refEstimate;
        double factor = -(n-1) / ( std::tgamma(n+1) );
        double normal = factor * val;
        double volume = (2 * M_PI)/3 * sigma * sigma * sigma;
        std::cout << "Estimation on LJ and HS potentials (expected -2.38656): " << normal/pow(volume, n-1) << std::endl;
    }
    return 0;
}