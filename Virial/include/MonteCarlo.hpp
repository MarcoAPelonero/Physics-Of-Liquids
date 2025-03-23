#ifndef MCHITORMISS_HPP
#define MCHITORMISS_HPP

#include <functional>
#include <vector>
#include <cmath>
#include "mayerSampling.hpp"

double monteCarloHitOrMiss(
    const std::function<double(const std::vector<double>&)>& integrand,
    int dimension,
    int nFreeNodes,     // CHANGED: interpret as number of free nodes
    double sigma,
    long nSamples
);

double MonteCarloMayerMetropolis(
    Configuration config,
    const std::function<double(const std::vector<double>&)>& integrand,
    const std::function<double(const std::vector<double>&)>& referenceIntegrand, // fixed name
    int dimension,
    int nFreeNodes, 
    double sigma,
    long nSamples
);

#endif
