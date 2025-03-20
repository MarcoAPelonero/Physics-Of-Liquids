#ifndef MCHITORMISS_HPP
#define MCHITORMISS_HPP

#include <functional>
#include <vector>
#include <cmath>

double monteCarloHitOrMiss(
    const std::function<double(const std::vector<double>&)>& integrand,
    int dimension,
    int nFreeNodes,     // CHANGED: interpret as number of free nodes
    double sigma,
    long nSamples
);

double monteCarloMayerMetropolis(
    const std::function<double(const std::vector<double>&)>& integrand,
    int dimension,
    int nFreeNodes,     // CHANGED: interpret as number of free nodes
    double sigma,
    long nSamples
);

#endif
