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

double MonteCarloMetropolisMayer(
    const std::function<double(const std::vector<double>&)>& integrandFull,
    const std::function<double(const std::vector<double>&)>& integrandRef,
    int dimension,
    int nFreeNodes,  // (nNodes - 1)
    double sigma,
    long nSamples,
    bool useBridging   // if true, use the alpha–bridging procedure; otherwise, use a “direct” Metropolis sampling of the full integrand.
);

#endif
