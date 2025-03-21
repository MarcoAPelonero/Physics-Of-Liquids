#include "MonteCarlo.hpp"
#include <random>
#include <cmath>
#include <iostream> // <-- add this include for printing

double monteCarloHitOrMiss(
    const std::function<double(const std::vector<double>&)>& integrand,
    int dimension,
    int nFreeNodes,  // we've decided this = (nNodes - 1)
    double sigma,
    long nSamples
)
{
    // Integrate over [0, 2.5 * sigma]^(dimension * nFreeNodes).
    double sideLength = 5.0 * sigma;
    double volume = std::pow(sideLength, dimension * nFreeNodes);

    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> dist(-sideLength/2, sideLength/2);

    double sum = 0.0;
    std::vector<double> coords(dimension * nFreeNodes, 0.0);

    for(long i=0; i<nSamples; ++i)
    {
        for(int c=0; c<dimension * nFreeNodes; ++c)
            coords[c] = dist(rng);

        double fVal = integrand(coords);
        sum += fVal;
    }

    double avg = sum / double(nSamples);
    return volume * avg;
}

