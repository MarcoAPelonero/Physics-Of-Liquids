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

double MonteCarloMayerMetropolis(
    Configuration config,
    const std::function<double(const std::vector<double>&)>& integrand,
    const std::function<double(const std::vector<double>&)>& referenceIntegrand, // fixed name
    int dimension,
    int nFreeNodes, 
    double sigma,
    long nSamples
)
{
    (void) dimension;
    (void) nFreeNodes;
    (void) sigma;

    double delta = 0.01;
    double s_total = 0.0;
    double sRef_total = 0.0;

    std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> uniformDist(0.0, 1.0);

    // Initialize the lattice configuration.
    config.initialLattice();    
    
    for (long i = 0; i < nSamples; ++i)
    {
        std::vector<double> oldPositions = config.getPositions();
        double currentIntegrand = config.computeIntegrandOnConfiguration(integrand);
        double absCurrent = std::abs(currentIntegrand);
        
        config.moveRandomParticle(delta);
        double proposedIntegrand = config.computeIntegrandOnConfiguration(integrand);
        double absProposed = std::abs(proposedIntegrand);

        double acceptanceProb = (absCurrent > 0.0) ? std::min(1.0, absProposed / absCurrent) : 1.0;

        if (uniformDist(rng) >= acceptanceProb)
        {
            config.setPositions(oldPositions);
            double refValue = config.computeIntegrandOnConfiguration(referenceIntegrand);
            double s   = (absCurrent > 0.0) ? currentIntegrand / absCurrent : 0.0;
            double sRef = (absCurrent > 0.0) ? refValue / absCurrent : 0.0;
            s_total  += s;
            sRef_total += sRef;
        }
        else
        {
            double refValue = config.computeIntegrandOnConfiguration(referenceIntegrand);
            double s   = (absProposed > 0.0) ? proposedIntegrand / absProposed : 0.0;
            double sRef = (absProposed > 0.0) ? refValue / absProposed : 0.0;
            s_total  += s;
            sRef_total += sRef;
        }
    }
    double averageS = s_total / static_cast<double>(nSamples);
    double averageSRef = sRef_total / static_cast<double>(nSamples);
    
    return (averageSRef != 0.0) ? averageS / averageSRef : 0.0;
}