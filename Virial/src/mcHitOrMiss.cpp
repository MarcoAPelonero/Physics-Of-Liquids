#include "mcHitOrMiss.hpp"
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

/**
 * \brief Compute the integral of a cluster integrand (which may be +/-)
 *        using a Metropolis-based "Mayer-sampling" approach with alpha-bridging.
 *
 * Signature matches monteCarloHitOrMiss(...), but the *implementation* uses
 * Mayer-sampling FEP internally, handling sign changes in the integrand.
 *
 * \param integrand    A function that returns the product of Mayer f_{ij}
 *                     for a configuration (may be positive or negative).
 * \param dimension    Dimension of space (usually 3).
 * \param nFreeNodes   Number of free particle coordinates = (nNodes - 1).
 * \param sigma        Characteristic length scale (we use this to define
 *                     a bounding box [ -1.25*sigma, +1.25*sigma ] in each coord).
 * \param nSamples     Total number of Metropolis samples. Internally, we split
 *                     these among bridging steps and sign measurement.
 *
 * \return Monte Carlo estimate of the integral over the domain.
 */
double monteCarloMayerMetropolis(
    const std::function<double(const std::vector<double>&)>& integrand,
    int dimension,
    int nFreeNodes,  // (nNodes - 1)
    double sigma,
    long nSamples
)
{
    // ------------------------------------------------------------------
    // 0) Domain bounding box: [-R, R] in each coordinate
    // ------------------------------------------------------------------
    double R = 1.25 * sigma;
    double sideLength = 2.0 * R;
    double volume = std::pow(sideLength, dimension * nFreeNodes);

    // ------------------------------------------------------------------
    // 1) Setup alpha-bridging parameters:
    // ------------------------------------------------------------------
    int    nAlphaSteps = 80;
    long   bridgingSamples = (long)( (2.0/3.0) * double(nSamples) );
    long   signSamples      = nSamples - bridgingSamples;
    if(signSamples < 1) signSamples = 1;
    long nSamplesPerAlpha = bridgingSamples / (nAlphaSteps);
    std::vector<double> alphaVals(nAlphaSteps + 1, 0.0);
    for(int k = 0; k <= nAlphaSteps; ++k) {
        alphaVals[k] = double(k) / double(nAlphaSteps);
    }
    std::vector<double> logI(nAlphaSteps + 1, 0.0);
    logI[0] = std::log(volume);

    // ------------------------------------------------------------------
    // 2) Random number setup
    // ------------------------------------------------------------------
    static std::mt19937_64 rng(123456789ULL);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    double stepSize = 0.005 * sigma;
    std::uniform_real_distribution<double> unifStep(-stepSize, stepSize);
    std::uniform_real_distribution<double> unifCoord(-R, R);

    // ------------------------------------------------------------------
    // 3) signAndMagnitude(x): returns (sign, magnitude) of integrand(x)
    // ------------------------------------------------------------------
    auto signAndMagnitude = [&](const std::vector<double>& coords){
        double val = integrand(coords);
        double s = (val >= 0.0 ? 1.0 : -1.0);
        double m = std::fabs(val);
        const double EPS = 1e-30;
        if(m < EPS) m = EPS;
        return std::make_pair(s, m);
    };

    auto Galpha = [&](double alpha, double m) {
        return std::pow(m, alpha);
    };

    // ------------------------------------------------------------------
    // 4) Initialize a random coordinate
    // ------------------------------------------------------------------
    std::vector<double> x(dimension * nFreeNodes);
    for(auto &c : x) {
        c = unifCoord(rng);
    }
    double G_old = 1.0;
    auto [s_old, m_old] = signAndMagnitude(x);

    // ------------------------------------------------------------------
    // 5) Bridging loop: alpha=0..1 in steps
    // ------------------------------------------------------------------
    long acceptedBridging = 0;
    long totalBridging = 0;
    for(int k = 0; k < nAlphaSteps; ++k)
    {
        double alpha_k   = alphaVals[k];
        double alpha_kp1 = alphaVals[k+1];
        double sumRatio = 0.0;
        long count = 0;
        for(long sample=0; sample < nSamplesPerAlpha; ++sample)
        {
            totalBridging++;
            std::vector<double> x_new = x;
            for(auto &coord : x_new)
            {
                coord += unifStep(rng);
                if(coord < -R) coord = -R;
                if(coord >  R) coord =  R;
            }
            auto [s_new, m_new] = signAndMagnitude(x_new);
            double G_new = Galpha(alpha_k, m_new);
            double A = G_new / G_old;
            if(A >= 1.0 || unif01(rng) < A)
            {
                acceptedBridging++;
                x = x_new;
                s_old = s_new;
                m_old = m_new;
                G_old = G_new;
            }
            double G_kp1 = Galpha(alpha_kp1, m_old);
            double G_k   = Galpha(alpha_k   , m_old);
            double ratio = G_kp1 / G_k;
            sumRatio += ratio;
            count++;
        }
        double avgRatio = sumRatio / double(count);
        logI[k+1] = logI[k] + std::log(avgRatio);
    }
    double bridgingAcceptance = double(acceptedBridging) / double(totalBridging);
    std::cout << "Bridging Acceptance Ratio: " << bridgingAcceptance << std::endl;

    double I_magnitude = std::exp( logI[nAlphaSteps] );

    // ------------------------------------------------------------------
    // 6) Sign sampling at alpha = 1
    // ------------------------------------------------------------------
    auto [s_cur, m_cur] = signAndMagnitude(x);
    double G_cur = Galpha(1.0, m_cur);
    double sumSign = 0.0;
    long countSign = 0;
    long acceptedSign = 0;
    for(long sample=0; sample < signSamples; ++sample)
    {
        std::vector<double> x_new = x;
        for(auto &coord : x_new)
        {
            coord += unifStep(rng);
            if(coord < -R) coord = -R;
            if(coord >  R) coord =  R;
        }
        auto [s_new, m_new] = signAndMagnitude(x_new);
        double G_new = m_new; // since alpha=1
        double A = G_new / G_cur;
        if(A >= 1.0 || unif01(rng) < A)
        {
            acceptedSign++;
            x     = x_new;
            s_cur = s_new;
            m_cur = m_new;
            G_cur = G_new;
        }
        sumSign += s_cur;
        countSign++;
    }
    double signAcceptance = double(acceptedSign) / double(signSamples);
    std::cout << "Sign Sampling Acceptance Ratio: " << signAcceptance << std::endl;

    double avgSign = sumSign / double(countSign);
    double I_final = I_magnitude * avgSign;
    return I_final;
}