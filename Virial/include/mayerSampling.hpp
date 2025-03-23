#ifndef MAYER_SAMPLING_HPP
#define MAYER_SAMPLING_HPP

#include <vector>
#include <random>
#include <cmath>
#include "graphToIntegrand.hpp"
#include "potentials.hpp"
#include <functional>
#include <tuple>

#define M_PI 3.14159265358979323846

class Configuration {
private:
    int dimension;                 // Dimensionality of the system.
    int numFreeNodes;             // Number of movable nodes (n-1 if total is n).
    double sigma;                 // Particle size parameter.
    double sideLength;            // Box side length (if needed).
    std::vector<double> positions; // Now a single vector with length = numFreeNodes * dimension.
    std::mt19937_64 rng;          // Random number generator.

public:
    // Constructor that expects the number of *free* nodes.
    Configuration(int dimension, int numFreeNodes, double sigma);
    // Added copy constructor
    Configuration(const Configuration &other) = default;

    // Random initialization (only for the free nodes).
    void initialRandom();

    // Lattice initialization (only for the free nodes).
    void initialLattice();

    Configuration& operator=(const Configuration &other) {
        dimension = other.dimension;
        numFreeNodes = other.numFreeNodes;
        sigma = other.sigma;
        sideLength = other.sideLength;
        positions = other.positions;
        rng = other.rng;
        return *this;
    }
    // Randomly move one particle by up to ±delta in each dimension.
    void moveRandomParticle(double delta);

    // Compute a user-supplied integrand on this configuration.
    double computeIntegrandOnConfiguration(const std::function<double(const std::vector<double>&)> &integrand) const;

    // Getter for the single-vector positions
    // (x1, y1, z1, x2, y2, z2, ...)
    void setPositions(std::vector<double> &newPositions);
    const std::vector<double> &getPositions() const;
};

// Create integrands and config, as before.
std::tuple<
    std::function<double(const std::vector<double>&)>, // integrand for "full" potential
    std::function<double(const std::vector<double>&)>, // integrand for "reference" potential
    Configuration                                      // a newly created config
>
createIntegrandsAndConfig(const NDGraph &graph,
                          PotentialFunction potential,
                          PotentialFunction referencePotential,
                          double sigma,
                          double epsilon,
                          int dimension,
                          double beta);


class HardSpheresCoefficients {
    private:
        std::vector<double> coefficients;
    public:
        HardSpheresCoefficients();

        void changeForm(double sigma);

        double operator[](int i) const;
};

#endif // MAYER_SAMPLING_HPP