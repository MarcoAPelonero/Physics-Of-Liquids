#ifndef MAYER_SAMPLING_HPP
#define MAYER_SAMPLING_HPP

#include <vector>
#include <random>
#include <cmath>
#include "graphToIntegrand.hpp"  
#include "potentials.hpp"  
#include <functional>
#include <tuple>

class Configuration {
private:
    int dimension;                 // Dimensionality of the system.
    int numFreeNodes;             // Number of movable nodes (n-1 if total is n).
    double sigma;                 // Particle size parameter.
    double sideLength;            // Box side length (if needed).
    std::vector<std::vector<double>> positions; // Positions of the free nodes only!
    std::mt19937_64 rng;          // Random number generator.

public:
    // Constructor that expects the number of *free* nodes. 
    Configuration(int dimension, int numFreeNodes, double sigma);

    // Random initialization (only for the free nodes).
    void initialRandom();

    // Lattice initialization (only for the free nodes).
    void initialLattice();

    // Getter for the free-node positions (each entry has 'dimension' coords).
    const std::vector<std::vector<double>> &getPositions() const;
};

// Suppose you have these signatures in "graphToIntegrand.hpp":
//
// struct NDGraph {
//     int getNumNodes() const;  // or getNumberOfNodes()
//     const std::vector<Edge> &getEdges() const;
//     ...
// };
//
// double distanceFixedNode0(const std::vector<double> &coords,
//                           int i, int j,
//                           int dimension);
//
// double mayerF(double r, double sigma, double epsilon,
//               PotentialFunction potential, double beta);
//
// using PotentialFunction = double(*)(double r, double sigma, double epsilon);
//
// or you can use std::function<double(double,double,double)> if you prefer.

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

#endif // MAYER_SAMPLING_HPP