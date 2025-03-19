#ifndef GRAPH_TO_INTEGRAND_HPP
#define GRAPH_TO_INTEGRAND_HPP

#include "graph.hpp"
#include <functional>
#include <vector>
#include <cmath>

/**
 * @brief Type alias for a potential function that depends on
 *        (r, sigma, epsilon).
 * 
 *        The function signature should be:
 *          double potential(double r, double sigma, double epsilon)
 */
using PotentialFunction = std::function<double(double, double, double)>;

/**
 * @brief Creates an integrand from the given graph.
 *
 * The returned integrand is a function:
 *
 *    double integrand(const std::vector<double> &coords);
 *
 * where coords is a flat vector of 3D coordinates (if dimension=3)
 * or D-dimensional coordinates in general. Node i occupies coords
 * in the index range [i*dimension, i*dimension + (dimension-1)].
 *
 * @param graph       The NDGraph whose edges define which pairs to include.
 * @param potential   A user-supplied pair potential function.
 * @param sigma       The sigma parameter for the potential.
 * @param epsilon     The epsilon parameter for the potential.
 * @param dimension   The space dimension (e.g., 2D or 3D).
 * @return A std::function that calculates the product of Mayer functions
 *         for all edges in the graph, evaluated at the given coordinates.
 */
std::function<double(const std::vector<double> &)>
graphToIntegrand(const NDGraph &graph,
                 PotentialFunction potential,
                 double sigma,
                 double epsilon,
                 int dimension);

/**
 * @brief The standard Mayer function, f(r) = exp(-U(r)) - 1,
 *        given a pair potential U(r).
 * 
 * You can use or modify this if you want to directly compute
 * Mayer for a pair (r, sigma, epsilon).
 */
inline double mayerF(double r, double sigma, double epsilon,
                     const PotentialFunction &potential)
{
    const double U = potential(r, sigma, epsilon);
    return std::exp(-U) - 1.0;
}

#endif // GRAPH_TO_INTEGRAND_HPP
