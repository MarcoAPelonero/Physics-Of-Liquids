#include "graphToIntegrand.hpp"
#include <cmath>

namespace {

// Distance helper when node 0 is fixed at (0,0,...). 
// coords has length = dimension*(nNodes-1). 
// If i=0, we treat that as the origin. Otherwise, i>0 means 
// we read coords[(i-1)*dimension + d].
double distanceFixedNode0(const std::vector<double>& coords,
    int i, int j,
    int dimension,
    double sideLength)
{
    double dist2 = 0.0;
    for (int d = 0; d < dimension; ++d) {
    double xi = (i == 0) ? 0.0 : coords[(i - 1) * dimension + d];
    double xj = (j == 0) ? 0.0 : coords[(j - 1) * dimension + d];
    double diff = xi - xj;
    // Apply the minimum image convention.
    diff = diff - sideLength * std::round(diff / sideLength);
    dist2 += diff * diff;
    }
    return std::sqrt(dist2);
}

} // end anonymous namespace

std::function<double(const std::vector<double>&)>
graphToIntegrand(const NDGraph &graph,
                 PotentialFunction potential,
                 double sigma,
                 double epsilon,
                 int dimension,
                 double beta,
                 double sideLength)
{
    // We assume node 0 is the "root" and won't appear in the coords array.
    // The coords array then has dimension*(n-1) real numbers.
    return [graph, potential, sigma, epsilon, dimension, beta, sideLength](const std::vector<double>& coords) -> double {
        double product = 1.0;
        for (const auto &edge : graph.getEdges()) {
            int i = edge.from;
            int j = edge.to;
            double r = distanceFixedNode0(coords, i, j, dimension, sideLength);
            double f = mayerF(r, sigma, epsilon, potential, beta);
            product *= f;
        }
        return product;
    };
}