#include "graphToIntegrand.hpp"
#include <cmath>

namespace {

// Distance helper when node 0 is fixed at (0,0,...). 
// coords has length = dimension*(nNodes-1). 
// If i=0, we treat that as the origin. Otherwise, i>0 means 
// we read coords[(i-1)*dimension + d].
double distanceFixedNode0(const std::vector<double> &coords,
                          int i, int j,
                          int dimension)
{
    double dist2 = 0.0;
    for(int d=0; d<dimension; ++d)
    {
        double xi = (i == 0) ? 0.0 : coords[(i-1)*dimension + d];
        double xj = (j == 0) ? 0.0 : coords[(j-1)*dimension + d];
        double diff = xi - xj;
        dist2 += diff*diff;
    }
    return std::sqrt(dist2);
}

} // end anonymous namespace

std::function<double(const std::vector<double> &)>
graphToIntegrand(const NDGraph &graph,
                 PotentialFunction potential,
                 double sigma,
                 double epsilon,
                 int dimension)
{
    // We assume node 0 is the "root" and won't appear in the coords array.
    // The coords array then has dimension*(n-1) real numbers.
    return [graph, potential, sigma, epsilon, dimension](const std::vector<double> &coords) -> double
    {
        double product = 1.0;
        for(const auto &edge : graph.getEdges())
        {
            int i = edge.from;
            int j = edge.to;
            double r = distanceFixedNode0(coords, i, j, dimension);

            double f = mayerF(r, sigma, epsilon, potential);
            product *= f;
        }
        return product;
    };
}
