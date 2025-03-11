#ifndef GRAPHUTILS_HPP
#define GRAPHUTILS_HPP

#include <vector>
#include "graph.hpp"

namespace GraphUtils {

    std::vector< std::vector<Edge> > generateCombinationsOfSize(const std::vector<Edge>& edges, int k);
    std::vector<NDGraph> generateAllConnectedGraphsOptimized(int n);
    std::vector<NDGraph> generateAllBiconnectedGraphsOptimized(int n);

}

#endif // GRAPHUTILS_HPP
