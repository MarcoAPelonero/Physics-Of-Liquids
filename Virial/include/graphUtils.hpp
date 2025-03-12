#ifndef GRAPHUTILS_HPP
#define GRAPHUTILS_HPP

#include <vector>
#include <algorithm>
#include <string>
#include <set>
#include <sstream>
#include <limits>
#include <iterator>
#include "graph.hpp"
#include "progressBar.hpp"

namespace GraphUtils {

    std::vector< std::vector<Edge> > generateCombinationsOfSize(const std::vector<Edge>& edges, int k);
    std::vector<NDGraph> generateAllConnectedGraphs(int n);
    std::vector<NDGraph> generateAllBiconnectedGraphs(int n);
    std::vector<NDGraph> generateAllConnectedGraphsOptimized(int n, bool show_progress = true);
    std::vector<NDGraph> generateAllBiconnectedGraphsOptimized(int n, bool show_progress = true);

}

#endif // GRAPHUTILS_HPP
