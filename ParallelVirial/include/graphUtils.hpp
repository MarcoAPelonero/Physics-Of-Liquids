#ifndef GRAPH_UTILS_HPP
#define GRAPH_UTILS_HPP

#include <vector>
#include <string>
#include <unordered_set>
#include "graph.hpp"

namespace GraphUtils {

    // Generates all biconnected graphs (up to isomorphism) with the specified number of nodes.
    // That is, no two output graphs will be isomorphic. We do this by brute-forcing all 2^(n(n-1)/2)
    // labeled graphs, checking connectivity & biconnectivity, then filtering out duplicates up to isomorphism.
    std::vector<NDGraph> generateBiconnectedGraphsNoIsomorphism(int numNodes);

    // Computes the degeneracy of the given graph, defined as N! / g, where
    // N is the number of nodes and g is the automorphism count of the graph.
    double computeDegeneracy(const NDGraph &graph);

    // ------------------------------------------------------------------------
    // Below are helper routines for “in-house nauty.” They can be used standalone
    // if desired, but they’re also used internally by generateBiconnectedGraphsNoIsomorphism.

    // Returns a lexicographically minimal adjacency-matrix string that serves
    // as a canonical form for the graph. We do this by brute-forcing all permutations
    // of node labels and picking the adjacency matrix ordering that is lexicographically smallest.
    std::string getCanonicalLabel(const NDGraph &graph);

    // Tests if two graphs are isomorphic by checking if they have the same canonical label.
    bool areIsomorphic(const NDGraph &g1, const NDGraph &g2);

} // namespace GraphUtils

#endif // GRAPH_UTILS_HPP
