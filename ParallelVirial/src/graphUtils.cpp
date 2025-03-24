#include "graphUtils.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <string>

// For permutations:
#include <numeric>   // std::iota
#include <sstream>   // building adjacency matrix strings

namespace {

// Helper: build an NxN adjacency matrix (as a 2D vector<bool>) from NDGraph edges
std::vector<std::vector<bool>> buildAdjMatrix(const NDGraph &graph) {
    int n = graph.getNumNodes();
    std::vector<std::vector<bool>> adj(n, std::vector<bool>(n, false));

    // The NDGraph edges are undirected, so set both [from][to] and [to][from] to true.
    auto edges = graph.getEdges();
    for (auto &e : edges) {
        adj[e.from][e.to] = true;
        adj[e.to][e.from] = true;
    }
    return adj;
}

// Helper: produce adjacency matrix string for a *particular labeling* of the nodes
// labeling[i] = new label for old node i. We reorder rows & columns accordingly.
std::string adjacencyMatrixString(const std::vector<std::vector<bool>> &originalAdj,
                                  const std::vector<int> &labeling) {
    int n = (int)labeling.size();
    // We'll build a text row for each row in the permuted adjacency matrix
    std::ostringstream oss;
    for (int i = 0; i < n; ++i) {
        int oldRow = labeling[i];
        for (int j = 0; j < n; ++j) {
            int oldCol = labeling[j];
            oss << (originalAdj[oldRow][oldCol] ? '1' : '0');
        }
        oss << '\n';
    }
    return oss.str();
}

} // end anonymous namespace

namespace GraphUtils {

//------------------------------------------------------------------------------
// 1) Canonical Label (Simple “in-house nauty” by brute-force permutations)
//------------------------------------------------------------------------------

std::string getCanonicalLabel(const NDGraph &graph) {
    int n = graph.getNumNodes();
    if (n == 0) {
        // If no nodes, just return something trivial
        return "";
    }
    // Build the adjacency matrix (size n x n).
    auto adj = buildAdjMatrix(graph);

    // Generate all permutations of {0, 1, ..., n-1}.
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    // We'll track the lexicographically minimal adjacency matrix representation
    // as our canonical label.
    bool firstTime = true;
    std::string bestLabel;

    do {
        std::string matrixStr = adjacencyMatrixString(adj, perm);
        if (firstTime) {
            bestLabel = matrixStr;
            firstTime = false;
        } else {
            // Keep whichever is lexicographically smaller
            if (matrixStr < bestLabel) {
                bestLabel = matrixStr;
            }
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    return bestLabel;
}

bool areIsomorphic(const NDGraph &g1, const NDGraph &g2) {
    if (g1.getNumNodes() != g2.getNumNodes() ||
        g1.getNumEdges() != g2.getNumEdges()) {
        // Quick rejection: different number of nodes or edges => not isomorphic
        return false;
    }
    // Otherwise, compare canonical forms
    return (getCanonicalLabel(g1) == getCanonicalLabel(g2));
}

//------------------------------------------------------------------------------
// 2) Generate Biconnected Graphs Up to Isomorphism
//------------------------------------------------------------------------------

std::vector<NDGraph> generateBiconnectedGraphsNoIsomorphism(int numNodes) {
    std::vector<NDGraph> biconnectedReps;
    if (numNodes <= 0) {
        return biconnectedReps;
    }

    // We'll store canonical forms to detect duplicates
    std::unordered_set<std::string> seenCanonicalForms;
    seenCanonicalForms.reserve(1 << (numNodes*(numNodes-1)/2));

    // All possible edges (i < j)
    std::vector<std::pair<int, int>> possibleEdges;
    for (int i = 0; i < numNodes; ++i) {
        for (int j = i + 1; j < numNodes; ++j) {
            possibleEdges.push_back({i, j});
        }
    }
    int maxEdges = (int)possibleEdges.size();
    int totalSubsets = 1 << maxEdges;

    for (int mask = 0; mask < totalSubsets; ++mask) {
        NDGraph g(numNodes, false);
        // Add edges according to bits in mask
        for (int e = 0; e < maxEdges; ++e) {
            if (mask & (1 << e)) {
                auto &ed = possibleEdges[e];
                g.addEdge(ed.first, ed.second);
            }
        }
        // Check for connectivity + biconnectivity
        if (g.isConnected() && g.isBiconnected()) {
            // Compute canonical label
            std::string cForm = getCanonicalLabel(g);
            // Insert into set if we haven't seen it
            if (seenCanonicalForms.find(cForm) == seenCanonicalForms.end()) {
                seenCanonicalForms.insert(cForm);
                biconnectedReps.push_back(g);
            }
        }
    }
    return biconnectedReps;
}

//------------------------------------------------------------------------------
// 3) Degeneracy
//------------------------------------------------------------------------------

double computeDegeneracy(const NDGraph &graph) {
    int n = graph.getNumNodes();
    // Compute factorial of n
    unsigned long long factorial = 1ULL;
    for (int i = 2; i <= n; ++i) {
        factorial *= i;
    }

    int automorphismCount = graph.getAutomorphismCount();
    if (automorphismCount == 0) {
        throw std::runtime_error("Automorphism count is zero, cannot compute degeneracy.");
    }
    
    double degeneracy = static_cast<double>(factorial) / automorphismCount;
    return degeneracy;
}

} // namespace GraphUtils
