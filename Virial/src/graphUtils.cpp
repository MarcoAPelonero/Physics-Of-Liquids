#include "graphUtils.hpp"
#include <algorithm>
#include <string>
#include <set>
#include <vector>
#include <sstream>
#include <limits>
#include <iterator>

namespace {

std::vector<Edge> getCompleteEdgeList(int n) {
    std::vector<Edge> edges;
    edges.reserve(n*(n-1)/2);
    for (int i = 0; i < n; ++i)
        for (int j = i+1; j < n; ++j)
            edges.push_back(Edge(i, j));
    return edges;
}

void generateCombinationsRecur(const std::vector<Edge>& edges, int startIndex, int k, std::vector<Edge>& current, std::vector< std::vector<Edge> >& result) {
    if(k == 0) {
        result.push_back(current);
        return;
    }
    if((int)edges.size() - startIndex < k)
        return;
    current.push_back(edges[startIndex]);
    generateCombinationsRecur(edges, startIndex+1, k-1, current, result);
    current.pop_back();
    generateCombinationsRecur(edges, startIndex+1, k, current, result);
}

std::string computeCanonicalLabel(const NDGraph& graph) {
    int n = graph.getNumNodes();
    const std::vector<Edge>& edges = graph.getEdges();
    std::vector<int> perm(n);
    for(int i=0; i<n; ++i) perm[i] = i;
    std::string best = std::string(n*10, char(127));
    do {
        std::vector<std::pair<int,int>> relabeled;
        for(auto &e : edges) {
            int u = perm[e.from], v = perm[e.to];
            if(u > v) std::swap(u,v);
            relabeled.push_back({u,v});
        }
        std::sort(relabeled.begin(), relabeled.end());
        std::ostringstream oss;
        for(auto &p : relabeled) {
            oss << p.first << "," << p.second << ";";
        }
        std::string label = oss.str();
        if(label < best) best = label;
    } while(std::next_permutation(perm.begin(), perm.end()));
    return best;
}

int computeDegeneracy(const NDGraph& graph) {
    int n = graph.getNumNodes();
    const std::vector<Edge>& edges = graph.getEdges();
    std::vector<std::vector<int>> adj(n);
    for(auto &e : edges) {
        adj[e.from].push_back(e.to);
        adj[e.to].push_back(e.from);
    }
    std::vector<int> degree(n, 0);
    std::vector<bool> removed(n, false);
    for(int i=0;i<n;i++){
        degree[i] = adj[i].size();
    }
    int degeneracy = 0;
    for(int i=0;i<n;i++){
        int minDegree = std::numeric_limits<int>::max();
        int minVertex = -1;
        for(int j=0;j<n;j++){
            if(!removed[j] && degree[j] < minDegree){
                minDegree = degree[j];
                minVertex = j;
            }
        }
        if(minVertex == -1)
            break;
        if(minDegree > degeneracy)
            degeneracy = minDegree;
        removed[minVertex] = true;
        for(auto neighbor : adj[minVertex]){
            if(!removed[neighbor]){
                degree[neighbor]--;
            }
        }
    }
    return degeneracy;
}

NDGraph buildGraphFromRemovals(int n, const std::vector<Edge>& removals) {
    NDGraph g(n, true);
    for(auto &e : removals) {
        g.removeEdge(e.from, e.to);
    }
    return g;
}

} // anonymous namespace

namespace GraphUtils {

std::vector< std::vector<Edge> > generateCombinationsOfSize(const std::vector<Edge>& edges, int k) {
    std::vector< std::vector<Edge> > result;
    if(k < 0 || k > (int)edges.size())
        return result;
    std::vector<Edge> current;
    generateCombinationsRecur(edges, 0, k, current, result);
    return result;
}

std::vector<NDGraph> generateAllConnectedGraphsOptimized(int n) {
    std::vector<NDGraph> results;
    if(n <= 0)
        return results;
    if(n == 1) {
        NDGraph g(1, false);
        int d = computeDegeneracy(g);
        g.setDegeneracy(d);
        results.push_back(g);
        return results;
    }
    std::vector<Edge> allEdges = getCompleteEdgeList(n);
    int M = allEdges.size();
    int minEdges = n-1;
    int maxRemovable = M - minEdges;
    std::set<std::string> canonSet;
    for(int removeCount = 0; removeCount <= maxRemovable; removeCount++){
        std::vector< std::vector<Edge> > removalSubsets = generateCombinationsOfSize(allEdges, removeCount);
        for(auto &subset : removalSubsets){
            NDGraph g = buildGraphFromRemovals(n, subset);
            if(!g.isConnected())
                continue;
            std::string canon = computeCanonicalLabel(g);
            if(canonSet.find(canon) != canonSet.end())
                continue;
            int d = computeDegeneracy(g);
            g.setDegeneracy(d);
            canonSet.insert(canon);
            results.push_back(g);
        }
    }
    return results;
}

std::vector<NDGraph> generateAllBiconnectedGraphsOptimized(int n) {
    std::vector<NDGraph> results;
    if(n <= 0)
        return results;
    if(n == 1)
        return results;
    std::vector<Edge> allEdges = getCompleteEdgeList(n);
    int M = allEdges.size();
    int minEdges = (n <= 2) ? 1 : n;
    int maxRemovable = M - minEdges;
    std::set<std::string> canonSet;
    for(int removeCount = 0; removeCount <= maxRemovable; removeCount++){
        std::vector< std::vector<Edge> > removalSubsets = generateCombinationsOfSize(allEdges, removeCount);
        for(auto &subset : removalSubsets){
            NDGraph g = buildGraphFromRemovals(n, subset);
            if(!g.isBiconnected())
                continue;
            std::string canon = computeCanonicalLabel(g);
            if(canonSet.find(canon) != canonSet.end())
                continue;
            int d = computeDegeneracy(g);
            g.setDegeneracy(d);
            canonSet.insert(canon);
            results.push_back(g);
        }
    }
    return results;
}

} // namespace GraphUtils
