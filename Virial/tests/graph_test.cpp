// test_graph.cpp
#include <iostream>
#include <cassert>
#include <stdexcept>
#include "graph.hpp"

//----------------------------------------------------------------------
// Test 1: Empty Graph (default constructor)
//----------------------------------------------------------------------
void testEmptyGraph() {
    std::cout << "[TestEmptyGraph] Running empty graph tests...\n";
    NDGraph emptyGraph;
    // Default-constructed graph should have zero nodes and edges.
    assert(emptyGraph.getNumNodes() == 0);
    assert(emptyGraph.getNumEdges() == 0);
    // isConnected returns true by definition for an empty graph.
    assert(emptyGraph.isConnected());
}

//----------------------------------------------------------------------
// Test 2: Graph Constructor with Complete Graph Option
//----------------------------------------------------------------------
void testGraphConstructorAndComplete() {
    std::cout << "[TestGraphConstructorAndComplete] Running complete graph tests...\n";
    // Create a complete graph with 4 nodes.
    NDGraph completeGraph(4, true);
    assert(completeGraph.getNumNodes() == 4);
    // In a complete graph, number of edges = n(n-1)/2 = 4*3/2 = 6.
    assert(completeGraph.getNumEdges() == 6);
    // Complete graphs are connected and biconnected.
    assert(completeGraph.isConnected());
    assert(completeGraph.isBiconnected());
}

//----------------------------------------------------------------------
// Test 3: Adding and Removing Edges
//----------------------------------------------------------------------
void testAddAndRemoveEdge() {
    std::cout << "[TestAddAndRemoveEdge] Running add/remove edge tests...\n";
    NDGraph graph(5, false);
    // Initially, there should be no edges.
    assert(graph.getNumEdges() == 0);
    
    // Add several edges.
    graph.addEdge(0, 1);
    graph.addEdge(1, 2);
    graph.addEdge(3, 4);
    assert(graph.getNumEdges() == 3);
    
    // Graph is not fully connected in this state.
    assert(!graph.isConnected());
    
    // Remove an edge and check the count.
    graph.removeEdge(1, 2);
    assert(graph.getNumEdges() == 2);
    
    // Removing a non-existent edge should not change the edge count.
    graph.removeEdge(1, 2);
    assert(graph.getNumEdges() == 2);
}

//----------------------------------------------------------------------
// Test 4: Adding and Removing Nodes
//----------------------------------------------------------------------
void testAddAndRemoveNode() {
    std::cout << "[TestAddAndRemoveNode] Running add/remove node tests...\n";
    NDGraph graph(3, false);
    assert(graph.getNumNodes() == 3);
    
    // Add a new node. It should receive the next available index.
    int newNode = graph.addNode();
    assert(newNode == 3);
    assert(graph.getNumNodes() == 4);
    
    // Optionally add an edge involving the new node.
    graph.addEdge(3, 0);
    assert(graph.getNumEdges() == 1);
    
    // Remove an existing node.
    int removedNode = graph.removeNode(1);
    // After removal, node count should decrease.
    assert(graph.getNumNodes() == 3);
    // The removed node's id should be returned.
    assert(removedNode == 1);
    
    // Note: The removal does not update edges, so further tests may be needed in your codebase.
}

//----------------------------------------------------------------------
// Test 5: Connectivity Checks
//----------------------------------------------------------------------
void testIsConnected() {
    std::cout << "[TestIsConnected] Running connectivity tests...\n";
    // Create a graph with 4 nodes and add only one edge.
    NDGraph graph(4, false);
    graph.addEdge(0, 1);
    // With nodes 2 and 3 isolated, the graph should not be connected.
    assert(!graph.isConnected());
    
    // Add edges to connect all nodes.
    graph.addEdge(1, 2);
    graph.addEdge(2, 3);
    assert(graph.isConnected());
}

//----------------------------------------------------------------------
// Test 6: Biconnectivity Checks
//----------------------------------------------------------------------
void testIsBiconnected() {
    std::cout << "[TestIsBiconnected] Running biconnectivity tests...\n";
    // (A) Complete triangle graph: should be biconnected.
    NDGraph triangleGraph(3, false);
    triangleGraph.addEdge(0, 1);
    triangleGraph.addEdge(1, 2);
    triangleGraph.addEdge(0, 2);
    assert(triangleGraph.isConnected());
    assert(triangleGraph.isBiconnected());
    
    // (B) Linear chain: node 1 is an articulation point.
    NDGraph chainGraph(3, false);
    chainGraph.addEdge(0, 1);
    chainGraph.addEdge(1, 2);
    assert(chainGraph.isConnected());
    assert(!chainGraph.isBiconnected());
    
    // (C) Single node graph: trivially biconnected.
    NDGraph singleNodeGraph(1, false);
    assert(singleNodeGraph.isConnected());
    assert(singleNodeGraph.isBiconnected());
}

//----------------------------------------------------------------------
// Test 7: Error Handling for Invalid Inputs
//----------------------------------------------------------------------
void testErrorHandling() {
    std::cout << "[TestErrorHandling] Running error handling tests...\n";
    // Test: Creating a graph with non-positive number of nodes should throw.
    try {
        NDGraph invalidGraph(0, false);
        // Should not reach here.
        assert(false);
    } catch (const std::invalid_argument& e) {
        // Exception expected.
    }
    
    NDGraph graph(3, false);
    // Test: addEdge with an invalid index.
    try {
        graph.addEdge(-1, 2);
        assert(false);
    } catch (const std::out_of_range& e) {
        // Expected exception.
    }
    
    try {
        graph.addEdge(0, 5);
        assert(false);
    } catch (const std::out_of_range& e) {
        // Expected exception.
    }
    
    // Test: removeNode with an invalid index.
    try {
        graph.removeNode(10);
        assert(false);
    } catch (const std::out_of_range& e) {
        // Expected exception.
    }
}

//----------------------------------------------------------------------
// Main: Run all tests
//----------------------------------------------------------------------
int main() {
    std::cout << "Starting NDGraph tests...\n";
    testEmptyGraph();
    testGraphConstructorAndComplete();
    testAddAndRemoveEdge();
    testAddAndRemoveNode();
    testIsConnected();
    testIsBiconnected();
    testErrorHandling();
    std::cout << "All NDGraph tests passed successfully.\n";
    return 0;
}
