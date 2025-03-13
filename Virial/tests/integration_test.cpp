// test_integration.cpp
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <array>
#include <stdexcept>

// Include the integration routines and graph-to-integral functions.
#include "integration.hpp"
#include "graphToIntegral.hpp"
#include "graph.hpp"

// Tolerance for floating-point comparisons.
constexpr double TOL = 1e-2;
constexpr double M_PI = 3.14159265358979323846;
//---------------------------------------------------------------------
// Test 1: Monte Carlo Integration for a 1D linear function f(x)=x over [0,1]
// Exact result: ∫_0^1 x dx = 0.5
//---------------------------------------------------------------------
void testMonteCarloLinear() {
    std::cout << "[TestMonteCarloLinear] Integrating f(x)=x on [0,1]...\n";
    auto f = [](const std::vector<double>& x) -> double {
        return x[0];
    };
    std::vector<std::pair<double, double>> limits = { {0.0, 1.0} };
    int numSamples = 1000000;
    double result = monteCarloIntegration(f, limits, numSamples);
    double expected = 0.5;
    std::cout << "Result: " << result << "   Expected: " << expected << "\n";
    assert(std::fabs(result - expected) < TOL);
}

//---------------------------------------------------------------------
// Test 2: Monte Carlo Integration for a 1D quadratic function f(x)=x^2 over [-1,1]
// Exact result: ∫_{-1}^{1} x^2 dx = 2/3 ≈ 0.66667
//---------------------------------------------------------------------
void testMonteCarloQuadratic() {
    std::cout << "[TestMonteCarloQuadratic] Integrating f(x)=x^2 on [-1,1]...\n";
    auto f = [](const std::vector<double>& x) -> double {
        return x[0] * x[0];
    };
    std::vector<std::pair<double, double>> limits = { {-1.0, 1.0} };
    int numSamples = 1000000;
    double result = monteCarloIntegration(f, limits, numSamples);
    double expected = 2.0 / 3.0;
    std::cout << "Result: " << result << "   Expected: " << expected << "\n";
    assert(std::fabs(result - expected) < TOL);
}

//---------------------------------------------------------------------
// Test 3: Monte Carlo Integration for a 2D function f(x,y)=x+y over [0,1]×[0,1]
// Exact result: ∫_0^1 x dx + ∫_0^1 y dy = 0.5 + 0.5 = 1.0
//---------------------------------------------------------------------
void testMonteCarlo2D() {
    std::cout << "[TestMonteCarlo2D] Integrating f(x,y)=x+y on [0,1]×[0,1]...\n";
    auto f = [](const std::vector<double>& x) -> double {
        return x[0] + x[1];
    };
    std::vector<std::pair<double, double>> limits = { {0.0, 1.0}, {0.0, 1.0} };
    int numSamples = 1000000;
    double result = monteCarloIntegration(f, limits, numSamples);
    double expected = 1.0;
    std::cout << "Result: " << result << "   Expected: " << expected << "\n";
    assert(std::fabs(result - expected) < TOL);
}

//---------------------------------------------------------------------
// Test 4: Monte Carlo Integration for a 1D Gaussian function f(x)=exp(-x^2) on [-3,3]
// Exact result: ∫_{-3}^{3} exp(-x^2) dx = sqrt(pi)*erf(3) ~ sqrt(pi) (erf(3) ~ 0.99998)
// sqrt(pi) ≈ 1.77245
//---------------------------------------------------------------------
void testMonteCarloGaussian1D() {
    std::cout << "[TestMonteCarloGaussian1D] Integrating f(x)=exp(-x^2) on [-3,3]...\n";
    auto f = [](const std::vector<double>& x) -> double {
        return std::exp(-x[0] * x[0]);
    };
    std::vector<std::pair<double, double>> limits = { {-3.0, 3.0} };
    int numSamples = 2000000;
    double result = monteCarloIntegration(f, limits, numSamples);
    double expected = std::sqrt(M_PI); // ≈ 1.77245
    std::cout << "Result: " << result << "   Expected: " << expected << "\n";
    // Allow a slightly looser tolerance for this stochastic integration.
    assert(std::fabs(result - expected) < 0.02);
}

//---------------------------------------------------------------------
// Test 5: computeGraphIntegral for a trivial graph with one node (n=1)
// Expected: Per implementation, the integral should be exactly 1.0
//---------------------------------------------------------------------
void testComputeGraphIntegral_n1() {
    std::cout << "[TestComputeGraphIntegral_n1] Testing computeGraphIntegral for a trivial graph (n=1)...\n";
    NDGraph graph(1, false); // One node, no edges.
    double R = 5.0;
    double result = computeGraphIntegral(graph, R, 1.0, 1.0, 1.0, 1.0, 100000);
    std::cout << "Result: " << result << "   Expected: 1.0\n";
    assert(std::fabs(result - 1.0) < TOL);
}

//---------------------------------------------------------------------
// Test 6: computeGraphIntegral for a graph with 2 nodes and no edge
// Expected: The integrand always returns 1, so the integral equals the volume factor,
//           which is (2R)^3 for one free particle (node 0 fixed at (0,0,0)).
//---------------------------------------------------------------------
void testComputeGraphIntegral_n2_noEdge() {
    std::cout << "[TestComputeGraphIntegral_n2_noEdge] Testing computeGraphIntegral for n=2 with no edges...\n";
    NDGraph graph(2, false); // 2 nodes, no edges.
    double R = 5.0;
    double expected = std::pow(2.0 * R, 3);
    double result = computeGraphIntegral(graph, R, 1.0, 1.0, 1.0, 1.0, 100000);
    std::cout << "Result: " << result << "   Expected (volume factor): " << expected << "\n";
    assert(std::fabs(result - expected) < TOL * expected);
}

//---------------------------------------------------------------------
// Test 7: computeMayerFunction at r = sigma
// At r = sigma, (sigma/sigma)=1 so Lennard-Jones potential U = 4ε(1 - 1) = 0,
// and thus computeMayerFunction should return exp(0) - 1 = 0.
//---------------------------------------------------------------------
void testComputeMayerFunction_sigma() {
    std::cout << "[TestComputeMayerFunction_sigma] Testing computeMayerFunction at r = sigma...\n";
    double epsilon = 1.0, sigma = 1.0, kb = 1.0, T = 1.0;
    double r = sigma;
    double result = computeMayerFunction(r, epsilon, sigma, kb, T);
    double expected = 0.0;
    std::cout << "Result: " << result << "   Expected: " << expected << "\n";
    assert(std::fabs(result - expected) < 1e-6);
}

//---------------------------------------------------------------------
// Test 8: graphToIntegrand for a graph with no edges
// With no edges, the product over edges is 1.0 regardless of coordinates.
//---------------------------------------------------------------------
void testGraphToIntegrand_noEdges() {
    std::cout << "[TestGraphToIntegrand_noEdges] Testing graphToIntegrand for a graph with no edges...\n";
    NDGraph graph(3, false); // 3 nodes, no edges.
    double epsilon = 1.0, sigma = 1.0, kb = 1.0, T = 1.0;
    auto integrand = graphToIntegrand(graph, epsilon, sigma, kb, T);
    // Create sample coordinates for 3 nodes in 3D.
    std::vector<std::array<double, 3>> coords = {
        {0.0, 0.0, 0.0},
        {1.0, 2.0, 3.0},
        {-1.0, -2.0, -3.0}
    };
    double result = integrand(coords);
    double expected = 1.0;
    std::cout << "Result: " << result << "   Expected: " << expected << "\n";
    assert(std::fabs(result - expected) < 1e-6);
}

//---------------------------------------------------------------------
// Test 9: Error handling in computeGraphIntegral for an empty graph (0 nodes)
// The function should throw a runtime error.
//---------------------------------------------------------------------
void testComputeGraphIntegral_error() {
    std::cout << "[TestComputeGraphIntegral_error] Testing error handling in computeGraphIntegral for an empty graph...\n";
    NDGraph graph; // Empty graph (no nodes).
    double R = 5.0;
    bool thrown = false;
    try {
        double result = computeGraphIntegral(graph, R, 1.0, 1.0, 1.0, 1.0, 1000);
        (void)result; // Silence unused variable warning.
    } catch (const std::runtime_error& e) {
        thrown = true;
    }
    assert(thrown && "computeGraphIntegral should throw an error for an empty graph.");
}

//---------------------------------------------------------------------
// Main: Run all tests
//---------------------------------------------------------------------
int main() {
    std::cout << "Starting Integration and GraphToIntegral tests...\n";
    testMonteCarloLinear();
    testMonteCarloQuadratic();
    testMonteCarlo2D();
    testMonteCarloGaussian1D();
    testComputeGraphIntegral_n1();
    testComputeGraphIntegral_n2_noEdge();
    testComputeMayerFunction_sigma();
    testGraphToIntegrand_noEdges();
    testComputeGraphIntegral_error();
    std::cout << "All Integration and GraphToIntegral tests passed successfully.\n";
    return 0;
}
