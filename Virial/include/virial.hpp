#ifndef VIRIAL_HPP
#define VIRIAL_HPP

#include "lennardJones.hpp"
#include "graph.hpp"
#include "mcint.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <omp.h>      
#include <functional>
#include <utility>

#define M_PI 3.14159265358979323846

namespace Virial {

    // Given a configuration for vertices 1..n–1 (vertex 0 fixed at the origin),
    // compute the product of Mayer functions for all edges present in graph g.
    // The configuration is stored in a flat vector (length = 3*(n–1)) with each 3 entries giving (x,y,z).
    inline double computeIntegrand(const Graph &g, const std::vector<double>& sample, double beta) {
        int n = g.n;
        // Build an array of positions (vertex 0 = origin)
        std::vector<std::array<double, 3>> positions(n);
        positions[0] = {0.0, 0.0, 0.0};
        for (int i = 1; i < n; ++i) {
            positions[i] = { sample[(i-1)*3], sample[(i-1)*3 + 1], sample[(i-1)*3 + 2] };
        }
        double prod = 1.0;
        // For every edge (i,j) in the graph, multiply by f(r_ij)
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                if (g.adj[i][j]) {
                    double dx = positions[i][0] - positions[j][0];
                    double dy = positions[i][1] - positions[j][1];
                    double dz = positions[i][2] - positions[j][2];
                    double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                    double f = LJ::mayer(r, beta);
                    prod *= f;
                }
            }
        }
        return prod;
    }
    
    // Monte Carlo integration for a given graph.
    // The integration is carried out over the positions of vertices 1..(n–1) (in 3D) with a cutoff r_max.
    // We use importance sampling by generating each vertex’s position from a truncated exponential–angular distribution.
    inline double computeGraphIntegral(const Graph &g, int numSamples, double r_max, double beta) {
        int n = g.n;
        double sum = 0.0;
        // Parallelize the Monte Carlo loop.
        #pragma omp parallel
        {
            // Each thread gets its own random generator.
            std::mt19937 localGen(std::random_device{}());
            std::uniform_real_distribution<double> uniform01(0.0, 1.0);
            // Sampler lambda: returns a configuration and its probability density.
            auto sampler = [&localGen, uniform01, r_max, n]() mutable -> std::pair<std::vector<double>, double> {
                std::vector<double> sample(3 * (n - 1));
                double prob = 1.0;
                for (int i = 0; i < n - 1; ++i) {
                    // Sample radial coordinate from a truncated exponential (λ = 1).
                    double u = uniform01(localGen);
                    double r = -std::log(1 - u * (1 - std::exp(-r_max)));
                    // The PDF for r is: p_r = exp(–r) / (1 – exp(–r_max))
                    double p_r = std::exp(-r) / (1 - std::exp(-r_max));
                    
                    // Sample polar angle theta from p(theta) = sin(theta)/2 over [0,π].
                    double u_theta = uniform01(localGen);
                    double theta = std::acos(1 - 2 * u_theta);
                    double p_theta = std::sin(theta) / 2.0;
                    
                    // Sample azimuthal angle phi uniformly in [0,2π].
                    double phi = 2 * M_PI * uniform01(localGen);
                    double p_phi = 1.0 / (2 * M_PI);
                    
                    // Convert spherical to Cartesian coordinates.
                    double x = r * std::sin(theta) * std::cos(phi);
                    double y = r * std::sin(theta) * std::sin(phi);
                    double z = r * std::cos(theta);
                    
                    sample[i * 3]     = x;
                    sample[i * 3 + 1] = y;
                    sample[i * 3 + 2] = z;
                    
                    double p_val = p_r * p_theta * p_phi;
                    prob *= p_val;
                }
                return std::make_pair(sample, prob);
            };
            
            double localSum = 0.0;
            #pragma omp for
            for (int i = 0; i < numSamples; ++i) {
                auto [sample, p] = sampler();
                double f_val = computeIntegrand(g, sample, beta);
                localSum += f_val / p;
            }
            #pragma omp atomic
            sum += localSum;
        }
        return sum / numSamples;
    }
    
    // Compute the virial coefficient of order n.
    // Bₙ = (–1)^(n–1)/n * Σ₍G∈𝒢ₙ₎ [I(G)/σ(G)]
    // where the sum runs over all irreducible (biconnected) graphs on n vertices.
    inline double computeVirialCoefficient(int n, int numSamples, double r_max, double beta) {
        auto graphs = GraphUtils::generateIrreducibleGraphs(n);
        double sum = 0.0;
        for (const auto &g : graphs) {
            int symm = GraphUtils::automorphismCount(g);
            double I = computeGraphIntegral(g, numSamples, r_max, beta);
            sum += I / symm;
        }
        double B = std::pow(-1.0, n - 1) / n * sum;
        return B;
    }
}

#endif // VIRIAL_HPP
