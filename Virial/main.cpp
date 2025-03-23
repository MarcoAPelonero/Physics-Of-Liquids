#include "graph.hpp"
#include "graphUtils.hpp"
#include "graphToIntegrand.hpp"
#include "MonteCarlo.hpp"
#include "potentials.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#define M_PI 3.14159265358979323846

int main(int argc, char **argv) {

    int order = 3;
    int nSamples = 20000000;
    int dimension = 3;
    double sigma = 1.0;
    double epsilon = 1.0;
    double T = 1.0;
    std::string outfileName = "results.txt";

    if (argc > 1) {
        order = std::stoi(argv[1]);
    }
    if (argc > 2) {
        nSamples = std::stoi(argv[2]);
    }
    if (argc > 3) {
        dimension = std::stoi(argv[3]);
    }
    if (argc > 4) {
        sigma = std::stod(argv[4]);
    }
    if (argc > 5) {
        epsilon = std::stod(argv[5]);
    }
    if (argc > 6) {
        T = std::stod(argv[6]);
    }
    if (argc > 7) {
        outfileName = argv[7];
    }

    std::cout << "Order: " << order << std::endl;
    std::cout << "nSamples: " << nSamples << std::endl;
    std::cout << "Dimension: " << dimension << std::endl;
    std::cout << "Sigma: " << sigma << std::endl;
    std::cout << "Epsilon: " << epsilon << std::endl;
    std::cout << "T: " << T << std::endl;

    double beta = 1.0 / T;

    // Generate Vector of n values to save the virial coefficients computed
    std::vector<double> virialCoefficientsHitOrMiss(order + 1, 0.0);
    std::vector<double> virialCoefficientsMetropolis(order + 1, 0.0);

    // Hard-sphere potential. Sigma is DIAMETER
    PotentialFunction potHS = [](double r, double sigma, double epsilon) {
        // If r < sigma => "infinite" => exp(-U)=0 => f(r)=-1
      return LJ(r, sigma, epsilon);
    };

    // Generate all biconnected graphs up to isomorphism
    std::cout << "##########################################" << std::endl;
    std::cout << "Running Hit or Miss Monte Carlo Integration" << std::endl;
    std::cout << "##########################################" << std::endl;

    for (int n = 0; n<=order; ++n) {
        if (n < 2) {
            virialCoefficientsHitOrMiss[n] = 0.0;
            continue;
        }
        
        std::vector<NDGraph> graphs = GraphUtils::generateBiconnectedGraphsNoIsomorphism(n);
        
        // Compute the integrals associated to this order
        double integral = 0.0;

        int counter = 0;

        for (auto &g : graphs) {
            auto integrand = graphToIntegrand(g, potHS, sigma, epsilon, dimension, beta, 5.0*sigma);
            double estimate = monteCarloHitOrMiss(integrand, dimension, n-1, sigma, nSamples);
            double deg = GraphUtils::computeDegeneracy(g);
            integral += estimate * deg;
            counter++;
        }

        double factor = -(n-1) / ( std::tgamma(n+1) );
        
        double finalContribution = factor * integral;
        virialCoefficientsHitOrMiss[n] = finalContribution;
    }

    // Print on an outfile the results
    std::ofstream outfile(outfileName);
    outfile << "Virial coefficients in terms of the packing fraction:" << std::endl;
    outfile << "T* = " << T << std::endl;

    double coeff = (M_PI / 3.0) * 2;
    double v0 = coeff * sigma * sigma * sigma;
    
    for (int i = 0; i <= order; ++i) {
        outfile << "n=" << i << ": ";
        if (i < 2) {
            outfile << "0 " << std::endl;
            continue;
        }
        double term = std::pow(v0, i-1);
        double virialCoefficient = virialCoefficientsHitOrMiss[i] / term;
        outfile << virialCoefficient << " " << std::endl;

    }
    return 0;
}