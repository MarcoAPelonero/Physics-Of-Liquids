#include "graph.hpp"
#include "graphUtils.hpp"
#include "graphToIntegrand.hpp"
#include "mayerSampling.hpp"
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

    std::vector<double> virialCoefficientsMayerMetropolis(order, 0.0);
    std::vector<double> virialCoefficientsHardSpheres(10, 0.0);

    // Define the reference values for the HS potential

    virialCoefficientsHardSpheres[0] = 0.0; 
    virialCoefficientsHardSpheres[1] = 0.0; 
    virialCoefficientsHardSpheres[2] = 4.0; 
    virialCoefficientsHardSpheres[4] = 10.0;
    virialCoefficientsHardSpheres[5] = 18.365;
    virialCoefficientsHardSpheres[6] = 28.244;
    virialCoefficientsHardSpheres[7] = 39.82;
    virialCoefficientsHardSpheres[8] = 53.34;
    virialCoefficientsHardSpheres[9] = 68.54;

    PotentialFunction potential = [](double r, double sigma, double epsilon) {
      return LJ(r, sigma, epsilon);
    };
    PotentialFunction referencePotential = [](double r, double sigma, double epsilon) {
        return HS(r,sigma,epsilon);
    };

    // Compute all biconnected graphs for each order

    for (int n = 0; n<=order; ++n) {
        if (n < 2) {
            virialCoefficientsMayerMetropolis[n] = 0.0;
            continue;
        }
        
        std::vector<NDGraph> graphs = GraphUtils::generateBiconnectedGraphsNoIsomorphism(n);
        
        double integral = 0.0;

        int counter = 0;

        for (auto &g : graphs) {
            auto [integrandFull, integrandRef, config] = createIntegrandsAndConfig(
                g,
                potential,
                referencePotential,
                sigma,
                epsilon,
                dimension,
                beta
            );
            
            double deg = GraphUtils::computeDegeneracy(g);
            integral += estimate * deg;
            counter++;
        }

        double factor = -(n-1) / ( std::tgamma(n+1) );
        
        double finalContribution = factor * integral;
        virialCoefficientsMayerMetropolis[n] = finalContribution;
    }

    return 0;
}