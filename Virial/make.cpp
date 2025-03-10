#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "include/virial.hpp"

int main(int argc, char* argv[]) {
    // Maximum order to compute (n = 2 to maxOrder). Default is 5.
    int maxOrder = 3;
    if (argc > 1) {
        maxOrder = std::atoi(argv[1]);
        if(maxOrder < 2) {
            std::cerr << "Maximum order must be >= 2.\n";
            return 1;
        }
    }
    
    // Monte Carlo parameters.
    int numSamples = 100000;  // Increase this for better precision.
    double r_max = 5.0;       // Cutoff for integration (in units of sigma).
    double beta = 1.0;        // Reduced inverse temperature.
    
    std::cout << "Computing virial coefficients for the Lennard-Jones potential (β = " 
              << beta << ")\n";
    std::cout << "Using " << numSamples << " Monte Carlo samples per graph.\n\n";
    
    std::cout << std::setw(10) << "Order" 
              << std::setw(25) << "B_n (computed)" << "\n";
    std::cout << "-------------------------------------\n";
    for (int n = 2; n <= maxOrder; ++n) {
        double B = Virial::computeVirialCoefficient(n, numSamples, r_max, beta);
        std::cout << std::setw(10) << n 
                  << std::setw(25) << B << "\n";
    }
    
    // For reference, here are literature–based values (in reduced units) for n = 2 to 5:
    std::cout << "\nReference virial coefficients (approximate):\n";
    std::cout << "   Order     B_n\n";
    std::cout << "    2      -1.523\n";
    std::cout << "    3       0.362\n";
    std::cout << "    4      -0.080\n";
    std::cout << "    5       0.012\n";
    
    return 0;
}
