#ifndef LENNARD_JONES_HPP
#define LENNARD_JONES_HPP

#include <cmath>

namespace LJ {
    constexpr double sigma = 1.0;
    constexpr double epsilon = 1.0;
    
    // Lennard–Jones potential in reduced units (r in units of sigma).
    inline double potential(double r) {
        double sr = sigma / r;
        double sr6 = std::pow(sr, 6);
        double sr12 = sr6 * sr6;
        return 4.0 * epsilon * (sr12 - sr6);
    }
    
    // Mayer f–function: f(r) = exp(–βU(r)) – 1.
    // Here we assume reduced units with β = 1 by default.
    inline double mayer(double r, double beta = 1.0) {
        return std::exp(-beta * potential(r)) - 1.0;
    }
}

#endif // LENNARD_JONES_HPP