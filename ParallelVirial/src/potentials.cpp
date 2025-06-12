#include "potentials.hpp"

double HS(double r, double sigma, double) {
    if (r < sigma) {
        // Return "infinity" if the particles are overlapping
        return 1.0e6;
    }
    return 0.0;
}

double LJ(double r, double epsilon, double sigma) {
    double sr = sigma / r;
    double sr6 = sr * sr * sr * sr * sr * sr;
    return 4.0 * epsilon * (sr6 * sr6 - sr6);
}
