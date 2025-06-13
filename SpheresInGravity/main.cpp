#include "particleEnsemble.hpp"

int main() {

    ParticleEnsemble ensemble(10, 0.01, 1.0, 1.0, 1.0, 1.0, Vec(0, 0, -9.81), Vec(10, 10, 10), "output.txt");
    
    ensemble.printEnsemble(0);
    
    return 0;
}