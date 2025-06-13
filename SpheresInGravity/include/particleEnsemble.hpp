#ifndef PARTICLEENSEMBLE_HPP
#define PARTICLEENSEMBLE_HPP

#include "particle.hpp"
#include "vec.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>

class ParticleEnsemble {
private:
    std::vector<Particle> particles;

    int numParticles;
    double timeStep;

    Vec gravity;

    Vec boxDimensions;

    std::string outputFileName;
    std::ofstream outputFile;

public:
    ParticleEnsemble(int num);
    ParticleEnsemble(int num, double dt, double mass, double sigma, double epsilon, double softness, 
        const Vec& g, const Vec& boxDim, const std::string& outputFileName = "output.txt");

    ~ParticleEnsemble();

    void updatePositions(double dt);

    void updateVelocities(double dt);

    void updateEnsemble(double dt);

    int getNumParticles() const { return numParticles;}
    double getTimeStep() const { return timeStep; }
    Vec getGravity() const { return gravity; }
    Vec getBoxDimensions() const { return boxDimensions; }

    Particle operator[](int i) const { return particles[i]; }

    void printEnsemble(int step);
};

#endif // PARTICLE_HPP