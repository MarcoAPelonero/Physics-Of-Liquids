#include "particleEnsemble.hpp"

ParticleEnsemble::ParticleEnsemble(int num) : numParticles(num), timeStep(0.01), gravity(0, 0, -9.81), boxDimensions(1, 1, 1) {
    particles.resize(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        particles[i] = Particle(Vec(i * 0.1, i * 0.1, i * 0.1), Vec(0, 0, 0), Vec(0, 0, 0), 1.0);
    }
    outputFile.open("output.txt");
    if (!outputFile) {
        std::cerr << "Failed to open output file." << std::endl;
    }
}

ParticleEnsemble::ParticleEnsemble(int num, double dt, double mass, double sigma, double epsilon ,double softness, 
    const Vec& g, const Vec& boxDim, const std::string& outputFileName)
    : numParticles(num), timeStep(dt), gravity(g), boxDimensions(boxDim), outputFileName(outputFileName), outputFile() { 
    particles.resize(numParticles);
    for (int i = 0; i < numParticles; ++i) {

        double rx = (rand() / (double)RAND_MAX) * boxDimensions.getX();
        double ry = (rand() / (double)RAND_MAX) * boxDimensions.getY();
        double rz = (rand() / (double)RAND_MAX) * boxDimensions.getZ();

        particles[i] = Particle(Vec(rx, ry, rz), Vec(0, 0, 0), Vec(0, 0, 0), 1.0);

        particles[i].setSigma(sigma);
        particles[i].setEpsilon(epsilon);
        particles[i].setSoftness(softness);

    }
    outputFile.open(outputFileName);
    if (!outputFile) {
        std::cerr << "Failed to open output file." << std::endl;
    }
}


ParticleEnsemble::~ParticleEnsemble() {
    if (outputFile.is_open()) {
        outputFile.close();
    }
}

void ParticleEnsemble::computeAcceleration() {
    for (for int ) {
        
    }
}

void ParticleEnsemble::updatePositions(double dt) {
    for (auto& p : particles) {
        Vec newPos = p.getPosition() + p.getVelocity() * dt + p.getAcceleration() * 0.5 * dt * dt;
        p.setPastPosition(p.getPosition());
        p.setPosition(newPos);
    }
}

void ParticleEnsemble::updateVelocities(double dt) {
    for (auto& p : particles) {
        Vec newVel = p.getVelocity() + p.getAcceleration() * dt;
        p.setPastVelocity(p.getVelocity());
        p.setVelocity(newVel);
    }
}

void ParticleEnsemble::printEnsemble(int step) {
    if (outputFile.is_open()) {
        outputFile << "Step: " << step << std::endl;
        outputFile << "Number of Particles: " << numParticles << std::endl;
        outputFile << "Time Step: " << timeStep << std::endl;
        outputFile << "Gravity: (" << gravity.getX() << ", " << gravity.getY() << ", " << gravity.getZ() << ")" << std::endl;
        outputFile << "Box Dimensions: (" << boxDimensions.getX() << ", " << boxDimensions.getY() << ", " << boxDimensions.getZ() << ")" << std::endl;
        outputFile << "Particles:" << std::endl;
        for (const auto& p : particles) {
            outputFile << "Particle Position: (" << p.getPosition().getX() << ", " << p.getPosition().getY() << ", " << p.getPosition().getZ() << ") "
                       << "Velocity: (" << p.getVelocity().getX() << ", " << p.getVelocity().getY() << ", " << p.getVelocity().getZ() << ") "
                       << "Acceleration: (" << p.getAcceleration().getX() << ", " << p.getAcceleration().getY() << ", " << p.getAcceleration().getZ() << ")" 
                       << " Radius: " << p.getSigma() 
                       << std::endl;
        }
    } 
    else {
        std::cerr << "Output file is not open!" << std::endl;
    }
}