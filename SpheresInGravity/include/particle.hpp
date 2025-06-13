#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "vec.hpp"

class Particle {
private:
    Vec position;
    Vec velocity;
    Vec acceleration;
    Vec pastAcceleration;
    Vec pastVelocity;
    Vec pastPosition;

    double epsilon;
    double sigma;
    double softness;
    double mass;
public:
    Particle() : position(), velocity(), acceleration(), mass(1.0) {}
    Particle(const Vec& pos, const Vec& vel, const Vec& acc, double m) 
        : position(pos), velocity(vel), acceleration(acc), mass(m) {}

    Vec getPosition() const { return position; }
    Vec getVelocity() const { return velocity; }
    Vec getAcceleration() const { return acceleration; }
    Vec getPastAcceleration() const { return pastAcceleration; }
    Vec getPastVelocity() const { return pastVelocity; }
    Vec getPastPosition() const { return pastPosition; }

    double getEpsilon() const { return epsilon; }
    double getSigma() const { return sigma; }
    double getSoftness() const { return softness; }

    double getMass() const { return mass; }

    void setPosition(const Vec& pos) { position = pos; }
    void setVelocity(const Vec& vel) { velocity = vel; }
    void setAcceleration(const Vec& acc) { acceleration = acc; }
    void setPastAcceleration(const Vec& acc) { pastAcceleration = acc; }
    void setPastVelocity(const Vec& vel) { pastVelocity = vel; }
    void setPastPosition(const Vec& pos) { pastPosition = pos; }

    void setEpsilon(double eps) { epsilon = eps; }
    void setSigma(double sig) { sigma = sig; }
    void setSoftness(double soft) { softness = soft; }
    void setMass(double m) { mass = m; }
};

#endif // PARTICLE_HPP