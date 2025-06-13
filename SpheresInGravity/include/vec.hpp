#ifndef VEC_HPP
#define VEC_HPP

#include <cmath>

class Vec {
private:
    double x, y, z;
public:

    Vec() : x(0), y(0), z(0) {}
    Vec(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec operator+(const Vec& other) const {
        return Vec(x + other.x, y + other.y, z + other.z);
    }

    Vec operator-(const Vec& other) const {
        return Vec(x - other.x, y - other.y, z - other.z);
    }

    Vec operator*(float scalar) const {
        return Vec(x * scalar, y * scalar, z * scalar);
    }

    Vec operator/(float scalar) const {
        return Vec(x / scalar, y / scalar, z / scalar);
    }

    Vec operator-() const {
        return Vec(-x, -y, -z);
    }

    Vec& operator+=(const Vec& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vec& operator-=(const Vec& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    Vec& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    Vec& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    double dot(const Vec& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Vec cross(const Vec& other) const {
        return Vec(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    double length() const {
        return sqrt(x * x + y * y + z * z);
    }

    double lengthSquared() const {
        return x * x + y * y + z * z;
    }

    void normalize() {
        double len = length();
        if (len > 0) {
            x /= len;
            y /= len;
            z /= len;
        }
    }

    double getX() const { return x; }
    double getY() const { return y; }
    double getZ() const { return z; }
};

#endif // VEC_HPP