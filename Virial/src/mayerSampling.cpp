#include "mayerSampling.hpp"

Configuration::Configuration(int dimension, int numFreeNodes, double sigma)
    : dimension(dimension),
      numFreeNodes(numFreeNodes),
      sigma(sigma),
      sideLength(5 * sigma), // Example choice
      rng(123456789ULL)
{
    // positions has length = (n-1) * dimension
    positions.resize(numFreeNodes * dimension, 0.0);
}

void Configuration::initialRandom() {
    std::uniform_real_distribution<double> dist(-sideLength / 2.0, sideLength / 2.0);
    // Fill each coordinate
    for (int i = 0; i < numFreeNodes; ++i) {
        for (int d = 0; d < dimension; ++d) {
            positions[i * dimension + d] = dist(rng);
        }
    }
}

void Configuration::initialLattice() {
    int gridPoints = std::ceil(std::pow(numFreeNodes, 1.0 / dimension));
    double spacing = sideLength / gridPoints;
    int count = 0;

    // Place these (n-1) free nodes on a 1D/2D/3D lattice, depending on dimension.
    if (dimension == 1) {
        for (int i = 0; i < gridPoints && count < numFreeNodes; ++i) {
            positions[count * dimension + 0] = -sideLength / 2 + spacing / 2 + i * spacing;
            count++;
        }
    }
    else if (dimension == 2) {
        for (int i = 0; i < gridPoints && count < numFreeNodes; ++i) {
            for (int j = 0; j < gridPoints && count < numFreeNodes; ++j) {
                positions[count * dimension + 0] = -sideLength / 2 + spacing / 2 + i * spacing;
                positions[count * dimension + 1] = -sideLength / 2 + spacing / 2 + j * spacing;
                count++;
            }
        }
    }
    else if (dimension == 3) {
        for (int i = 0; i < gridPoints && count < numFreeNodes; ++i) {
            for (int j = 0; j < gridPoints && count < numFreeNodes; ++j) {
                for (int k = 0; k < gridPoints && count < numFreeNodes; ++k) {
                    positions[count * dimension + 0] = -sideLength / 2 + spacing / 2 + i * spacing;
                    positions[count * dimension + 1] = -sideLength / 2 + spacing / 2 + j * spacing;
                    positions[count * dimension + 2] = -sideLength / 2 + spacing / 2 + k * spacing;
                    count++;
                }
            }
        }
    }
}

void Configuration::moveRandomParticle(double delta) {
    // Choose a random particle to move.
    std::uniform_int_distribution<int> intDist(0, numFreeNodes - 1);
    int particle = intDist(rng);

    // Generate a random number in [-1, 1] for each dimension.
    std::uniform_real_distribution<double> realDist(-1.0, 1.0);
    double halfSide = sideLength / 2.0;

    for (int d = 0; d < dimension; ++d) {
        int idx = particle * dimension + d;
        // Propose a new position.
        double newPos = positions[idx] + delta * realDist(rng);

        // Apply periodic boundary conditions by wrapping the coordinate.
        while (newPos > halfSide) {
            newPos -= sideLength;
        }
        while (newPos < -halfSide) {
            newPos += sideLength;
        }
        positions[idx] = newPos;
    }
}

double Configuration::computeIntegrandOnConfiguration(
    const std::function<double(const std::vector<double>&)> &integrand
) const {
    return integrand(positions);
}


double Configuration::getSideLength() const {
    return sideLength;
}

void Configuration::setPositions(std::vector<double> &newPositions) {
    positions = newPositions;
}

const std::vector<double> &Configuration::getPositions() const {
    return positions;
}

void Configuration::printConfiguration() const {
    for (int i = 0; i < numFreeNodes; ++i) {
        std::cout << "Particle " << i << ": ";
        for (int d = 0; d < dimension; ++d) {
            std::cout << positions[i * dimension + d] << " ";
        }
        std::cout << std::endl;
    }
}

void Configuration::printConfiguration(std::ofstream &outFile) const {
    for (int i = 0; i < numFreeNodes; ++i) {
        outFile << "Particle " << i << ": ";
        for (int d = 0; d < dimension; ++d) {
            outFile << positions[i * dimension + d] << " ";
        }
        outFile << std::endl;
    }
}

std::tuple<
    std::function<double(const std::vector<double>&)>, 
    std::function<double(const std::vector<double>&)>, 
    Configuration
>
createIntegrandsAndConfig(const NDGraph &graph,
                          PotentialFunction potential,
                          PotentialFunction referencePotential,
                          double sigma,
                          double epsilon,
                          int dimension,
                          double beta)
{
    // Number of nodes minus one for the fixed particle (node 0).
    int numFreeNodes = graph.getNumNodes() - 1;

    // Create a configuration for (n-1) particles in 'dimension' dimensions.
    Configuration config(dimension, numFreeNodes, sigma);

    // For instance, initialize on a lattice
    config.initialRandom();

    // The integrand for the full potential
    auto integrandFull = graphToIntegrand(
        graph, potential, sigma, epsilon, dimension, beta, config.getSideLength()
    );

    // The integrand for the reference potential
    auto integrandRef = graphToIntegrand(
        graph, referencePotential, sigma, epsilon, dimension, beta, config.getSideLength()
    );

    return std::make_tuple(integrandFull, integrandRef, config);
}

HardSpheresCoefficients::HardSpheresCoefficients() {
    coefficients.resize(10, 0.0);
    coefficients[0] = 0.0; 
    coefficients[1] = 0.0; 
    coefficients[2] = 4.0; 
    coefficients[3] = 10.0;
    coefficients[4] = 18.365;
    coefficients[5] = 28.244;
    coefficients[6] = 39.82;
    coefficients[7] = 53.34;
    coefficients[8] = 68.54;
}

void HardSpheresCoefficients::changeForm(double sigma) {
    double v0 = (M_PI / 6.0) * sigma * sigma * sigma;
    for (int i = 0; i < 10; ++i) {
        coefficients[i] *= pow(v0, i - 1);
    }
}

void HardSpheresCoefficients::getGraphIntegral(double sigma) {
    changeForm(sigma);

    // Factor should be -(n-1)/n!
    for (int i = 0; i < 10; ++i) {
        if (i < 2) {
            continue;
        }
        double factor = -(i-1) / ( std::tgamma(i+1) );
        coefficients[i] /= factor;
    }
}

double HardSpheresCoefficients::operator[](int i) const {
    return coefficients[i];
}