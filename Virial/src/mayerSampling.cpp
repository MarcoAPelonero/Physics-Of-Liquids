#include "mayerSampling.hpp"

Configuration::Configuration(int dimension, int numFreeNodes, double sigma)
    : dimension(dimension),
      numFreeNodes(numFreeNodes),
      sigma(sigma),
      sideLength(5.0 * sigma), // Example choice
      rng(123456789ULL)
{
    // We store only (n-1) node positions, each in 'dimension' coordinates.
    positions.resize(numFreeNodes, std::vector<double>(dimension, 0.0));
}

void Configuration::initialRandom() {
    std::uniform_real_distribution<double> dist(-sideLength / 2.0, sideLength / 2.0);
    for (auto &coords : positions) {
        for (int d = 0; d < dimension; ++d) {
            coords[d] = dist(rng);
        }
    }
}

void Configuration::initialLattice() {
    int gridPoints = std::ceil(std::pow(numFreeNodes, 1.0 / dimension));
    double spacing = sideLength / gridPoints;
    int count = 0;

    // Place these (n-1) free nodes on a cubic/square/line lattice.
    if (dimension == 1) {
        for (int i = 0; i < gridPoints && count < numFreeNodes; ++i) {
            positions[count][0] = -sideLength / 2 + spacing / 2 + i * spacing;
            count++;
        }
    } 
    else if (dimension == 2) {
        for (int i = 0; i < gridPoints && count < numFreeNodes; ++i) {
            for (int j = 0; j < gridPoints && count < numFreeNodes; ++j) {
                positions[count][0] = -sideLength / 2 + spacing / 2 + i * spacing;
                positions[count][1] = -sideLength / 2 + spacing / 2 + j * spacing;
                count++;
            }
        }
    } 
    else if (dimension == 3) {
        for (int i = 0; i < gridPoints && count < numFreeNodes; ++i) {
            for (int j = 0; j < gridPoints && count < numFreeNodes; ++j) {
                for (int k = 0; k < gridPoints && count < numFreeNodes; ++k) {
                    positions[count][0] = -sideLength / 2 + spacing / 2 + i * spacing;
                    positions[count][1] = -sideLength / 2 + spacing / 2 + j * spacing;
                    positions[count][2] = -sideLength / 2 + spacing / 2 + k * spacing;
                    count++;
                }
            }
        }
    }
}

const std::vector<std::vector<double>> &Configuration::getPositions() const {
    return positions;
}

std::tuple<
    std::function<double(const std::vector<double>&)>, // integrand for "full" potential
    std::function<double(const std::vector<double>&)>, // integrand for "reference" potential
    Configuration                                      // a newly created config
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

    config.initialLattice();

    auto integrandFull = graphToIntegrand(
        graph,
        potential,
        sigma,
        epsilon,
        dimension,
        beta
    );

    auto integrandRef = graphToIntegrand(
        graph,
        referencePotential,
        sigma,
        epsilon,
        dimension,
        beta
    );

    return std::make_tuple(integrandFull, integrandRef, config);
}

