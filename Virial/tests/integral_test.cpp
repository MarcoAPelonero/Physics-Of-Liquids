#include <iostream>
#include <cmath>
#include "graph.hpp"
#include "graphUtils.hpp"
#include "graphToIntegrand.hpp"
#include "MonteCarlo.hpp"
#include "potentials.hpp"

int main()
{
    // Graph with 2 nodes, 1 edge:
    NDGraph myGraph(3,false);
    myGraph.addEdge(0,1); 
    myGraph.addEdge(1,2);
    myGraph.addEdge(2,0);

    int nNodes = myGraph.getNumNodes();

    // Hard-sphere potential. Sigma is DIAMETER
    PotentialFunction potHS = [](double r, double sigma, double epsilon) {
        // If r < sigma => "infinite" => exp(-U)=0 => f(r)=-1
        // else f(r)=0
        return HS(r, sigma, epsilon);
    };

    // Create integrand: dimension=3, node #0 pinned at origin
    double sigma   = 1.0;  // diameter
    double epsilon = 1.0;  // not used in HS
    int dimension  = 3;

    auto integrand = graphToIntegrand(myGraph, potHS, sigma, epsilon, dimension);

    // Monte Carlo with node #0 fixed => integrate over (nNodes-1)=1 in 3D => 3 coords
    long nSamples = 1000000; 
    double estimate = monteCarloHitOrMiss(
        integrand,
        dimension,
        nNodes - 1,    // because node #0 is pinned
        sigma,
        nSamples
    );

    // Multiply by graph degeneracy:
    double deg = GraphUtils::computeDegeneracy(myGraph);
    double graphValue = estimate * deg;

    std::cout << "Raw cluster integral (node0 fixed) * degeneracy = "
              << graphValue << std::endl;

    // If you want the standard factor for e.g. a virial expansion term:
    // B_n = (-1)^(n-1)/ n! * (1/Volume) * sum_over_biconnected( graphValue ).
    // For n=2 in some volume V, that factor might look like:
    int n = nNodes;
    double factor = -(n-1) / ( std::tgamma(n+1) );
    // If you want 1/V => define "Volume = side^3" or your box, etc.
    double sideLength = 2.5 * sigma; 
    double physicalVolume = sideLength*sideLength*sideLength; 
    // double finalContribution = factor * (1.0/physicalVolume) * graphValue;
    double finalContribution = factor * graphValue;

    std::cout << "With factor = [(-1)^(n-1)/n!] * (1/V), value = "
              << finalContribution << std::endl;

    return 0;
}
