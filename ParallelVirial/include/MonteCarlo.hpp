#ifndef MONTE_CARLO_INTEGRATION_HPP
#define MONTE_CARLO_INTEGRATION_HPP

// Runs Monte Carlo integration on the GPU for a given graph.
// h_edgeFrom and h_edgeTo are host arrays for the graph’s edges (length = numEdges).
// Returns the estimated integral.
double runMonteCarloIntegration(int dimension,
                                int nFreeNodes,
                                double sigma,
                                double epsilon,
                                double beta,
                                long nSamples,
                                double sideLength,
                                const int* h_edgeFrom,
                                const int* h_edgeTo,
                                int numEdges);

#endif // MONTE_CARLO_INTEGRATION_HPP
