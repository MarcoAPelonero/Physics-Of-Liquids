#include "MonteCarlo.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// Device structure to hold graph data.
struct GraphDeviceData {
    int numEdges;
    int* edgeFrom;
    int* edgeTo;
};

// Device function: Lennard-Jones potential (with safeguard for r==0).
__device__ double potentialFunction(double r, double sigma, double epsilon) {
    if (r == 0.0) return 1e12;
    double sr = sigma / r;
    double sr6 = pow(sr, 6);
    double sr12 = sr6 * sr6;
    return 4.0 * epsilon * (sr12 - sr6);
}

// Device function: Mayer f-function = exp(-beta*U) - 1.
__device__ double mayerF(double r, double sigma, double epsilon, double beta) {
    return exp(-beta * potentialFunction(r, sigma, epsilon)) - 1.0;
}

// Device function: Computes the distance with node 0 fixed at the origin.
// 'coords' has length (dimension*(nNodes-1)); for node i > 0, its coordinates
// are stored starting at index (i-1)*dimension.
__device__ double distanceFixedNode0(const double* coords, int i, int j, int dimension, double sideLength) {
    double dist2 = 0.0;
    for (int d = 0; d < dimension; ++d) {
        double xi = (i == 0) ? 0.0 : coords[(i - 1) * dimension + d];
        double xj = (j == 0) ? 0.0 : coords[(j - 1) * dimension + d];
        double diff = xi - xj;
        diff = diff - sideLength * round(diff / sideLength);
        dist2 += diff * diff;
    }
    return sqrt(dist2);
}

// Device function: Evaluates the integrand for one sample given the graph.
__device__ double deviceIntegrandGeneric(const double* coords,
                                          const GraphDeviceData& graphData,
                                          int dimension,
                                          double sideLength,
                                          double sigma,
                                          double epsilon,
                                          double beta) {
    double product = 1.0;
    for (int e = 0; e < graphData.numEdges; ++e) {
        int i = graphData.edgeFrom[e];
        int j = graphData.edgeTo[e];
        double r = distanceFixedNode0(coords, i, j, dimension, sideLength);
        double f = mayerF(r, sigma, epsilon, beta);
        product *= f;
    }
    return product;
}

// CUDA kernel: each thread generates several samples, evaluates the integrand,
// and accumulates a partial sum.
__global__ void monteCarloKernel(double *partialSums,
                                 int dimension,
                                 int nFreeNodes,
                                 double sigma,
                                 double epsilon,
                                 double beta,
                                 long samplesPerThread,
                                 double sideLength,
                                 GraphDeviceData graphData) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCoords = dimension * nFreeNodes;
    curandState state;
    curand_init(1234, idx, 0, &state);
    
    double partialSum = 0.0;
    // Assume totalCoords <= 64; adjust if necessary.
    double coords[64];
    
    for (long i = 0; i < samplesPerThread; i++) {
        for (int j = 0; j < totalCoords; j++) {
            double u = curand_uniform_double(&state);
            coords[j] = -sideLength / 2.0 + sideLength * u;
        }
        double fVal = deviceIntegrandGeneric(coords, graphData, dimension, sideLength, sigma, epsilon, beta);
        partialSum += fVal;
    }
    partialSums[idx] = partialSum;
}

// Host function that wraps the CUDA kernel launch and reduction.
double runMonteCarloIntegration(int dimension,
                                int nFreeNodes,
                                double sigma,
                                double epsilon,
                                double beta,
                                long nSamples,
                                double sideLength,
                                const int* h_edgeFrom,
                                const int* h_edgeTo,
                                int numEdges) {
    int threadsPerBlock = 256;
    int numBlocks = 256;
    int totalThreads = threadsPerBlock * numBlocks;
    long samplesPerThread = nSamples / totalThreads;
    if (samplesPerThread == 0) {
        std::cerr << "nSamples too low for number of threads." << std::endl;
        return 0.0;
    }
    int totalCoords = dimension * nFreeNodes;
    double volume = pow(sideLength, totalCoords);

    // Allocate device memory for edge arrays.
    int *d_edgeFrom, *d_edgeTo;
    cudaMalloc((void**)&d_edgeFrom, numEdges * sizeof(int));
    cudaMalloc((void**)&d_edgeTo, numEdges * sizeof(int));
    cudaMemcpy(d_edgeFrom, h_edgeFrom, numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeTo, h_edgeTo, numEdges * sizeof(int), cudaMemcpyHostToDevice);

    GraphDeviceData graphData;
    graphData.numEdges = numEdges;
    graphData.edgeFrom = d_edgeFrom;
    graphData.edgeTo = d_edgeTo;

    double *d_partialSums;
    cudaMalloc((void**)&d_partialSums, totalThreads * sizeof(double));

    monteCarloKernel<<<numBlocks, threadsPerBlock>>>(d_partialSums,
                                                     dimension,
                                                     nFreeNodes,
                                                     sigma,
                                                     epsilon,
                                                     beta,
                                                     samplesPerThread,
                                                     sideLength,
                                                     graphData);
    cudaDeviceSynchronize();

    double *h_partialSums = new double[totalThreads];
    cudaMemcpy(h_partialSums, d_partialSums, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);

    double totalSum = 0.0;
    for (int i = 0; i < totalThreads; i++) {
        totalSum += h_partialSums[i];
    }
    double avg = totalSum / (double)(samplesPerThread * totalThreads);
    double integralEstimate = volume * avg;

    delete[] h_partialSums;
    cudaFree(d_partialSums);
    cudaFree(d_edgeFrom);
    cudaFree(d_edgeTo);

    return integralEstimate;
}