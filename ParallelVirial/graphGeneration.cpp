#include <fstream>
#include "GraphGeneration.hpp"

int main(int argc, char **argv) {
    int maxOrder = 5;
    std::string filename = "graphs.dat";
    if (argc > 1) {
        maxOrder = std::atoi(argv[1]);
    }
    if (argc > 2) {
        filename = argv[2];
    }
    generateAndSaveGraphs(maxOrder, filename);
    return 0;
}