#ifndef GRAPH_GENERATION_HPP
#define GRAPH_GENERATION_HPP

#include <string>
#include <vector>
#include "graph.hpp"       // Your NDGraph and Edge definitions
#include "graphUtils.hpp"  // Your graph generation and degeneracy functions

// Generates biconnected graphs up to maxOrder and saves them to a file.
// The file format (simple text) is as follows:
//   First line: maxOrder
//   For each order (n) from 2 to maxOrder:
//     Line: <order> <number_of_graphs>
//     Then for each graph:
//       Line: <nNodes> <nEdges> <degeneracy>
//       Then one line per edge: <from> <to>
void generateAndSaveGraphs(int maxOrder, const std::string &filename);

#endif // GRAPH_GENERATION_HPP
