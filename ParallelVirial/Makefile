CC = g++
CFLAGS = -Iinclude -std=c++17 -Wall -Wextra -Wpedantic

NVCC = nvcc
NVCCFLAGS = -std=c++17 -O3

# Automatically include all headers (so that changes to any header trigger a rebuild)
INCLUDE_HPP := $(wildcard include/*.hpp)

# Targets for graph generation mode (graphGeneration.cpp)
GRAPH_SRCS = graphGeneration.cpp src/GraphGeneration.cpp src/graphUtils.cpp src/graph.cpp
GRAPH_EXE = graphGeneration.exe

# Targets for main simulation mode (main.cpp)
# Note: removed src/MonteCarlo.cpp as it's not present
MAIN_SRCS = main.cpp src/graph.cpp src/potentials.cpp src/GraphGeneration.cpp src/graphUtils.cpp
CUDA_SRC = src/MonteCarlo.cu
CUDA_OBJ = parallelMonteCarlo.o
MAIN_EXE = main.exe

.PHONY: all clean graphgeneration main

all: graphgeneration main

graphgeneration: $(GRAPH_EXE)
	@echo "Running graph generation mode..."
	./$(GRAPH_EXE)

$(GRAPH_EXE): $(GRAPH_SRCS) $(INCLUDE_HPP)
	$(CC) $(CFLAGS) $^ -o $(GRAPH_EXE)

main: $(MAIN_EXE)
	@echo "Running main simulation mode..."
	./$(MAIN_EXE)

$(CUDA_OBJ): $(CUDA_SRC) $(INCLUDE_HPP)
	$(NVCC) $(NVCCFLAGS) -Iinclude -c $(CUDA_SRC) -o $(CUDA_OBJ)

$(MAIN_EXE): $(MAIN_SRCS) $(CUDA_OBJ) $(INCLUDE_HPP)
	$(CC) $(CFLAGS) $(MAIN_SRCS) $(CUDA_OBJ) -o $(MAIN_EXE) -LC:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.8/lib/x64 -lcudart


clean:
	rm -f *.exe $(CUDA_OBJ)