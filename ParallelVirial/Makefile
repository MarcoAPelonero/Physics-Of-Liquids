# Use g++ for CPU-only code (graph generation mode)
CC = g++
CFLAGS = -Iinclude -std=c++17 -Wall -Wextra -Wpedantic

# Use nvcc for CUDA compilation and linking for the main simulation mode
NVCC = nvcc
NVCCFLAGS = -std=c++17 -O3 -rdc=true

# Automatically include all headers (so that changes to any header trigger a rebuild)
INCLUDE_HPP := $(wildcard include/*.hpp)

# -------------------------------
# Graph Generation Mode Settings
# -------------------------------
GRAPH_SRCS = graphGeneration.cpp src/GraphGeneration.cpp src/graphUtils.cpp src/graph.cpp
GRAPH_EXE = graphGeneration.exe

# -------------------------------
# Main Simulation Mode Settings
# -------------------------------
MAIN_SRCS = main.cpp src/graph.cpp src/potentials.cpp src/GraphGeneration.cpp src/graphUtils.cpp
CUDA_SRC = src/montecarlo.cu
CUDA_OBJ = parallelMonteCarlo.o
MAIN_EXE = main.exe

.PHONY: all clean graphgeneration main

all: graphgeneration main

# -------------------------------
# Graph Generation Target (using g++)
# -------------------------------
graphgeneration: $(GRAPH_EXE)
	@echo "Running graph generation mode..."
	./$(GRAPH_EXE)

$(GRAPH_EXE): $(GRAPH_SRCS) $(INCLUDE_HPP)
	$(CC) $(CFLAGS) $^ -o $(GRAPH_EXE)

# -------------------------------
# Main Simulation Target (using nvcc for linking)
# -------------------------------
main: $(MAIN_EXE)
	@echo "Running main simulation mode..."
	./$(MAIN_EXE)

# Compile the CUDA source with RTTI and exception support enabled
$(CUDA_OBJ): $(CUDA_SRC) $(INCLUDE_HPP)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fexceptions -Xcompiler -frtti -Iinclude -c $(CUDA_SRC) -o $(CUDA_OBJ)

# Link main simulation executable using nvcc (without the problematic flag)
$(MAIN_EXE): $(MAIN_SRCS) $(CUDA_OBJ) $(INCLUDE_HPP)
	$(NVCC) $(NVCCFLAGS) -Iinclude $(MAIN_SRCS) $(CUDA_OBJ) -o $(MAIN_EXE) -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64" -lcudart

# -------------------------------
# Clean up build artifacts
# -------------------------------
clean:
	rm -f *.exe $(CUDA_OBJ)
