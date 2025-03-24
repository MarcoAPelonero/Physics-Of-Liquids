# Compiler and flags for C++.
CC = g++
CFLAGS = -Iinclude -std=c++17 -Wall -Wextra -Wpedantic

# Compiler and flags for CUDA.
NVCC = nvcc
NVCCFLAGS = -std=c++17 -O3

# Source files.
SRC_FILES = $(wildcard src/*.cpp)
CUDA_SRC = src/parallelMonteCarlo.cu

# Targets.
SRCS = main.cpp $(SRC_FILES)
TARGET = main.exe

# Object file for the CUDA source.
CUDA_OBJ = parallelMonteCarlo.o

all: $(TARGET)

# Compile CUDA source.
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) -c $(CUDA_SRC) -o $(CUDA_OBJ)

# Link final executable.
$(TARGET): $(SRCS) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(SRCS) $(CUDA_OBJ) -o $(TARGET) -L/usr/local/cuda/lib64 -lcudart

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f *.exe $(CUDA_OBJ)
