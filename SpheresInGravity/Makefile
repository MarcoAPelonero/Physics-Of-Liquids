# Directories
SRC_DIR   := src
INC_DIR   := include
BUILD_DIR := build

CXX      := g++
CXXFLAGS := -I$(INC_DIR) -Wall -std=c++17

SOURCES  := main.cpp $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS  := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(notdir $(SOURCES)))
TARGET   := main.exe

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)