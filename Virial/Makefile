CXX = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp
INCLUDES = -Iinclude

# Only make.cpp needs to be compiled
SOURCES = make.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = make  # Change this if you want a different executable name

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)

run: $(TARGET)
	./$(TARGET)
