CC = g++
CFLAGS = -Iinclude -std=c++11 -Wall
SRCS = main.cpp src/graph.cpp
TARGET = main

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
