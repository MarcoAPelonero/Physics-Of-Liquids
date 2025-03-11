CC = g++
CFLAGS = -Iinclude -std=c++11 -Wall

# Automatically collect all .cpp files in src folder
SRC_FILES = $(wildcard src/*.cpp)

# For the main target, compile main.cpp plus all files from src
SRCS = main.cpp $(SRC_FILES)
TARGET = main.exe

# For the test target, compile tests.cpp plus all files from src
TEST_SRC = tests.cpp $(SRC_FILES)
TEST_TARGET = tests.exe

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRC)
	$(CC) $(CFLAGS) $(TEST_SRC) -o $(TEST_TARGET)

ifeq ($(OS),Windows_NT)
RM = del /F
else
RM = rm -f
endif

clean:
	$(RM) *.exe