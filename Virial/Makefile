CC = g++
CFLAGS = -Iinclude -std=c++17 -Wall -Wextra -Wpedantic

SRC_FILES = $(wildcard src/*.cpp)

SRCS = main.cpp $(SRC_FILES)
TARGET = main.exe

TEST_SRC = tests.cpp $(SRC_FILES)
TEST_TARGET = tests.exe

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

runtest:
ifndef TEST
	$(error Please specify a test file with TEST=<filename> (without .cpp extension))
endif
	$(CC) $(CFLAGS) tests/$(TEST).cpp $(SRC_FILES) -o $(TEST).exe
	./$(TEST).exe
	$(RM) $(TEST).exe

$(TEST_TARGET): $(TEST_SRC)
	$(CC) $(CFLAGS) $(TEST_SRC) -o $(TEST_TARGET)

runsim:
ifndef SIM
	$(error Please specify a simulation name with SIM=<name> for runsim)
endif
	$(CC) $(CFLAGS) $(SIM)Simulation.cpp $(SRC_FILES) -o $(SIM)Simulation.exe
	./$(SIM)Simulation.exe

ifeq ($(findstring bash,$(SHELL)),bash)
RM = rm -f
else
RM = rm /F
endif

clean:
	$(RM) *.exe