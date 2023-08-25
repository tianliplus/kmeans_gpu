CC = nvcc 
SRCS = ./src/*.cpp
INC = ./src/
OPTS = -O2

EXEC = bin/kmeans

all: clean compile

compile:
	$(CC) $(SRCS) $(OPTS) -I$(INC) -o $(EXEC)

clean:
	rm -f $(EXEC)
