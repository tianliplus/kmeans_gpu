CC = nvcc 
SRCS = ./src/*.cpp ./src/*.cu
INC = ./src/
OPTS =  -std=c++11 -O2

EXEC = bin/kmeans

all: clean compile

compile:
	$(CC) $(SRCS) $(OPTS) -I$(INC) -o $(EXEC)

clean:
	rm -f $(EXEC)
