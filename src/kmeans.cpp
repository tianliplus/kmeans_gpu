#include "kmeans.h"
#include "argparse.h"

#include <stdio.h>
#include <fstream>

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}

int main(int argc, char **argv) {
	struct options_t opts;
	get_opts(argc, argv, &opts);

	// Read and parse file
	std::ifstream infile(opts.inputfilename);
	std::string item;
	infile >> item;
	int total_points = stoi(item);
	double *points[total_points];
	for (int i  = 0; i < total_points; i++) {
		points[i] = (double*)malloc(sizeof(double) * opts.dims);
		// discard point id
		infile >> item;
		for (int j = 0; j < opts.dims; j++) {
			infile >> item;
			*points[i][j] = stod(item);
		}
	}
	std::cout << "total points: " << total_points << std::endl;
	
	// Init centers
	double *centers[opts.num_cluster];
	kmeans_srand(opts.seed);
	for (int i = 0; i < opts.num_cluster; i++){
	    int index = kmeans_rand() % total_points;
	    centers[i] = points[index];
	    std::cout << "center point: " << index << std::endl;
		for (int j = 0; j < opts.dims; j++) {
			std::cout << *centers[i][j] << " ";
		}
		std::cout << std::endl;
	}

	if (opts.implement_type == "cpu") {
		std::cout << "todo" << std::endl;
	} else {
		std::cout << "Unsupported implement_type: " << opts.implement_type << std::endl;
	}

	for (int i = 0; i < total_points; i++) {
		free(points[i]);
	}
	free(points);
}

void kmeans_cpu() {
	// TODO
}
