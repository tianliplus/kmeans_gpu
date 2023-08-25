#pragma once

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {
	int num_cluster;
	int dims;
	char *inputfilename;
	int max_num_iter;
	double threshold;
	bool centroids;
	int seed;
};

void get_opts(int argc, char **argv, struct options_t *opts);
