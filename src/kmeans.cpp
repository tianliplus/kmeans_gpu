#include "kmeans.h"
#include "argparse.h"

#include <stdio.h>

int main(int argc, char **argv) {
	struct options_t opts;
	get_opts(argc, argv, &opts);
	std::cout << "kmeans args: k - " << opts.num_cluster << std::endl;
}