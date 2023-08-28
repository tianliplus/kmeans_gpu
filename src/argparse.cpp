#include "argparse.h"

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--num_cluster or -k <num_cluster>" << std::endl;
        std::cout << "\t--dims or -d <dims>" << std::endl;
        std::cout << "\t--inputfilename or -i <inputfilename>" << std::endl;
        std::cout << "\t--max_num_iter or -m <max_num_iter>" << std::endl;
        std::cout << "\t--threshold or -t <threshold>" << std::endl;
        std::cout << "\t--seed or -s <seed>" << std::endl;
        std::cout << "\t--implement_type or -x <implement_type> (cpu / cuda / thrust)" << std::endl;
        std::cout << "\t[Optional] --centroids or -c" << std::endl;
        std::cout << "\t[Optional] --unchanged_converge or -u" << std::endl;
        exit(0);
    }

    opts->centroids = false;
    opts->unchanged_converge = false;

    struct option l_opts[] = {
        {"num_cluster", required_argument, NULL, 'k'},
        {"dims", required_argument, NULL, 'd'},
        {"inputfilename", required_argument, NULL, 'i'},
        {"max_num_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"seed", required_argument, NULL, 's'},
        {"implement_type", required_argument, NULL, 'x'},
        {"centroids", no_argument, NULL, 'c'},
        {"unchanged_converge", no_argument, NULL, 'u'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:x:cus:", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'k':
            opts->num_cluster = atoi((char *)optarg);
            break;
        case 'd':
            opts->dims = atoi((char *)optarg);
            break;
        case 'i':
            opts->inputfilename = (char *)optarg;
            break;
        case 'm':
            opts->max_num_iter = atoi((char *)optarg);
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 'c':
            opts->centroids = true;
            break;
        case 's':
            opts->seed = atoi((char *)optarg);
            break;
        case 'x':
            opts->implement_type = (char *)optarg;
            break;
        case 'u':
            opts->unchanged_converge = true;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}
