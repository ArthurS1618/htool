#include "../test_hlu.hpp"
#include <htool/hmatrix/hmatrix.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    bool is_error = false;

    MPI_Init(&argc, &argv);
    if (argc = !5) {
        std::cerr << " PARAMETERS : SIZEMAX, nb iteration, EPSILON ETA" << std::endl;
    }
    int size_max   = std::stoi(argv[1]);
    int nb         = std::stoi(argv[2]);
    double epsilon = std::stod(argv[3]);
    double eta     = std::stod(argv[4]);

    for (int k = 1; k < nb; ++k) {
        int size = size_max * k / (nb - 1);
        is_error = is_error || test_hlu<double, GeneratorDouble>(size, epsilon, eta);
    }
    MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}