
#include "../test_sum_expression.hpp"
#include "../../../../include/htool/hmatrix/hmatrix.hpp"

int main(int argc, char *argv[]) {
    // int rankWorld;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    // MPI_Init(&argc, &argv);
    // int size = std::stoi(argv[1]);
    int size = 5000;
    // int rank       = std::stoi(argv[2]);
    // double epsilon = std::stod(argv[3]);
    // int size       = 4000;
    int rank       = 100;
    double epsilon = 1e-6;
    int data       = test_sum_expression<double>(size, rank, epsilon);
    // MPI_Finalize();

    return 0;
}