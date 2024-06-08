
#include "../test_sum_expression.hpp"
#include <htool/hmatrix/hmatrix.hpp>

int main(int argc, char *argv[]) {
    // int size       = std::stoi(argv[1]);
    // int rank       = std::stoi(argv[2]);
    // double epsilon = std::stod(argv[3]);
    int size       = 4000;
    int rank       = 100;
    double epsilon = 1e-6;
    auto data      = test_sum_expression<double>(size, rank, epsilon);

    return 0;
}