#include <ctime>
#include <htool/basic_types/matrix.hpp>
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/sum_expressions.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>
#include <htool/wrappers/wrapper_lapack.hpp>
#include <iostream>

#include <random>
using namespace htool;
Matrix<double> poisson_matrix(int n) {
    int m    = n - 2;
    int size = m * m;
    Matrix<double> A(size, size);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            int k = i * m + j;

            if (i == 0 || i == m - 1 || j == 0 || j == m - 1) {
                A(k, k) = 1.0;
            } else {
                A(k, k)     = -4.0;
                A(k, k - 1) = 1.0;
                A(k, k + 1) = 1.0;
                A(k, k - m) = 1.0;
                A(k, k + m) = 1.0;
            }
        }
    }

    return A * (1 / ((n - 1) * (n - 1)));
}
// int main() {
//     int size = 100;
//     std::cout << "______________________________________" << std::endl;
//     Matrix<double> poisson = poisson_matrix(size);
//     std::cout << "assemblage ok" << std::endl;
//     auto temp = LU(poisson);
//     auto L    = get_lu(temp).first;
//     auto U    = get_lu(temp).second;
//     std::cout << "erreur M-Lu " << normFrob(L * U - poisson) / normFrob(poisson) << std::endl;
// }
int main() {
    std::cout << "_________________________________________" << std::endl;
    int size = 100;
    Matrix<double> seed(size, size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            seed(i, j) = dist(gen);
        }
    }

    std::vector<int> ipiv(size);
    int info;
    Lapack<double>::getrf(&size, &size, seed.data(), &size, ipiv.data(), 0);
}