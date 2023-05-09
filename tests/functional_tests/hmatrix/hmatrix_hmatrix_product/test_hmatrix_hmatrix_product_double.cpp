#include "../test_hmatrix_hmatrix_product.hpp"
#include <htool/hmatrix/hmatrix.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    bool is_error = false;

    MPI_Init(&argc, &argv);
    for (int k = 3; k < 4; ++k) {
        std::cout << " size = " << std::pow(2, 8 + k) << std::endl;
        const int number_of_points_1 = std::pow(2, 8 + k);
        // const int number_of_points_1 = 30;

        const int number_of_points_2 = std::pow(2, 8 + k);
        const int number_of_points_3 = std::pow(2, 8 + k);
        // const int number_of_points_1 = 10000;
        // const int number_of_points_2 = 10000;
        // const int number_of_points_3 = 10000;

        std::vector<double> epsilon = {1e-5};

        for (auto i : epsilon) {

            // Square matrix
            is_error = is_error || test_hmatrix_hmatrix_product<double, GeneratorTestDouble>(number_of_points_1, number_of_points_2, number_of_points_3, i);
        }
    }
    MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}
