#include "../test_hmatrix_hmatrix_product.hpp"
#include <htool/hmatrix/hmatrix.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    bool is_error                = false;
    const int number_of_points_1 = 200;
    const int number_of_points_2 = 200;
    const int number_of_points_3 = 200;

    for (auto epsilon : {1e-14, 1e-6}) {

        // Square matrix
        is_error = is_error || test_hmatrix_hmatrix_product<double, GeneratorTestDouble>(number_of_points_1, number_of_points_2, number_of_points_3, epsilon);
    }

    MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}
