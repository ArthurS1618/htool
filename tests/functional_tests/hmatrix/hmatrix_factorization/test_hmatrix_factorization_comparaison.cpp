#include "../test_hmatrix_factorization_comparaison.hpp"
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/testing/generate_test_case.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>

using namespace std;
using namespace htool;

void save_vectors_to_csv(const std::vector<double> &x, const std::string &filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        for (size_t i = 0; i < x.size() - 1; ++i) {
            file << x[i] << ",";
        }
        file << x[x.size() - 1];
        file.close();
        std::cout << "Data saved to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}
int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    if (argc != 6) {
        std::cerr << " PARAMETERS : SIZEMAX, nb iteration, EPSILON ETA , save " << std::endl;
    }
    int size_max   = std::stoi(argv[1]);
    int nb         = std::stoi(argv[2]);
    double epsilon = std::stod(argv[3]);
    double eta     = std::stod(argv[4]);
    int save       = std::stoi(argv[5]);
    std::vector<double> sizes(nb);
    std::vector<double> time(nb);
    std::vector<double> compr(nb);
    std::vector<double> error(nb);
    for (int k = 1; k < nb + 1; ++k) {
        int size     = size_max * k / nb;
        auto data    = test_hlu<double, GeneratorTestDoubleSymmetric>(size, epsilon, eta);
        sizes[k - 1] = data[0];
        time[k - 1]  = data[1];
        compr[k - 1] = data[2];
        error[k - 1] = data[3];
    }
    if (save == 1) {
        save_vectors_to_csv(sizes, "Size_Pierre_big_eps3_eta10.csv");
        save_vectors_to_csv(time, "Time_Pierre_big_eps3_eta10.csv");
        save_vectors_to_csv(compr, "Compr_Pierre_big_eps3_eta10.csv");
        save_vectors_to_csv(error, "Error_Pierre_big_eps3_eta10.csv");
    }
    MPI_Finalize();
    return 0;
}
