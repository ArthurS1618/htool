#include "../test_hlu.hpp"
#include <htool/hmatrix/hmatrix.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    bool is_error = false;

    // MPI_Init(&argc, &argv);
    if (argc != 5) {
        std::cerr << " PARAMETERS : SIZEMAX, nb iteration, EPSILON ETA" << std::endl;
    }
    int size_max   = std::stoi(argv[1]);
    int nb         = std::stoi(argv[2]);
    double epsilon = std::stod(argv[3]);
    double eta     = std::stod(argv[4]);

    std::vector<double> sizes;
    std::vector<double> time;
    std::vector<double> err;
    std::vector<double> comprl;
    std::vector<double> compru;
    for (int k = 1; k < nb; ++k) {
        int size = size_max * k / (nb - 1);
        // is_error = is_error || test_hlu<double, GeneratorTestDoubleSymmetric>(size, epsilon, eta);
        std::vector<double> data = test_hlu<double, GeneratorTestDoubleSymmetric>(size, epsilon, eta);
        sizes.push_back(data[0]);
        time.push_back(data[1]);
        err.push_back(data[2]);
        comprl.push_back(data[3]);
        compru.push_back(data[4]);
        std::cout << "-------------->" << data[0] << ',' << data[1] << ',' << data[2] << ',' << data[3] << ',' << data[4] << std::endl;
    }
    save_to_csv(sizes, "size_test_hlu_eps6_eta_10.csv");
    save_to_csv(time, "time_test_hlu_eps6_eta_10.csv");
    save_to_csv(err, "err_test_hlu_eps6_eta_10.csv");
    save_to_csv(comprl, "comprl_test_hlu_eps6_eta_10.csv");
    save_to_csv(compru, "compru_test_hlu_eps6_eta_10.csv");
    // MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}