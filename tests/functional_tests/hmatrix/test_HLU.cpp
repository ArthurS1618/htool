#include <chrono>
#include <ctime>
// #include <htool/basic_types/matrix.hpp>
// #include <htool/clustering/clustering.hpp>
// #include <htool/hmatrix/hmatrix.hpp>
// #include <htool/hmatrix/hmatrix_output.hpp>
// #include <htool/hmatrix/sum_expressions.hpp>
// #include <htool/hmatrix/tree_builder/tree_builder.hpp>
// #include <htool/testing/generator_input.hpp>
// #include <htool/testing/generator_test.hpp>
// #include <htool/testing/geometry.hpp>
// #include <htool/testing/partition.hpp>
// #include <htool/wrappers/wrapper_lapack.hpp>
#include <iostream>

#include <random>
using namespace htool;
std::vector<double> generate_random_vector(int size) {
    std::random_device rd;  // Source d'entropie aléatoire
    std::mt19937 gen(rd()); // Générateur de nombres pseudo-aléatoires

    std::uniform_real_distribution<double> dis(1.0, 10.0); // Plage de valeurs pour les nombres aléatoires (ici de 1.0 à 10.0)

    std::vector<double> random_vector;
    random_vector.reserve(size); // Allocation de mémoire pour le vecteur

    for (int i = 0; i < size; ++i) {
        random_vector.push_back(dis(gen)); // Ajout d'un nombre aléatoire dans la plage à chaque itération
    }

    return random_vector;
}

void forward_substitution(const Matrix<double> &L, const std::vector<double> &y, std::vector<double> &x) {
    int n = L.nb_cols();

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L(i, j) * x[j];
        }
        x[i] = (y[i] - sum);
    }
}
void backward_substitution(const Matrix<double> &U, const std::vector<double> &y, std::vector<double> &x) {
    int n = U.nb_cols();

    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += U(i, j) * x[j];
        }
        x[i] = (y[i] - sum) / U(i, i);
    }
}

int main() {
    std::cout << "hello world" << std::endl;
}