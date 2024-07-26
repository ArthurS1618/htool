#include <chrono>
#include <ctime>
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/matrix/matrix.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>
#include <htool/wrappers/wrapper_lapack.hpp>
#include <iostream>

#include <random>
using namespace htool;

////////////////////////
/////// ROUTINES
// Fonction pour sauvegarder un vecteur en CSV
template <typename T>
void save_to_csv(const std::vector<T> &data, const std::string &nomFichier) {
    std::ofstream fichier(nomFichier);
    if (!fichier.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir le fichier pour écriture." << std::endl;
        return;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        fichier << data[i];
        if (i < data.size() - 1) {
            fichier << ","; // Séparateur de colonnes
        }
    }
    fichier << "\n"; // Fin de ligne

    fichier.close();
    std::cout << "Données sauvegardées dans " << nomFichier << std::endl;
}

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

///// fonction pour appliquer la perm inverse de la numerotation de htool a une matrice (j'arrive pas a me servir de l'option "NOPERM")
template <class CoefficientPrecision, class CoordinatePrecision>
Matrix<CoefficientPrecision> get_unperm_mat(const Matrix<CoefficientPrecision> mat, const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source) {
    CoefficientPrecision *ptr      = new CoefficientPrecision[target.get_size() * source.get_size()];
    const auto &target_permutation = target.get_permutation();
    const auto &source_permutation = source.get_permutation();
    for (int i = 0; i < target.get_size(); i++) {
        for (int j = 0; j < source.get_size(); j++) {
            ptr[target_permutation[i] + source_permutation[j] * target.get_size()] = mat(i, j);
        }
    }
    Matrix<CoefficientPrecision> M;
    M.assign(target.get_size(), source.get_size(), ptr, true);
    return M;
}
////// Générateur virtuelle pour manipuler des matrices en entier
template <class CoefficientPrecision, class CoordinatePrecision>
class Matricegenerator : public VirtualGenerator<CoefficientPrecision> {
  private:
    Matrix<CoefficientPrecision> mat;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;

  public:
    Matricegenerator(Matrix<CoefficientPrecision> &mat0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : mat(mat0), target(target0), source(source0) {}

    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = mat(target_permutation[i + row_offset], source_permutation[j + col_offset]);
            }
        }
    }

    Matrix<CoefficientPrecision> get_mat() { return mat; }

    Matrix<CoefficientPrecision> get_perm_mat() {
        CoefficientPrecision *ptr = new CoefficientPrecision[target.get_size() * source.get_size()];
        this->copy_submatrix(target.get_size(), target.get_size(), 0, 0, ptr);
        Matrix<CoefficientPrecision> M;
        M.assign(target.get_size(), source.get_size(), ptr, true);
        return M;
    }
    Matrix<CoefficientPrecision> get_unperm_mat() {
        CoefficientPrecision *ptr      = new CoefficientPrecision[target.get_size() * source.get_size()];
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < target.get_size(); i++) {
            for (int j = 0; j < source.get_size(); j++) {
                ptr[target_permutation[i] + source_permutation[j] * target.get_size()] = mat(i, j);
            }
        }
        Matrix<CoefficientPrecision> M;
        M.assign(target.get_size(), source.get_size(), ptr, true);
        return M;
    }
};

/// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% /////

template <typename T, typename GeneratorTestType>
std::vector<T> test_hlu(int size, htool::underlying_type<T> epsilon, htool::underlying_type<T> eta) {

    bool is_error = false;
    std::vector<double> p1(3 * size);
    std::cout << p1.size() << ',' << size << std::endl;
    create_disk(3, 0.0, size, p1.data());

    //////////////////////

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    //////////////////////
    // Clustering
    ClusterTreeBuilder<double> recursive_build_strategy_1;
    std::shared_ptr<Cluster<double>> root_cluster = std::make_shared<Cluster<double>>(recursive_build_strategy_1.create_cluster_tree(size, 3, p1.data(), 2, 2));
    Matrix<double> reference(size, size);
    // GeneratorTestDoubleSymmetric generator(3, size, size, p1, p1, *root_cluster, *root_cluster, true, true);
    GeneratorTestDoubleSymmetric generator(3, p1, p1);

    generator.copy_submatrix(size, size, 0, 0, reference.data());

    HMatrixTreeBuilder<double, double> hmatrix_tree_builder(*root_cluster, *root_cluster, epsilon, eta, 'N', 'N', -1, -1, -1);

    auto root_hmatrix = hmatrix_tree_builder.build(generator);
    auto format       = root_hmatrix.get_format();

    // htool::TestCaseSolve<T, GeneratorTestType> test_case('L', trans, n1, n2, 1, -1);

    // // HMatrix
    // HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(*test_case.root_cluster_A_output, *test_case.root_cluster_A_input, epsilon, eta, 'N', 'N', -1, -1, -1);
    // HMatrix<T, htool::underlying_type<T>> root_hmatrix = hmatrix_tree_builder_A.build(*test_case.operator_A);

    // // Matrix
    // int ni_A = test_case.root_cluster_A_input->get_size();
    // int no_A = test_case.root_cluster_A_output->get_size();
    // int ni_X = test_case.root_cluster_X_input->get_size();
    // int no_X = test_case.root_cluster_X_output->get_size();
    // Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    // test_case.operator_A->copy_submatrix(no_A, ni_A, test_case.root_cluster_A_output->get_offset(), test_case.root_cluster_A_input->get_offset(), A_dense.data());

    //////////////////////
    /// tesst hlu
    HMatrix<double, double> Lres(*root_cluster, *root_cluster);
    Lres.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
    Lres.set_eta(eta);
    Lres.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
    Lres.set_epsilon(root_hmatrix.get_epsilon());

    HMatrix<double, double> Ures(*root_cluster, *root_cluster);
    Ures.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
    Ures.set_eta(eta);
    Ures.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
    Ures.set_epsilon(root_hmatrix.get_epsilon());

    auto start = std::chrono::high_resolution_clock::now();
    HLU_fast(format, *root_cluster, &Lres, &Ures);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::vector<double> res(size);
    auto xrand  = generate_random_vector(size);
    auto yy     = reference * xrand;
    auto ytest  = Lres.solve_LU_triangular(Lres, Ures, yy);
    auto err    = norm2(xrand - ytest) / norm2(xrand);
    auto time   = duration.count();
    auto compru = Ures.get_compression();
    auto comprl = Lres.get_compression();
    std::cout
        << "________________________________________________________" << std::endl;
    std::cout << "size ----------------------------------------> " << size << std::endl;
    std::cout << "erreur solve LU :----------------------------> " << err << std::endl;
    std::cout << " duration LU factorisation : ----------------> " << duration.count() << std::endl;
    std::cout << "comrpession L et U : ------------------------> " << Lres.get_compression() << ',' << Ures.get_compression() << std::endl;
    std::vector<T> result;
    result.push_back(size * 1.0);
    result.push_back(time);
    result.push_back(err);
    result.push_back(comprl);
    result.push_back(compru);

    return result;
}
