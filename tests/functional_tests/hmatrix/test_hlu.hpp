#include <chrono>
#include <ctime>
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/sum_expressions.hpp>
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
bool test_hlu(int size, htool::underlying_type<T> epsilon, htool::underlying_type<T> eta) {

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
    GeneratorTestDoubleSymmetric generator(3, size, size, p1, p1, *root_cluster, *root_cluster, true, true);
    generator.copy_submatrix(size, size, 0, 0, reference.data());

    HMatrixTreeBuilder<double, double> hmatrix_tree_builder(*root_cluster, *root_cluster, epsilon, eta, 'N', 'N', -1, -1, -1);

    auto root_hmatrix = hmatrix_tree_builder.build(generator);
    auto format       = root_hmatrix.get_format();

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
    auto end                               = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::vector<double> res(size);
    auto xrand = generate_random_vector(size);
    auto yy    = reference * xrand;
    auto ytest = Lres.solve_LU_triangular(Lres, Ures, yy);
    err        = xrand
              std::cout
          << "________________________________________________________" << std::endl;
    std::cout << "size ----------------------------------------> " << size << std::endl;
    std::cout << "erreur solve LU :----------------------------> " << norm2(xrand - ytest) / norm2(xrand) << std::endl;
    std::cout << " duration LU factorisation : ----------------> " << duration.count() << std::endl;
    std::cout << "comrpession L et U : ------------------------> " << Lres.get_compression() << ',' << Ures.get_compression() << std::endl;
    std::vector<double> res;
    res.push_back()

        return is_error;
}
