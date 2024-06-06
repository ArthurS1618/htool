#include <chrono>
#include <ctime>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/linalg/interface.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/matrix/linalg/interface.hpp>
#include <htool/testing/generate_test_case.hpp>
#include <htool/testing/generator_input.hpp>

using namespace std;
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

// //// Pour extraire Lh et Uh de lu_factorisation() on luyiu donne L ou U
// template <class CoefficientPrecision, class CoordinatePrecision>
// HMatrix<CoefficientPrecision, CoordinatePrecision> extract(HMatrix<CoefficientPrecision, CoordinatePrecision> &res, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const char trans) {
//     if (trans == 'U') {
//         if (A->get_children().size() > 0) {
//             for (auto &child : A->get_children()) {
//                 auto res_child = res.add_child(child->get_target_cluster().get(), child->get_source_cluster().get());
//                 if (child->get_source_cluster().get_offset() >= child->get_target_cluster().get_offset()) {
//                     extract(res_child, child, trans);
//                 } else {
//                     Matrix<CoefficientPrecision> zero(child->get_target_cluster().get_size(), child->get_source_cluster().get_size());
//                     res_child->set_dense_data(zero);
//                 }
//             }
//         } else {
//             // la matrice dont on veut extraire est une feuille avec s >=t
//             if (A->get_target_cluster() == A->get_source_cluster()) {
//                 // s=t on est dense
//                 auto mat = A->get_dense_data();
//                 auto udiag(A->get_target_cluster().get_size(), A->get_target_cluster().get_size());
//                 for (int k = 0; k < A->get_target_cluster().get_size(); ++k) {
//                     for (int l = k; l < A->get_target_cluster().get_size(); ++l) {
//                         ldiag(k, l) = mat(k, l);
//                     }
//                 }
//             } else {
//                 if (A->is_dense()) {
//                     res->set_dense_data(*A->get_dense_data());
//                 } else {
//                 }
//             }
//         }
//     }
// }
template <typename T, typename GeneratorTestType>
std::vector<T> test_hlu(int size, htool::underlying_type<T> epsilon, htool::underlying_type<T> eta) {
    // bool is_error = false;
    // double eta    = 100;
    htool::underlying_type<T> error;

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', size, size, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(*test_case.root_cluster_A_output, *test_case.root_cluster_A_input, epsilon, eta, 'N', 'N', -1, -1, -1);
    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder_A.build(*test_case.operator_A);
    std::cout << " compression A " << get_compression(A) << std::endl;
    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    // int ni_X = test_case.root_cluster_X_input->get_size();
    // int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(ni_A, 1), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    test_case.operator_A->copy_submatrix(no_A, ni_A, test_case.root_cluster_A_output->get_offset(), test_case.root_cluster_A_input->get_offset(), A_dense.data());
    generate_random_matrix(X_dense);

    add_matrix_matrix_product('N', 'N', T(1.), A_dense, X_dense, T(0.), B_dense);

    // LU factorization
    matrix_test        = B_dense;
    auto start_time_lu = std::chrono::high_resolution_clock::now();
    lu_factorization(A);
    auto end_time_lu = std::chrono::high_resolution_clock::now();
    auto duration_lu = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_lu - start_time_lu).count();

    auto compr_A = get_compression(A);
    lu_solve('N', A, matrix_test);
    error = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    std::cout << "______________________________________________" << std::endl;
    std::cout << "Size : -------------------------------------->" << size << std::endl;
    std::cout << "time hlu : ---------------------------------->" << duration_lu << std::endl;
    std::cout << "compression: -------------------------------->" << compr_A << std::endl;
    std::cout << "error lu solve: ----------------------------->" << error << endl;
    std::cout << "______________________________________________" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::vector<T> res(4);
    res[0] = size;
    res[1] = duration_lu;
    res[2] = compr_A;
    res[3] = error;
    return res;
}
