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
    Matricegenerator(Matrix<CoefficientPrecision> &mat0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : mat(mat0), target(target0), source(source0){};

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

//// Pour extraire Lh et Uh de lu_factorisation() on luyiu donne L ou U
template <class CoefficientPrecision, class CoordinatePrecision>
HMatrix<CoefficientPrecision, CoordinatePrecision> extract(HMatrix<CoefficientPrecision, CoordinatePrecision> &res, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const char trans) {
    if (trans == 'U') {
        if (A->get_children().size() > 0) {
            for (auto &child : A->get_children()) {
                auto res_child = res.add_child(child->get_target_cluster().get(), child->get_source_cluster().get());
                if (child->get_source_cluster().get_offset() >= child->get_target_cluster().get_offset()) {
                    extract(res_child, child, char);
                } else {
                    Matrix<CoefficientPrecision> zero(child->get_target_cluster().get_size(), child->get_source_cluster().get_size());
                    res_child->set_dense_data(zero);
                }
            }
        } else {
            // la matrice dont on veut extraire est une feuille avec s >=t
            if (A->get_target_cluster() == A->get_source_cluster()) {
                // s=t on est dense
                auto mat = A->get_dense_data();
                auto udiag(A->get_target_cluster().get_size(), A->get_target_cluster().get_size());
                for (int k = 0; k < A->get_target_cluster().get_size(); ++k) {
                    for (int l = k; l < A->get_target_cluster().get_size(); ++l) {
                        ldiag(k, l) = mat(k, l);
                    }
                }
            } else {
                if (A->is_dense()) {
                    res->set_dense_data(*A->get_dense_data());
                } else {
                }
            }
        }
    }
}
template <typename T, typename GeneratorTestType>
bool test_hmatrix_lu(int size, htool::underlying_type<T> epsilon, htool::underlying_type<T> eta) {
    bool is_error = false;
    htool::underlying_type<T> error;

    ///////////////////////
    ////// GENERATION MAILLAGE  :  size points en 3 dimensions
    std::vector<double> p1(3 * size);
    create_disk(3, 0.0, size, p1.data());

    std::cout << "création cluster ok" << std::endl;
    //////////////////////

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    //////////////////////
    // Clustering
    ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> recursive_build_strategy_1(size, 3, p1.data(), 2, 2);
    std::shared_ptr<Cluster<double>> root_cluster_1 = std::make_shared<Cluster<double>>(recursive_build_strategy_1.create_cluster_tree());
    std::cout << "Cluster tree ok" << std::endl;
    //////////////////////

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    ///////////////////
    // Generator: donné en argument de la méthode
    auto permutation = root_cluster_1->get_permutation();
    GeneratorTestType generator(3, size, size, p1, p1, root_cluster_1, root_cluster_1);

    Matrix<double> reference_num_htool(size, size);
    generator.copy_submatrix(size, size, 0, 0, reference_num_htool.data());

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    //////////////
    /// Hmatrix in the good permutation
    /////////////
    std::cout << " Facrotisation dgetrf pourt avoir le pivot" << std::endl;
    Matrix<double> L(size, size);
    Matrix<double> U(size, size);
    std::vector<int> ipiv(size, 0.0);
    int info = -1;
    auto A   = reference_num_htool;
    Lapack<double>::getrf(&size, &size, A->data(), &size, ipiv.data(), &info);

    for (int i = 0; i < size; ++i) {
        L(i, i) = 1;
        U(i, i) = A(i, i);

        for (int j = 0; j < i; ++j) {
            L(i, j) = A(i, j);
            U(j, i) = A(j, i);
        }
    }
    std::cout << " Pivot obtenue" << std::endl;

    // // Matrice Lh et Uh
    /// Warning : on peut pas juste donné L et U parce que Htool fais des permutations et la matrice qu'il compresserait serait pas forcemment triangtulaire
    /// IL faut mettre L et U dans la permutation inverse de clusters
    std::cout << " Assemblage Lh et Uh " << std::endl;
    std::cout << "L et U doivent être formatées pour que htool produise des matrices triangulaires" << std::endl;
    auto ll = get_unperm_mat(L, *root_cluster_1, *root_cluster_1);
    Matricegenerator<double, double> l_generator(ll, *root_cluster_1, *root_cluster_1);
    HMatrixTreeBuilder<double, double> lh_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    auto Lh = lh_builder.build(l_generator);
    Matrix<double> ldense(size, size);
    copy_to_dense(Lh, ldense.data());
    std::cout << "Lh assembled, error relative Lh-L  :" << normFrob(L - ldense) / normFrob(L) << std::endl;

    auto uu = get_unperm_mat(U, *root_cluster_1, *root_cluster_1);
    Matricegenerator<double, double> u_generator(uu, *root_cluster_1, *root_cluster_1);
    HMatrixTreeBuilder<double, double> uh_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    auto Uh = uh_builder.build(u_generator);
    Matrix<double> udense(size, size);
    copy_to_dense(Uh, udense.data());
    std::cout << "Uh assembled, error relative Uh-U : " << normFrob(U - udense) / normFrob(U) << std::endl;

    // Hmatrice dans la bonne permutation = Lh*Uh
    // on appelle HLU sur (Lh*Uh) et on aimerait retomber sur Lh et Uh
    Lh.set_epsilon(epsilon);
    Uh.set_epsilon(epsilon);
    HMatrix<double, double> ref_nopivot(root_cluster_1, root_cluster_1);
    add_hmatrix_hmatrix_product('N', 'N', 1.0, Lh, Uh, 0.0, ref_nopivot);
    auto produit_LU = Lh.hmatrix_triangular_product(Uh, 'L', 'U');
    Matrix<double> prod_dense(size, size);
    auto ref_lu = L * U;
    copy_to_dense(produit_LU, prod_dense.data());
    std::cout << "Compression Hmatrix dans la bonne numérotation (Lh*Uh ) : " << produit_LU.get_compression() << "   et l'erreur  : " << normFrob(prod_dense - ref_lu) / normFrob(ref_lu);
    lu_factorization(produit_LU)

        return is_error;
}
