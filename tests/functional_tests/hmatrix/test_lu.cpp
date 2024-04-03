#include <chrono>
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

int testt(int x, int y, int (*f)(int a, int b)) {
    return f(x, y);
}
int add(int x, int y) {
    return x + y;
}
using namespace htool;
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
//////////////////////////
//// GENERATOR -> Prend Une matrice ou un fichier
//////////////////////////
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
// std::vector<double> generate_random_vector(int size) {
//     std::random_device rd;  // Source d'entropie aléatoire
//     std::mt19937 gen(rd()); // Générateur de nombres pseudo-aléatoires

//     std::uniform_real_distribution<double> dis(1.0, 10.0); // Plage de valeurs pour les nombres aléatoires (ici de 1.0 à 10.0)

//     std::vector<double> random_vector;
//     random_vector.reserve(size); // Allocation de mémoire pour le vecteur

//     for (int i = 0; i < size; ++i) {
//         random_vector.push_back(dis(gen)); // Ajout d'un nombre aléatoire dans la plage à chaque itération
//     }

//     return random_vector;
// }

void ff(const Matrix<double> M, std::vector<double> &y_in, std::vector<double> &x_out) {
    for (int j = 0; j < M.nb_rows(); ++j) {
        x_out[j] = y_in[j];
        for (int i = j + 1; i < M.nb_rows(); ++i) {
            y_in[i] = y_in[i] - M(i, j) * x_out[j];
        }
    }
}

void solveLowerTriangular(const Matrix<double> &L, const std::vector<double> &b, std::vector<double> &x) {
    int n = L.nb_rows(); // Supposons que L soit une matrice carrée n x n

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L(i, j) * x[j];
        }
        x[i] = (b[i] - sum) / L(i, i);
    }
}

///////////////////////////////////////////////////////////////////
int main() {
    std::cout << "TEST HLU" << std::endl;
    std::cout << "____________________________" << std::endl;
    int size = std::pow(2, 13) + 111;

    std::cout << "_________________________________________" << std::endl;
    std::cout << "HMATRIX" << std::endl;
    double epsilon = 1e-3;
    double eta     = 1.0;

    ////// GENERATION MAILLAGE
    std::vector<double> p1(3 * size);
    create_disk(3, 0., size, p1.data());

    std::cout << "création cluster ok" << std::endl;
    //////////////////////
    // Clustering
    //////////////////////
    ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> recursive_build_strategy_1(size, 3, p1.data(), 2, 2);

    std::shared_ptr<Cluster<double>> root_cluster_1 = std::make_shared<Cluster<double>>(recursive_build_strategy_1.create_cluster_tree());
    std::cout << "Cluster tree ok" << std::endl;
    // Generator
    auto permutation = root_cluster_1->get_permutation();
    GeneratorDouble generator_LU(3, size, size, p1, p1, root_cluster_1, root_cluster_1);

    Matrix<double> reference_num_htool(size, size);
    // std::vector<int> row(size);
    // for (int k = 0; k < size; ++k) {
    //     row[k] = permutation[k];
    // }

    generator_LU.copy_submatrix(size, size, 0, 0, reference_num_htool.data());
    // HMATRIX
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder_LU(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');

    // build
    auto root_hmatrix_LU = hmatrix_tree_builder_LU.build(generator_LU);

    Matrix<double> lu(size, size);
    copy_to_dense(root_hmatrix_LU, lu.data());
    std::cout
        << "erreur d'approximation Hmatrices reference " << normFrob(reference_num_htool - lu) / normFrob(reference_num_htool) << std::endl;

    auto prod = root_hmatrix_LU.hmatrix_product(root_hmatrix_LU);
    // auto prod = root_hmatrix_LU.hmatrix_product_new(root_hmatrix_LU);

    Matrix<double> prod_dense(size, size);
    copy_to_dense(prod, prod_dense.data());

    auto prod_ref = reference_num_htool * reference_num_htool;
    std::cout << "error prod " << normFrob(prod_dense - prod_ref) / normFrob(prod_ref) << std::endl;

    Matrix<double> L(size, size);
    Matrix<double> U(size, size);
    std::vector<int> ipiv(size, 0.0);
    int info = -1;
    auto A   = reference_num_htool;
    Lapack<double>::getrf(&size, &size, A.data(), &size, ipiv.data(), &info);

    for (int i = 0; i < size; ++i) {
        L(i, i) = 1;
        U(i, i) = A(i, i);

        for (int j = 0; j < i; ++j) {
            L(i, j) = A(i, j);
            U(j, i) = A(j, i);
        }
    }

    // auto M  = root_hmatrix_LU.hmatrix_matrix(U);
    // auto mm = reference_num_htool * U;
    // std::cout << "erreur hmatrix matrix " << normFrob(M - mm) / normFrob(mm) << std::endl;

    // auto MT  = root_hmatrix_LU.matrix_hmatrix(L);
    // auto mmt = L * reference_num_htool;
    // std::cout << "erreur matrix hmatrix " << normFrob(MT - mmt) / normFrob(mmt) << std::endl;
    // std::cout << "________________________" << std::endl;
    // std::cout << "Assemblage de matrice hiérarchique triangulaire pour nos tests : " << std::endl;
    // // Matrice Lh et Uh
    auto ll = get_unperm_mat(L, *root_cluster_1, *root_cluster_1);
    Matricegenerator<double, double> l_generator(ll, *root_cluster_1, *root_cluster_1);
    HMatrixTreeBuilder<double, double> lh_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    auto Lh = lh_builder.build(l_generator);
    Matrix<double> ldense(size, size);
    copy_to_dense(Lh, ldense.data());
    std::cout << "erreur Lh :" << normFrob(L - ldense) / normFrob(L) << std::endl;

    auto uu = get_unperm_mat(U, *root_cluster_1, *root_cluster_1);
    Matricegenerator<double, double> u_generator(uu, *root_cluster_1, *root_cluster_1);
    HMatrixTreeBuilder<double, double> uh_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    auto Uh = uh_builder.build(u_generator);
    Matrix<double> udense(size, size);
    copy_to_dense(Uh, udense.data());
    std::cout << "erreur uh : " << normFrob(U - udense) / normFrob(U) << std::endl;

    // // ///////////////////////////////////////////
    // // //// Deux sous clusters pour tester les appelles aux sous blocs
    auto &t = ((root_cluster_1->get_children()[1])->get_children())[0];
    auto &s = ((root_cluster_1->get_children()[0])->get_children())[1];
    // //////////////////////////////////////////////////////
    // std::cout << "____________________________" << std::endl;
    std::cout << "+++++++++++ Forward +++++++++++" << std::endl;
    // std::cout << "------> Trouver x tq Lx = y" << std::endl;
    auto x_ref_forward = generate_random_vector(size);
    auto yy            = ldense * x_ref_forward;
    std::vector<double> xres(size, 0.0);
    Lh.forward_substitution_s(*root_cluster_1, 0, yy, xres);
    std::cout << " ereur forward  : " << norm2(xres - x_ref_forward) / norm2(x_ref_forward) << std::endl;

    std::cout << "      sur un bloc :  txt= " << t->get_size() << ',' << t->get_offset() << std::endl;
    auto x_ref_forward_bloc = generate_random_vector(t->get_size());
    std::vector<double> yforward_bloc(t->get_size());
    Lh.get_block(t->get_size(), t->get_size(), t->get_offset(), t->get_offset())->add_vector_product('N', 1.0, x_ref_forward_bloc.data(), 1.0, yforward_bloc.data());
    std::vector<double> res_forward_bloc(t->get_size());
    Lh.forward_substitution_s(*t, t->get_offset(), yforward_bloc, res_forward_bloc);
    std::cout << "                   erreur bock : " << norm2(x_ref_forward_bloc - res_forward_bloc) / norm2(x_ref_forward_bloc) << std::endl;

    std::cout
        << "____________________________" << std::endl;
    std::cout << "+++++++++++ Forward_T +++++++++++" << std::endl;
    std::cout << "------> Trouver x tq x^T U = y^T" << std::endl;
    auto x_ref_forward_T = generate_random_vector(size);
    std::vector<double> y_forward_T(size);
    Uh.add_vector_product('T', 1.0, x_ref_forward_T.data(), 1.0, y_forward_T.data());
    std::vector<double> res_forward_T(size);
    // Uh.forward_substitution_T(*root_cluster_1, res_forward_T, y_forward_T);
    Uh.forward_substitution_T_s(*root_cluster_1, 0, y_forward_T, res_forward_T);

    std::cout << "erreur forward subsitution_T :" << norm2(res_forward_T - x_ref_forward_T) / norm2(x_ref_forward_T) << std::endl;
    auto x_ref_forward_blockt = generate_random_vector(t->get_size());
    Matrix<double> ut(t->get_size(), t->get_size());
    for (int k = 0; k < t->get_size(); ++k) {
        for (int l = 0; l < t->get_size(); ++l) {
            ut(k, l) = udense(k + t->get_offset(), l + t->get_offset());
        }
    }
    auto y_forward_blockt = ut.transp(ut) * x_ref_forward_blockt;
    std::vector<double> res_forward_blockt(t->get_size());
    Uh.forward_substitution_T_s(*t, t->get_offset(), y_forward_blockt, res_forward_blockt);
    std::cout << "                   erreur bock : " << norm2(res_forward_blockt - x_ref_forward_blockt) / norm2(x_ref_forward_blockt) << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "____________________________________________________" << std::endl;
    // std::cout << "Forward_M : -------------------> LX = Y" << std::endl;

    auto Y_forward_M = Lh.hmatrix_product(root_hmatrix_LU);
    auto ref_prod    = L * reference_num_htool;
    Matrix<double> test_M(size, size);
    copy_to_dense(Y_forward_M, test_M.data());
    std::cout << "erreur produit : " << normFrob(test_M - ref_prod) / normFrob(ref_prod) << std::endl;
    HMatrix<double, double> res_Forward_M(root_cluster_1, root_cluster_1);
    res_Forward_M.copy_zero(Y_forward_M);
    FM(Lh, *root_cluster_1, *root_cluster_1, Y_forward_M, res_Forward_M);
    // FM_build(Lh, *root_cluster_1, *root_cluster_1, Y_forward_M, res_Forward_M);

    Matrix<double> res_Forward_M_dense(size, size);
    copy_to_dense(res_Forward_M, res_Forward_M_dense.data());
    std::cout << "erreur Forward_M : X-Xref :  " << normFrob(res_Forward_M_dense - reference_num_htool) / normFrob(reference_num_htool) << std::endl;
    // std::cout << "erreur Forward_M : L*X-L*Xref :  " << normFrob(ldense * res_Forward_M_dense - ldense * lu) / normFrob(ldense * lu) << std::endl;

    // std::cout << "____________________________" << std::endl;
    std::cout << "+++++++++++ Forward_M_T +++++++++++" << std::endl;
    std::cout << "------> Trouver X tq XU = Y" << std::endl;
    auto Y_forward_M_T = root_hmatrix_LU.hmatrix_product(Uh);
    auto ref_prod_T    = reference_num_htool * U;
    Matrix<double> test_M_T(size, size);
    copy_to_dense(Y_forward_M_T, test_M_T.data());
    std::cout << "erreur produit : " << normFrob(test_M_T - reference_num_htool * U) / normFrob(reference_num_htool * U) << std::endl;
    HMatrix<double, double> res_Forward_M_T(root_cluster_1, root_cluster_1);
    Y_forward_M_T.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    res_Forward_M_T.copy_zero(Y_forward_M_T);

    // FM_T(Uh, *root_cluster_1, *root_cluster_1, Y_forward_M_T, res_Forward_M_T);

    // FM_T_build(Uh, *root_cluster_1, *root_cluster_1, Y_forward_M_T, res_Forward_M_T);
    FM_T(Uh, *root_cluster_1, *root_cluster_1, Y_forward_M_T, res_Forward_M_T);
    Matrix<double> res_Forward_M_T_dense(size, size);

    copy_to_dense(res_Forward_M_T, res_Forward_M_T_dense.data());
    std::cout << "erreur Forward_M_T : X-Xref :  " << normFrob(res_Forward_M_T_dense - reference_num_htool) / normFrob(reference_num_htool) << std::endl;
    std::cout << "erreur Forward_M_T : X*U-Xref*U : " << normFrob(ldense * res_Forward_M_T_dense - ldense * lu) / normFrob(ldense * lu) << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    // // std::cout << "____________________________" << std::endl;
    // // std::cout << "+++++++++++ HLU +++++++++++" << std::endl;
    // // std::cout << "------> Trouver Lh et Uh tq Lh*Uh=M " << std::endl;
    Lh.set_epsilon(epsilon);
    Uh.set_epsilon(epsilon);
    auto produit_LU = Lh.hmatrix_product(Uh);
    Matrix<double> ref_lu(size, size);
    copy_to_dense(produit_LU, ref_lu.data());

    produit_LU.set_epsilon(epsilon);
    HMatrix<double, double> Llh(root_cluster_1, root_cluster_1);
    Llh.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    Llh.set_epsilon(epsilon);
    Llh.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());

    HMatrix<double, double> Uuh(root_cluster_1, root_cluster_1);
    Uuh.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    Uuh.set_epsilon(epsilon);
    Uuh.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    Uuh.copy_zero(produit_LU);
    Llh.copy_zero(produit_LU);

    HLU_noperm(produit_LU, *root_cluster_1, Llh, Uuh);
    Matrix<double> uuu(size, size);
    Matrix<double> lll(size, size);
    copy_to_dense(Llh, lll.data());
    copy_to_dense(Uuh, uuu.data());
    std::cout << "erreur LU : " << normFrob(lll * uuu - ref_lu) / normFrob(ref_lu) << std::endl;
    // HMatrix<double, double> Permutation(root_cluster_1, root_cluster_1);
    // Permutation.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // Permutation.set_epsilon(epsilon);
    // Permutation.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // Permutation.copy_zero(produit_LU);
    // // Matrix<double> ref_lu(size, size);
    // // copy_to_dense(produit_LU, ref_lu.data());
    // Matrix<double> dense_perm(size, size);
    // // std::cout << "norme de Lh*Uh la matrice que l'on factorise  : " << normFrob(ref_lu) << "!!" << normFrob(ldense * udense) << std::endl;
    // HMatrix<double, double> unperm(root_cluster_1, root_cluster_1);

    // Permutation.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // unperm.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // Permutation.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // unperm.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // unperm.copy_zero(produit_LU);

    // HMatrix_PLU(produit_LU, *root_cluster_1, Llh, Uuh, Permutation, unperm);

    // // // std::vector<int> pivot(size);
    // // // HMatrix_PLU(produit_LU, *root_cluster_1, Llh, Uuh, pivot);
    // // // FACTORISATION
    // // // HMatrix_PLU_new(produit_LU, *root_cluster_1, Llh, Uuh, Permutation);
    // copy_to_dense(Permutation, dense_perm.data());
    // Matrix<double> denseunperm(size, size);
    // copy_to_dense(unperm, denseunperm.data());

    // auto test_lu = Llh.hmatrix_product(Uuh);
    // Matrix<double> lu_dense(size, size);
    // copy_to_dense(test_lu, lu_dense.data());
    // std::cout << "erreur PLU ::" << normFrob(ref_lu - lu_dense) / normFrob(ref_lu) << std::endl;
    // Matrix<double> ltest(size, size);
    // copy_to_dense(Llh, ltest.data());
    // Matrix<double> utest(size, size);
    // copy_to_dense(Uuh, utest.data());
    // std::cout << "erreur L  :" << normFrob(L - ltest) / normFrob(L) << std::endl;
    // std::cout << "erreur U : " << normFrob(U - utest) / normFrob(U) << std::endl;
    /////////////////////////////////////////////////////////////////////////
    // Matrix<double> FM_dense(size, size);
    // copy_to_dense(Forward_res, FM_dense.data());
    // std::cout << " erreur Forward_M :  " << normFrob(FM_dense - reference_num_htool) / normFrob(reference_num_htool) << std::endl;
    // ff(udense.transp(udense), yy, xres);
    // solveLowerTriangular(ldense, yy, xres);
    // double alpha = 1.0;
    // std::vector<double> ldense_vect(size * size);
    // int un = 1;
    // for (int k = 0; k < size; ++k) {
    //     for (int l = k; l < size; ++l) {
    //         ldense_vect[size * k + l] = ldense(l, k);
    //     }
    // }
    // Blas<double>::trsm("L", "L", "N", "N", &size, &un, &alpha, ldense_vect.data(), &size, yy.data(), &size); // std::cout << "trsm ok " << std::endl;
    // std::cout << norm2(yy - x_ref_forward) / norm2(x_ref_forward) << std::endl;
    // yy = ldense * x_ref_forward;
    // solveLowerTriangular(ldense, yy, xres);
    // std::cout << norm2(xres - x_ref_forward) / norm2(x_ref_forward) << std::endl;

    // auto rd = generate_random_vector(size * size);
    // for (int i = 0; i < size; ++i) {
    //     L(i, i) = 1.0;
    //     for (int j = 0; j < i; ++j) {
    //         L(i, j) = rd[std::max(size * i + j, size - 100)];
    //     }
    // }
    // auto yref = L * x_ref_forward;
    // std::cout << "norme " << norm2(yref) << std::endl;
    // Blas<double>::trsm("L", "L", "N", "U", &size, &un, &alpha, L.data(), &size, yref.data(), &size);
    // std::cout << "??????????" << norm2(yref - x_ref_forward) / norm2(x_ref_forward) << std::endl;
    // yy = L * x_ref_forward;
    // solveLowerTriangular(L, yy, xres);
    // std::cout << norm2(xres - x_ref_forward) / norm2(x_ref_forward) << std::endl;

    // std::cout << yy.size() << "=? " << x_ref_forward.size() << "            " << norm2(yy) << std::endl;
    // std::vector<double> res(size);
    // Blas<double>::gemv("N", &size, &size, &alpha, ldense.data(), &size, yy.data(), &size, &alpha, res.data(), &size);
    // std::cout << norm2(res - (ldense * x_ref_forward)) / norm2(ldense * x_ref_forward) << std::endl;

    // std::cout << "pour le dense " << norm2(xres - x_ref_forward) / norm2(x_ref_forward) << std::endl;
    // std::cout << normFrob(udense.transp(udense) - ldense) / normFrob(udense) << std::endl;
    // auto yy = ldense * xx;
    // Matrix<double> lup(root_cluster_1->get_children()[0]->get_size(), root_cluster_1->get_children()[1]->get_size());
    // copy_to_dense(*Lh.get_block(root_cluster_1->get_children()[0]->get_size(), root_cluster_1->get_children()[1]->get_size(), root_cluster_1->get_children()[0]->get_offset(), root_cluster_1->get_children()[1]->get_offset()), lup.data());
    // std::cout << "0 =?????????????????" << normFrob(lup) << std::endl;
    // auto x_ref_forward_block = generate_random_vector(t->get_size());
    // Matrix<double> lt(t->get_size(), t->get_size());
    // for (int k = 0; k < t->get_size(); ++k) {
    //     for (int l = 0; l < t->get_size(); ++l) {
    //         lt(k, l) = ldense(k + t->get_offset(), l + t->get_offset());
    //     }
    // }
    // auto y_forward_block = lt * x_ref_forward_block;
    // std::vector<double> res_forward_block(t->get_size());
    // Lh.forward_substitution_s(*t, t->get_offset(), y_forward_block, res_forward_block);
    // std::cout << "erreur bock : " << norm2(res_forward_block - x_ref_forward_block) / norm2(x_ref_forward_block) << std::endl;

    // auto x_ref_forward_blockt = generate_random_vector(t->get_size());
    // Matrix<double> ut(t->get_size(), t->get_size());
    // for (int k = 0; k < t->get_size(); ++k) {
    //     for (int l = 0; l < t->get_size(); ++l) {
    //         ut(k, l) = udense(k + t->get_offset(), l + t->get_offset());
    //     }
    // }
    // auto y_forward_blockt = ut.transp(ut) * x_ref_forward_blockt;
    // std::vector<double> res_forward_blockt(t->get_size());
    // Uh.forward_substitution_T_s(*t, t->get_offset(), y_forward_blockt, res_forward_blockt);
    // std::cout << "erreur bock : " << norm2(res_forward_blockt - x_ref_forward_blockt) / norm2(x_ref_forward_blockt) << std::endl;

    // std::cout << "____________________________" << std::endl;
    // std::cout << "+++++++++++ Forward_M +++++++++++" << std::endl;
    // std::cout << "------> Trouver X tq LX = Y" << std::endl;

    // auto Y_forward_M = Lh.hmatrix_product(root_hmatrix_LU);
    // Matrix<double> test_M(size, size);
    // copy_to_dense(Y_forward_M, test_M.data());
    // std::cout << "erreur produit : " << normFrob(test_M - ldense * lu) / normFrob(ldense * lu) << std::endl;
    // HMatrix<double, double> res_Forward_M(root_cluster_1, root_cluster_1);
    // res_Forward_M.copy_zero(Y_forward_M);
    // Forward_Matrix(Lh, *root_cluster_1, *root_cluster_1, Y_forward_M, res_Forward_M);
    // Matrix<double> res_Forward_M_dense(size, size);
    // copy_to_dense(res_Forward_M, res_Forward_M_dense.data());
    // std::cout << "erreur Forward_M : X-Xref :  " << normFrob(res_Forward_M_dense - lu) / normFrob(lu) << std::endl;
    // std::cout << "erreur Forward_M : L*X-L*Xref :  " << normFrob(ldense * res_Forward_M_dense - ldense * lu) / normFrob(ldense * lu) << std::endl;
    // ////////////////////////////////////////////////////////////////////////////
    // std::cout << "   sur un bloc" << std::endl;
    // HMatrix<double, double> Y_forward_M_restr(root_cluster_1, root_cluster_1);
    // Y_forward_M_restr.copy_zero(root_hmatrix_LU);
    // auto Yts = Lh.get_block(t->get_size(), t->get_size(), t->get_offset(), t->get_offset())->hmatrix_product(*root_hmatrix_LU.get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset()));
    // Y_forward_M_restr.assign(Yts, *t, *s);
    // Matrix<double> test_M_restr(t->get_size(), s->get_size());
    // copy_to_dense(Yts, test_M_restr.data());
    // HMatrix<double, double> res_Forward_M_restr(root_cluster_1, root_cluster_1);
    // res_Forward_M_restr.copy_zero(Y_forward_M_restr);
    // Forward_Matrix(Lh, *t, *s, Yts, res_Forward_M_restr);
    // Matrix<double> res_Forward_M_restr_dense(t->get_size(), s->get_size());
    // copy_to_dense(*res_Forward_M_restr.get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset()), res_Forward_M_restr_dense.data());
    // Matrix<double> lu_restr(t->get_size(), s->get_size());
    // for (int k = 0; k < t->get_size(); ++k) {
    //     for (int l = 0; l < s->get_size(); ++l) {
    //         lu_restr(k, l) = lu(k + t->get_offset(), l + s->get_offset());
    //     }
    // }
    // std::cout << "   erreur Forward_M_restr : X|ts-Xref|ts :  " << normFrob(res_Forward_M_restr_dense - lu_restr) / normFrob(lu_restr) << std::endl;
    // // std::cout << "   erreur Forward_M : L|tt*X|ts-L|tt*Xref|ts :  " << normFrob(ldense * res_Forward_M_dense - ldense * lu) / normFrob(ldense * lu) << std::endl;
    // ////////////////////////////////////////////////////////////////////////////
    // std::cout << "____________________________" << std::endl;
    // std::cout << "+++++++++++ Forward_M_T +++++++++++" << std::endl;
    // std::cout << "------> Trouver X tq XU = Y" << std::endl;
    // auto Y_forward_M_T = root_hmatrix_LU.hmatrix_product(Uh);
    // Matrix<double> test_M_T(size, size);
    // copy_to_dense(Y_forward_M_T, test_M_T.data());
    // std::cout << "erreur produit : " << normFrob(test_M_T - lu * udense) / normFrob(lu * udense) << std::endl;
    // HMatrix<double, double> res_Forward_M_T(root_cluster_1, root_cluster_1);
    // res_Forward_M_T.copy_zero(Y_forward_M_T);
    // Forward_Matrix_T(Uh, *root_cluster_1, *root_cluster_1, Y_forward_M_T, res_Forward_M_T);
    // Matrix<double> res_Forward_M_T_dense(size, size);
    // copy_to_dense(res_Forward_M_T, res_Forward_M_T_dense.data());
    // std::cout << "erreur Forward_M_T : X-Xref :  " << normFrob(res_Forward_M_T_dense - lu) / normFrob(lu) << std::endl;
    // std::cout << "erreur Forward_M_T : X*U-Xref*U : " << normFrob(ldense * res_Forward_M_T_dense - ldense * lu) / normFrob(ldense * lu) << std::endl;

    // ////////////////////////////////////////////////////////////////////////////
    // std::cout << "  sur un bloc " << std::endl;
    // HMatrix<double, double> Y_forward_M_T_restr(root_cluster_1, root_cluster_1);
    // Y_forward_M_T_restr.copy_zero(root_hmatrix_LU);
    // auto Yts_T = root_hmatrix_LU.get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset())->hmatrix_product(*Uh.get_block(s->get_size(), s->get_size(), s->get_offset(), s->get_offset()));
    // Y_forward_M_T_restr.assign(Yts_T, *t, *s);
    // Matrix<double> test_M_T_restr(t->get_size(), s->get_size());
    // copy_to_dense(Yts_T, test_M_T_restr.data());
    // HMatrix<double, double> res_Forward_M_T_restr(root_cluster_1, root_cluster_1);
    // res_Forward_M_T_restr.copy_zero(Y_forward_M_T_restr);
    // Forward_Matrix_T(Uh, *s, *t, Yts_T, res_Forward_M_T_restr);
    // Matrix<double> res_Forward_M_T_restr_dense(t->get_size(), s->get_size());
    // copy_to_dense(*res_Forward_M_T_restr.get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset()), res_Forward_M_T_restr_dense.data());
    // std::cout << "   erreur Forward_M_T_restr : X|ts-Xref|ts :  " << normFrob(res_Forward_M_T_restr_dense - lu_restr) / normFrob(lu_restr) << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // std::cout << "____________________________" << std::endl;
    // std::cout << "+++++++++++ HLU +++++++++++" << std::endl;
    // std::cout << "------> Trouver Lh et Uh tq Lh*Uh=M " << std::endl;
    // auto produit_LU = Lh.hmatrix_product(Uh);
    // // root_hmatrix_LU.save_plot("prod_LU_test0");
    // HMatrix<double, double> Llh(root_cluster_1, root_cluster_1);
    // Llh.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // Llh.set_epsilon(epsilon);
    // Llh.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());

    // HMatrix<double, double> Uuh(root_cluster_1, root_cluster_1);
    // Uuh.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // Uuh.set_epsilon(epsilon);
    // Uuh.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // Uuh.copy_zero(root_hmatrix_LU);
    // Llh.copy_zero(root_hmatrix_LU);

    // HMatrix<double, double> Permutation(root_cluster_1, root_cluster_1);
    // Permutation.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // Permutation.set_epsilon(epsilon);
    // Permutation.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // Permutation.copy_zero(root_hmatrix_LU);
    // // // Matrix<double> ref_lu(size, size);
    // // // copy_to_dense(produit_LU, ref_lu.data());
    // Matrix<double> dense_perm(size, size);
    // // std::cout << "norme de Lh*Uh la matrice que l'on factorise  : " << normFrob(ref_lu) << "!!" << normFrob(ldense * udense) << std::endl;
    // HMatrix<double, double> unperm(root_cluster_1, root_cluster_1);

    // Permutation.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // unperm.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // Permutation.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // unperm.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // unperm.copy_zero(root_hmatrix_LU);

    // // // // LA MATRICE QU ON VA FACTORISER
    // auto bt = root_hmatrix_LU.get_block(32, 32, 320, 64);
    // std::cout << "dense ? " << bt->is_dense() << "low rank ? " << bt->is_low_rank() << "nb child : " << bt->get_children().size() << std::endl;
    // Matrix<double> bb(32, 32);
    // copy_to_dense(*bt, bb.data());
    // for (int k = 0; k < 32; ++k) {
    //     for (int l = 0; l < 31; ++l) {
    //         std::cout << bb(k, l) << ',';
    //     }
    //     std::cout << bb(k, 31) << std::endl;
    // }
    // std::cout
    // << "_________________________________________" << std::endl;
    // HMatrix_PLU(produit_LU, *root_cluster_1, Llh, Uuh, Permutation, unperm);
    // std::cout << " BT ::::::::::::::::::::::::::::: is dense ? " << ct->is_dense() << " is_low_rank" << ct->is_low_rank() << std::endl;
    // HMatrix_PLU(root_hmatrix_LU, *root_cluster_1, Llh, Uuh, Permutation, unperm);

    // std::vector<int> pivot(size);
    // HMatrix_PLU(produit_LU, *root_cluster_1, Llh, Uuh, pivot);
    // FACTORISATION
    // HMatrix_PLU_new(produit_LU, *root_cluster_1, Llh, Uuh, Permutation);
    // copy_to_dense(Permutation, dense_perm.data());
    // Matrix<double> denseunperm(size, size);
    // copy_to_dense(unperm, denseunperm.data());

    // auto test_lu = Llh.hmatrix_product(Uuh);
    // Matrix<double> lu_dense(size, size);
    // copy_to_dense(test_lu, lu_dense.data());
    // // copy_to_dense(Permutation, dense_perm.data());
    // // for (int k = 0; k < 15; ++k) {
    // //     for (int l = 0; l < 15; ++l) {
    // //         std::cout << dense_perm(k, l) << ';';
    // //     }
    // //     std::cout << std::endl;
    // // }
    // std::cout << "erreur PLU ::" << normFrob(lu - dense_perm * lu_dense) / normFrob(lu) << std::endl;
    // std::cout << "erreur PLU ::" << normFrob(denseunperm * lu - lu_dense) / normFrob(lu) << std::endl;
    // std::cout << "erreur PLU ::" << normFrob(ref_lu - dense_perm * lu_dense) / normFrob(ref_lu) << std::endl;
    // std::cout << "erreur PLU ::" << normFrob(denseunperm * ref_lu - lu_dense) / normFrob(ref_lu) << std::endl;

    // auto test_lu = Llh.hmatrix_product(Uuh);
    // Matrix<double> ludense(size, size);
    // copy_to_dense(test_lu, ludense.data());
    // // PRODUIT Lh Uh-> DENSE
    // Matrix<double> ll(size, size);
    // Matrix<double> uu(size, size);
    // copy_to_dense(Uuh, uu.data());
    // Matrix<double> inv_perm(size, size);
    // for (int k = 0; k < size; ++k) {
    //     inv_perm(k, pivot[k] - 1) = 1.0;
    // }
    // for (int k = 0; k < 20; ++k) {
    //     std::cout << pivot[k] << ',';
    // }
    // for (auto &leaves : Llh.get_leaves()) {
    //     if (leaves->get_target_cluster() == leaves->get_source_cluster()) {
    //         auto feuille_dense = *leaves->get_dense_data();
    //         Matrix<double> perm_local(leaves->get_target_cluster().get_size(), leaves->get_target_cluster().get_size());
    //         for (int k = 0; k < leaves->get_target_cluster().get_size(); ++k) {
    //             perm_local(pivot[k + leaves->get_target_cluster().get_offset()] - 1, k);
    //         }
    //         auto l_perm = perm_local * feuille_dense;
    //         Llh.get_block(leaves->get_target_cluster().get_size(), leaves->get_target_cluster().get_size(), leaves->get_target_cluster().get_offset(), leaves->get_target_cluster().get_offset())->set_dense_data(l_perm);
    //     }
    // }
    // copy_to_dense(Llh, ll.data());

    // std::cout << std::endl;
    // // std::cout << "erreur sur L : " << normFrob(ldense - ll) / normFrob(ldense) << std::endl;
    // // std::cout << "erreur sur U : " << normFrob(udense - uu) / normFrob(udense) << std::endl;
    // // std::cout << "erreur sur les produits :" << normFrob(ldense * udense - ll * uu) / normFrob(ldense * udense) << std::endl;

    // std::cout << "erreur dense(Lh*Uh)-Ldense x Udense : " << normFrob(ref_lu - ll * uu) / normFrob(ref_lu) << std::endl;
    // std::cout << testt(2, 3, add) << std::endl;
}