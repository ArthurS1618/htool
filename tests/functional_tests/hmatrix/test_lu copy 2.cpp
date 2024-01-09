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

void forward_substitution(const Matrix<double> &L, const std::vector<double> &y, std::vector<double> &x, int offset) {
    int n = L.nb_cols();

    for (int i = offset; i < n; ++i) {
        double sum = 0.0;
        for (int j = offset; j < i; ++j) {
            sum += L(i - offset, j - offset) * x[j];
        }
        x[i] = (y[i] - sum);
    }
}

void backward_substitution(const Matrix<double> &U, const std::vector<double> &y, std::vector<double> &x, int offset) {
    int n = U.nb_cols();

    for (int i = n - 1 + offset; i >= offset; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n + offset; ++j) {
            sum += U(i - offset, j - offset) * x[j];
        }
        x[i] = (y[i] - sum) / U(i - offset, i - offset);
    }
}

////////////////////////////////////////////////////////
////////VIRTUAL GENERATOR POUR HTOOL
///////////////////////////////////////////
template <class CoefficientPrecision, class CoordinatePrecision>
class MatGenerator : public VirtualGenerator<CoefficientPrecision> {
  private:
    Matrix<CoefficientPrecision> mat;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;

  public:
    MatGenerator(Matrix<CoefficientPrecision> &mat0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : mat(mat0), target(target0), source(source0){};

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
///////////////////////////////////////////////////////////////////
int main() {
    std::cout << "TEST HLU" << std::endl;
    std::cout << "____________________________" << std::endl;
    int size = std::pow(2, 9);

    //////////////////////
    // OK
    ////////////////

    std::cout << "_________________________________________" << std::endl;
    std::cout << "HMATRIX" << std::endl;
    double epsilon = 1e-8;
    double eta     = 10;

    ////// GENERATION MAILLAGE
    std::vector<double> p1(3 * size), p2(3 * size), p3(3 * size);
    create_disk(3, 0., size, p1.data());
    create_disk(3, 0.5, size, p2.data());
    create_disk(3, 1., size, p3.data());
    std::cout << "création cluster ok" << std::endl;
    //////////////////////
    // Clustering
    //////////////////////
    ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> recursive_build_strategy_1(size, 3, p1.data(), 2, 2);
    ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> recursive_build_strategy_2(size, 3, p2.data(), 2, 2);
    ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> recursive_build_strategy_3(size, 3, p3.data(), 2, 2);

    std::shared_ptr<Cluster<double>> root_cluster_1 = std::make_shared<Cluster<double>>(recursive_build_strategy_1.create_cluster_tree());
    std::shared_ptr<Cluster<double>> root_cluster_2 = std::make_shared<Cluster<double>>(recursive_build_strategy_2.create_cluster_tree());
    std::shared_ptr<Cluster<double>> root_cluster_3 = std::make_shared<Cluster<double>>(recursive_build_strategy_3.create_cluster_tree());
    std::cout << "Cluster tree ok" << std::endl;
    // Generator
    GeneratorTestDouble generator_LU(3, size, size, p2, p1, root_cluster_2, root_cluster_1);

    // References
    // Elles sont donnée dans la numérotation Htool -> erreur c'est hmat - reference et pas hmat -perm(reference)
    Matrix<double> dense_matrix_LU(root_cluster_2->get_size(), root_cluster_1->get_size()), dense_matrix_32(root_cluster_3->get_size(), root_cluster_2->get_size());
    generator_LU.copy_submatrix(dense_matrix_LU.nb_rows(), dense_matrix_LU.nb_cols(), 0, 0, dense_matrix_LU.data());

    HMatrixTreeBuilder<double, double> hmatrix_tree_builder_LU(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');

    // build
    auto root_hmatrix_LU = hmatrix_tree_builder_LU.build(generator_LU);

    Matrix<double> lu(size, size);
    copy_to_dense(root_hmatrix_LU, lu.data());

    std::cout << "erreur d'approximation Hmatrices " << normFrob(dense_matrix_LU - lu) / normFrob(dense_matrix_LU) << std::endl;

    //////////////////
    // HMATRIX TRIANGULAIRES
    // Matrix<double> zero(root_cluster_1->get_size(), root_cluster_1->get_size());
    // DenseGenerator<double> Z(zero);
    // HMatrixTreeBuilder<double, double> hmatrix_tree_builder_Zl(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    // auto Lh = hmatrix_tree_builder_Zl.build(Z);
    HMatrix<double, double> Lh(root_cluster_1, root_cluster_1);
    Lh.Get_L(root_hmatrix_LU);

    Matrix<double> ldense(size, size), udense(size, size);
    copy_to_dense(Lh, ldense.data());
    std::cout << "Lh dense : " << std::endl;

    // for (int k = 0; k < 10; ++k) {
    //     for (int l = 0; l < 10; ++l) {
    //         std::cout << ldense(k, l) << ',';
    //     }
    //     std::cout << std::endl;
    // }
    HMatrix<double, double> Uh(root_cluster_1, root_cluster_1);
    Uh.Get_U(root_hmatrix_LU);
    copy_to_dense(Uh, udense.data());

    std::cout << "Uh dense : " << std::endl;
    Uh.set_epsilon(epsilon);
    Lh.set_epsilon(epsilon);
    // auto &ru = root_hmatrix_LU;
    // auto &rl = root_hmatrix_LU;
    // ru.Moins(Uh);
    // Matrix<double> ru_dense(size, size);
    // copy_to_dense(ru, ru_dense.data());
    // auto moinu = dense_matrix_LU - udense;
    // std::cout << normFrob(moinu - ru_dense) / normFrob(moinu) << std::endl;
    // std::cout << "avant le moins " << normFrob(dense_matrix_LU) << " moins : " << normFrob(udense) << "aprés le moins : " << normFrob(ru_dense) << std::endl;
    // rl.Moins(Lh);
    // Matrix<double> rl_dense(size, size);
    // copy_to_dense(rl, rl_dense.data());
    // auto moinl = lu - ldense;
    // std::cout << normFrob(moinl - rl_dense) / normFrob(moinl) << std::endl;

    // std::cout << normFrob(udense) << std::endl;
    // for (int k = 10; k < 20; ++k) {
    //     for (int l = 40; l < 70; ++l) {
    //         std::cout << udense(k, l) - lu(k, l) << ',';
    //     }
    //     std::cout << std::endl;
    // }
    // auto matrice_test = ldense * udense;

    // VirtualGeneratorNOPermutation<double> virt(matrice_test, 0, 0);
    // int M = 10;
    // int N = 10;
    // std::vector<double> data(M * N);
    // int of_t = 30;
    // int of_s = 10;

    // virt.copy_submatrix(M, N, of_t, of_s, data.data());
    // // for (int k = 0; k < M; ++k) {
    // //     for (int l = 0; l < N; ++l) {
    // //         std::cout << data[k + M * l] - matrice_test(k + of_t, l + of_s) << "   ,     ";
    // //     }
    // //     std::cout << std::endl;
    // // }
    // Lh.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // Lh.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // Uh.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // Uh.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // HMatrix<double, double> hmatrice_test = Lh.hmatrix_product(Uh);
    // Matrix<double> hmatrice_test_dense(size, size);
    // Matrix<double> hmatrice_test_moins(size, size);

    // // hmatrice_test.Moins(Lh);
    // // copy_to_dense(hmatrice_test, hmatrice_test_dense.data());
    // std::cout << normFrob(hmatrice_test_dense) << ',' << normFrob(ldense * udense - hmatrice_test_dense) / normFrob(ldense * udense) << std::endl;
    // copy_to_dense(hmatrice_test, hmatrice_test_moins.data());
    // auto t = hmatrice_test_dense - ldense;
    // std::cout << normFrob(t - hmatrice_test_moins) / normFrob(t) << std::endl;

    // for (int k = 0; k < 10; ++k) {
    //     for (int l = 100; l < 110; ++l) {
    //         std::cout << udense(k, l) << '!' << lu(k, l) << "  ,  ";
    //     }
    //     std::cout << std::endl;
    // }
    ////////////////////////////////////////////
    ///// FORWARD LX =Y
    //////////////////////////////////////

    std::cout << "_____________________" << std::endl;
    std::cout << "  ++ FORWARD ++  Lx = y " << std::endl;
    std::cout << "_____________________" << std::endl;

    auto xl  = generate_random_vector(size);
    auto xxl = xl;
    auto yl  = ldense * xl;
    std::vector<double> test_prod(size);
    Lh.add_vector_product('N', 1.0, xxl.data(), 1.0, test_prod.data());
    std::cout << "erreur matrice vecteur :" << norm2(yl - test_prod) << std::endl;
    std::vector<double> res_forward(size);

    Lh.forward_substitution(*root_cluster_1, res_forward, yl);
    std::cout << "erreur forward: " << norm2(xl - res_forward) / norm2(xl) << std::endl;

    auto &t = ((root_cluster_1->get_children()[1])->get_children())[0];
    auto &s = ((root_cluster_1->get_children()[1])->get_children())[1];

    Matrix<double> Lrestr(t->get_size(), t->get_size());
    for (int k = 0; k < t->get_size(); ++k) {
        for (int l = 0; l < t->get_size(); ++l) {
            Lrestr(k, l) = ldense(k + t->get_offset(), l + t->get_offset());
        }
    }
    auto xl_restr_test = generate_random_vector(t->get_size());
    auto yl_restr_test = Lrestr * xl_restr_test;
    std::vector<double> xl_restr(size);
    std::vector<double> yl_restr(size);
    for (int k = 0; k < t->get_size(); ++k) {
        xl_restr[k + t->get_offset()] = xl_restr_test[k];
        yl_restr[k + t->get_offset()] = yl_restr_test[k];
    }
    std::vector<double> res_forward_restr(size);
    Lh.forward_substitution(*t, res_forward_restr, yl_restr);
    std::cout << "erreur forward sur un bloc: " << norm2(xl_restr - res_forward_restr) / norm2(xl_restr) << std::endl;
    /////////////////////////////////////////////////////////////
    ///////////// TEST FORWARD_T  x^T U = y^t
    ////////////////////////////////////
    std::cout << "_____________________" << std::endl;
    std::cout << "  ++ FORWARD_T ++   x^T U = y^T " << std::endl;
    std::cout << "_____________________" << std::endl;

    auto xu      = generate_random_vector(size);
    auto xxu     = xu;
    auto utransp = udense.transp(udense);
    auto yu      = utransp * xu;

    std::vector<double> res_forward_T(size);
    Uh.forward_substitution_T(*root_cluster_1, res_forward_T, yu);
    std::cout << "erreur forward_T: " << norm2(xu - res_forward_T) / norm2(xu) << std::endl;

    Matrix<double> Urestr(t->get_size(), t->get_size());
    for (int k = 0; k < t->get_size(); ++k) {
        for (int l = 0; l < t->get_size(); ++l) {
            Urestr(k, l) = udense(k + t->get_offset(), l + t->get_offset());
        }
    }
    auto xu_restr_test = generate_random_vector(t->get_size());
    auto transptt      = Urestr.transp(Urestr);
    auto yu_restr_test = transptt * xu_restr_test;
    std::vector<double> xu_restr(size);
    std::vector<double> yu_restr(size);
    for (int k = 0; k < t->get_size(); ++k) {
        xu_restr[k + t->get_offset()] = xu_restr_test[k];
        yu_restr[k + t->get_offset()] = yu_restr_test[k];
    }
    std::vector<double> res_forward_T_restr(size);
    Uh.forward_substitution_T(*t, res_forward_T_restr, yu_restr);
    std::cout << "erreur forward_T su un bloc: " << norm2(xu_restr - res_forward_T_restr) / norm2(xu_restr) << std::endl;

    /////////////////////////
    ////// TEST FORWARD_M    LX = Y
    ////////////////////////////////:
    std::cout << "_____________________" << std::endl;
    std::cout << "  ++ FORWARD_M ++   LX = Y " << std::endl;
    std::cout << "_____________________" << std::endl;
    Lh.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    Lh.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    auto Lprod = Lh.hmatrix_product(root_hmatrix_LU);
    Matrix<double> Lprod_dense(size, size);
    copy_to_dense(Lprod, Lprod_dense.data());
    std::cout << " erreur produit mat mat " << normFrob(ldense * lu - Lprod_dense) / normFrob(ldense * lu) << std::endl;
    HMatrix<double, double> Xl(root_cluster_1, root_cluster_1);
    Xl.copy_zero(Lprod);
    Forward_Matrix(Lh, *root_cluster_1, *root_cluster_1, Lprod, Xl);
    Matrix<double> Xl_dense(size, size);
    copy_to_dense(Xl, Xl_dense.data());
    std::cout << "erreur Forward_Matrix :" << normFrob(lu - Xl_dense) / normFrob(lu) << std::endl;
    std::cout << "sur un bloc : " << std::endl;
    HMatrix<double, double> Lprod_restr(root_cluster_1, root_cluster_1);
    Lprod_restr.copy_zero(Lprod);
    auto Lprod_ts = Lh.get_block(t->get_size(), t->get_size(), t->get_offset(), t->get_offset())->hmatrix_product(*root_hmatrix_LU.get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset()));
    Lprod_restr.assign(Lprod_ts, *t, *s);
    HMatrix<double, double> Xl_restr_0(root_cluster_1, root_cluster_1);
    Xl_restr_0.copy_zero(Lprod);
    // HMatrix<double, double> test_restr(root_cluster_1, root_cluster_1);
    // test_restr.copy_zero(Lprod);
    // auto bloc = test_restr.get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset());
    // bloc      = &Lprod_restr;
    Forward_Matrix(Lh, *t, *s, Lprod_restr, Xl_restr_0);

    Matrix<double> XLrestr_dense(t->get_size(), s->get_size());
    copy_to_dense(*Xl_restr_0.get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset()), XLrestr_dense.data());

    Matrix<double> ref(t->get_size(), s->get_size());
    // copy_to_dense(*root_hmatrix_LU.get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset()), ref.data());
    for (int k = 0; k < t->get_size(); ++k) {
        for (int l = 0; l < s->get_size(); ++l) {
            ref(k, l) = lu(k + t->get_offset(), l + s->get_offset());
        }
    }
    std::cout << "erreur Forward_Matrix_restr :" << normFrob(ref - XLrestr_dense) / normFrob(ref) << std::endl;
    std::cout << "_____________________" << std::endl;
    std::cout << "  ++ FORWARD_M_T ++   XU = Y " << std::endl;
    std::cout << "_____________________" << std::endl;
    Uh.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    Uh.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    auto Uprod = root_hmatrix_LU.hmatrix_product(Uh);
    Uprod.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    Uprod.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    std::cout << (Uprod.get_admissibility_condition() == nullptr) << std::endl;
    Matrix<double> Uprod_dense(size, size);
    copy_to_dense(Uprod, Uprod_dense.data());
    std::cout << " erreur produit mat mat " << normFrob(lu * udense - Uprod_dense) / normFrob(lu * udense) << std::endl;
    HMatrix<double, double> Xu(root_cluster_1, root_cluster_1);
    Xu.copy_zero(Uprod);
    Forward_Matrix_T(Uh, *root_cluster_1, *root_cluster_1, Uprod, Xu);
    Matrix<double> Xu_dense(size, size);
    copy_to_dense(Xu, Xu_dense.data());
    std::cout << "erreur Forward_Matrix_T :" << normFrob(lu - Xu_dense) / normFrob(lu) << std::endl;

    std::cout << "sur un bloc : " << std::endl;
    HMatrix<double, double> Uprod_restr(root_cluster_1, root_cluster_1);
    Uprod_restr.copy_zero(Uprod);
    auto Uprod_ts = root_hmatrix_LU.get_block(s->get_size(), t->get_size(), s->get_offset(), t->get_offset())->hmatrix_product(*Uh.get_block(t->get_size(), t->get_size(), t->get_offset(), t->get_offset()));
    Uprod_restr.assign(Uprod_ts, *s, *t);
    HMatrix<double, double> Xu_restr_0(root_cluster_1, root_cluster_1);
    Xu_restr_0.copy_zero(Uprod);
    Uprod_restr.set_admissibility_condition(Uh.get_admissibility_condition());
    Uprod_restr.set_low_rank_generator(Uh.get_low_rank_generator());
    Xu_restr_0.set_admissibility_condition(Uh.get_admissibility_condition());
    Xu_restr_0.set_low_rank_generator(Uh.get_low_rank_generator());

    Forward_Matrix_T(Uh, *t, *s, Uprod_restr, Xu_restr_0);

    Matrix<double> XUrestr_dense(s->get_size(), t->get_size());
    copy_to_dense(*Xu_restr_0.get_block(s->get_size(), t->get_size(), s->get_offset(), t->get_offset()), XUrestr_dense.data());

    Matrix<double> refu(s->get_size(), t->get_size());
    for (int k = 0; k < s->get_size(); ++k) {
        for (int l = 0; l < t->get_size(); ++l) {
            refu(k, l) = lu(k + s->get_offset(), l + t->get_offset());
        }
    }
    std::cout << "erreur Forward_Matrix_restr :" << normFrob(refu - XUrestr_dense) / normFrob(refu) << std::endl;
    std::cout << "_____________________" << std::endl;

    ///////////////////////////////////////
    //// TEST H_LU
    ///////////////////////////////////
    HMatrix<double, double> Llh(root_cluster_1, root_cluster_1);
    HMatrix<double, double> Uuh(root_cluster_1, root_cluster_1);
    Llh.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    Llh.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    Uuh.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    Uuh.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    /////////////////////////////////
    // Si on prend Mh = Lh*Uh et qu'on essaye de les retrouver ca marche et on a aucun appelle tronqué.
    HMatrix<double, double> LU = Lh.hmatrix_product(Uh);
    Uuh.copy_zero(LU);
    Llh.copy_zero(LU);
    Matrix<double> ref_lu(size, size);
    copy_to_dense(LU, ref_lu.data());
    std::cout << "norme de Lh*Uh la matrice que l'on factorise  : " << normFrob(ref_lu) << std::endl;
    // // // LA MATRICE QU ON VA FACTORISER

    HMatrix_LU(LU, *root_cluster_1, Llh, Uuh);
    // FACTORISATION
    auto test_lu = Llh.hmatrix_product(Uuh);
    Matrix<double> ludense(size, size);
    copy_to_dense(test_lu, ludense.data());
    // PRODUIT Lh Uh-> DENSE

    std::cout << "erreur dense(Lh*Uh)-Ldense x Udense : " << normFrob(ludense - ldense * udense) / normFrob(ldense * udense) << std::endl;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    HMatrix<double, double> Llh2(root_cluster_1, root_cluster_1);
    HMatrix<double, double> Uuh2(root_cluster_1, root_cluster_1);
    Llh2.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    Llh2.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    Uuh2.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    Uuh2.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    auto &temp = Uh;
    temp.Moins(Lh);
    HMatrix_LU(temp, *root_cluster_1, Llh2, Uuh2);
    Matrix<double> temp_dense(size, size);
    Matrix<double> vrai_test_lu(size, size);
    auto prod2 = Llh2.hmatrix_product(Uuh2);
    copy_to_dense(prod2, vrai_test_lu.data());
    copy_to_dense(temp, temp_dense.data());
    std::cout << normFrob(temp_dense) << ',' << normFrob(vrai_test_lu) << std::endl;
    std::cout << normFrob(temp_dense - vrai_test_lu) << std::endl;

    /////////////////////////
    ////// TEST FORWARD_M_T    XU = Y
    ////////////////////////////////:

    // std::cout << "______________________" << std::endl;
    // std::cout << "Test Bacward Ux= y" << std::endl;
    // auto xu = generate_random_vector(size);
    // auto yu = udense * xu;
    // std::vector<double> res_backward(size);
    // auto ut = udense;
    // Uh.backward_substitution(*root_cluster_1, yu, res_backward);
    // std::cout << "erreur backward : " << norm2(xu - res_backward) / norm2(xu) << std::endl;

    // std::cout << "______________________" << std::endl;
    // std::cout << "Test forward_T  x^T U =y^T" << std::endl;
    // auto xtu = generate_random_vector(size);
    // auto ytu = ut.transp(ut) * xtu;
    // std::vector<double> res_forward_T(size);

    // Uh.forward_substitution_T(*root_cluster_1, ytu, res_forward_T);
    // std::cout << "erreur forward_T : " << norm2(xtu - res_forward_T) / norm2(xtu) << std::endl;

    // //////////////////////////
    // // SUR UN SOUS BLOC
    // ///////////////////
    // std::cout << "__________________________" << std::endl;
    // auto &t = ((root_cluster_1->get_children()[1])->get_children())[1];
    // std::cout << "on est sur le bloc size, offset= " << t->get_size() << ',' << t->get_offset() << std::endl;
    // Matrix<double> utt(t->get_size(), t->get_size());
    // for (int k = 0; k < t->get_size(); ++k) {
    //     for (int l = 0; l < t->get_size(); ++l) {
    //         utt(k, l) = udense(k + t->get_offset(), l + t->get_offset());
    //     }
    // }
    // auto xut = generate_random_vector(t->get_size());
    // auto yut = utt * xut;
    // std::vector<double> Yt(size);
    // for (int k = t->get_offset(); k < t->get_size() + t->get_offset(); ++k) {
    //     Yt[k] = yut[k - t->get_offset()];
    // }
    // std::vector<double> res_backwardt(size);
    // Uh.backward_substitution(*t, Yt, res_backwardt);
    // std::vector<double> x_restr(t->get_size());
    // for (int k = 0; k < t->get_size(); ++k) {
    //     x_restr[k] = res_backwardt[k + t->get_offset()];
    // }
    // std::cout << "erreur backward sur le bloc : Utt xt = yt " << norm2(xut - x_restr) / norm2(xut) << std::endl;

    // Matrix<double> ltt(t->get_size(), t->get_size());
    // for (int k = 0; k < t->get_size(); ++k) {
    //     for (int l = 0; l < t->get_size(); ++l) {
    //         ltt(k, l) = ldense(k + t->get_offset(), l + t->get_offset());
    //     }
    // }
    // auto xlt = generate_random_vector(t->get_size());
    // auto ylt = ltt * xlt;
    // std::vector<double> Y(size);
    // for (int k = 0; k < t->get_size(); ++k) {
    //     Y[k + t->get_offset()] = ylt[k];
    // }
    // std::vector<double> res_forwardt(size);
    // Lh.forward_substitution(*t, Y, res_forwardt);
    // std::vector<double> x_restr_forward(t->get_size());
    // for (int k = 0; k < t->get_size(); ++k) {
    //     x_restr_forward[k] = res_forwardt[k + t->get_offset()];
    // }
    // std::cout << "erreur forward sur le bloc : Ltt xt = yt " << norm2(xlt - x_restr_forward) / norm2(xlt) << std::endl;
    // auto xutt    = generate_random_vector(t->get_size());
    // auto transpp = utt;
    // auto yutt    = transpp.transp(transpp) * xutt;
    // std::vector<double> YT(size);
    // for (int k = 0; k < t->get_size(); ++k) {
    //     YT[k + t->get_offset()] = yutt[k];
    // }
    // std::vector<double> res_forwardTt(size);
    // Uh.forward_substitution_T(*t, YT, res_forwardTt);
    // std::vector<double> x_restr_forwardT(t->get_size());
    // for (int k = 0; k < t->get_size(); ++k) {
    //     x_restr_forwardT[k] = res_forwardTt[k + t->get_offset()];
    // }
    // std::cout << "erreur forward Tsur le bloc : xt^T U= yt^T " << norm2(xutt - x_restr_forwardT) / norm2(xutt) << std::endl;
    // std::cout << "_____________________________________" << std::endl;
    // //////////////////////////////////////////
    // ///// TEST LU x = y
    // ///////////////////////////////////////
    // auto xrd = generate_random_vector(size);
    // auto ylu = ldense * udense * xrd;
    // std::vector<double> res_lu(size);
    // Lh.solve_LU(Lh, Uh, ylu, res_lu);
    // std::cout << "erreur lu : " << norm2(xrd - res_lu) / norm2(xrd) << std::endl;

    // std::cout << "__________________________________" << std::endl;
    // ////////////////////////////////////////
    // /// TEST FORWARD_M  LX=Z
    // /////////////////////////
    // auto Lprod = Lh.hmatrix_product(root_hmatrix_LU);

    // Matrix<double> test_prod(size, size);
    // copy_to_dense(Lprod, test_prod.data());
    // std::cout << "erreur sur le produit L*dense :" << normFrob(test_prod - ldense * lu) / normFrob(ldense * lu) << std::endl;

    // HMatrix<double, double> Xl(root_cluster_1, root_cluster_1);
    // Xl.copy_zero(Lprod);
    // Forward_Matrix(Lh, *root_cluster_1, *root_cluster_1, Lprod, Xl);
    // Matrix<double> Xl_dense(size, size);
    // copy_to_dense(Xl, Xl_dense.data());
    // std::cout << "erreur Forward_Matrix :" << normFrob(lu - Xl_dense) / normFrob(lu) << std::endl;

    // std::cout << "_____________________________" << std::endl;
    // std::cout << "erreur forward_M sur un sous blocs" << std::endl;
    // auto &t1 = ((root_cluster_1->get_children()[0])->get_children())[1];
    // auto &t2 = ((root_cluster_1->get_children()[1])->get_children())[1];
    // Matrix<double> lttt(t1->get_size(), t1->get_size());
    // for (int k = 0; k < t1->get_size(); ++k) {
    //     for (int l = 0; l < t1->get_size(); ++l) {
    //         lttt(k, l) = ldense(k + t1->get_offset(), l + t1->get_offset());
    //     }
    // }
    // Matrix<double> xts(t1->get_size(), t2->get_size());
    // for (int k = 0; k < t1->get_size(); ++k) {
    //     for (int l = 0; l < t2->get_size(); ++l) {
    //         xts(k, l) = lu(k + t1->get_offset(), l + t2->get_offset());
    //     }
    // }
    // auto Lprod2 = Lh.hmatrix_product(root_hmatrix_LU);
    // auto zts    = lttt * xts;
    // HMatrix<double, double> Xrestr(root_cluster_1, root_cluster_1);
    // Xrestr.copy_zero(Lprod2);
    // HMatrix<double, double> Zrestr(root_cluster_1, root_cluster_1);
    // Zrestr.copy_zero(Lprod2);
    // Zrestr.get_block(t1->get_size(), t2->get_size(), t1->get_offset(), t2->get_offset());
    // Forward_Matrix(Lh, *t1, *t2, Lprod2, Xrestr);
    // Matrix<double> xtest(t1->get_size(), t2->get_size());
    // copy_to_dense(*Xrestr.get_block(t1->get_size(), t2->get_size(), t1->get_offset(), t2->get_offset()), xtest.data());
    // std::cout << "erreur sur le bloc Ltt Xts = Zts " << normFrob(xts - xtest) / normFrob(xts) << std::endl;

    // //////////////////////////////////////////////////////////////
    // ////// TEST FORWARD_M_T  : XU =Y
    // ///////////////////////////////////////////
    // std::cout << "_____________________________________" << std::endl;
    // auto Uprod = root_hmatrix_LU.hmatrix_product(Uh);

    // Matrix<double> test_uprod(size, size);
    // copy_to_dense(Uprod, test_uprod.data());
    // std::cout << " erreur produit : " << normFrob(test_uprod - lu * udense) / normFrob(lu * udense) << std::endl;
    // HMatrix<double, double> uX(root_cluster_1, root_cluster_1);
    // uX.copy_zero(Uprod);
    // uX.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // uX.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());

    // Forward_Matrix_T(Uh, *root_cluster_1, *root_cluster_1, Uprod, uX);
    // Matrix<double> uX_dense(size, size);
    // copy_to_dense(uX, uX_dense.data());
    // std::cout << "erreur Forward_Matrix_T :" << normFrob(lu - uX_dense) / normFrob(lu) << std::endl;

    // std::cout << "_____________________________" << std::endl;
    // std::cout << "erreur forward_M_T sur un sous blocs" << std::endl;
    // Matrix<double> uttt(t2->get_size(), t2->get_size());
    // for (int k = 0; k < t2->get_size(); ++k) {
    //     for (int l = 0; l < t2->get_size(); ++l) {
    //         uttt(k, l) = udense(k + t2->get_offset(), l + t2->get_offset());
    //     }
    // }

    // auto Uprod2 = root_hmatrix_LU.hmatrix_product(Uh);
    // auto zuts   = uttt * xts;
    // HMatrix<double, double> Xrestru(root_cluster_1, root_cluster_1);
    // Xrestru.copy_zero(Uprod2);
    // Xrestru.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // Xrestru.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // Forward_Matrix_T(Uh, *t1, *t2, Uprod2, Xrestru);
    // Matrix<double> xtestu(t1->get_size(), t2->get_size());
    // copy_to_dense(*Xrestru.get_block(t1->get_size(), t2->get_size(), t1->get_offset(), t2->get_offset()), xtestu.data());
    // std::cout << "erreur sur le bloc Ltt Xts = Zts " << normFrob(xts - xtestu) / normFrob(xts) << std::endl;

    // // ////////////////////////////////////////////////
    // // /// TEST HLU
    // // //////////////////////////////////////////////////
    // std::cout << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;

    // std::cout << "+++++++++++++++++++++++++++ HLU ++++++++++++++++++++++++++++++" << std::endl;
    // HMatrix<double, double> Llh(root_cluster_1, root_cluster_1);
    // HMatrix<double, double> Uuh(root_cluster_1, root_cluster_1);
    // Llh.copy_zero(root_hmatrix_LU);
    // Llh.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // Llh.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // Uuh.copy_zero(root_hmatrix_LU);
    // Uuh.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // Uuh.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // /////////////////////////////////
    // // Si on prend Mh = Lh*Uh et qu'on essaye de les retrouver ca marche et on a aucun appelle tronqué.
    // HMatrix<double, double> LU = Lh.hmatrix_product(Uh);
    // Matrix<double> ref_lu(size, size);
    // copy_to_dense(LU, ref_lu.data());
    // for (int k = 0; k < 10; ++k) {
    //     for (int l = 70; l < 80; ++l) {
    //         std::cout << ldense(k, l + 30) << "!" << lu(k, l + 30) << ',';
    //     }
    //     std::cout << std::endl;
    // }
    // LA MATRICE QU ON VA FACTORISER

    // hmatrix_LU(LU, *root_cluster_1, Llh, Uuh);
    // // FACTORISATION
    // auto test_lu = Llh.hmatrix_product(Uuh);
    // Matrix<double> ludense(size, size);
    // copy_to_dense(test_lu, ludense.data());
    // // PRODUIT Lh Uh-> DENSE

    // std::cout << "erreur dense(Lh*Uh)-Ldense x Udense : " << normFrob(ludense - ref_lu) / normFrob(ref_lu) << std::endl;

    // // Par contre la ca tronque dns tout les sens. on veut le blocs (16,0,16,0) alors que (64,0,64,0) c'est une feuille + les permutations du LU
    // hmatrix_LU(root_hmatrix_LU, *root_cluster_1, Llh, Uuh);
    // auto test_lu = Llh.hmatrix_product(Uuh);
    // Matrix<double> ludense(size, size);
    // copy_to_dense(test_lu, ludense.data());
    // std::cout << "erreur M-LhxUh : " << normFrob(ludense - dense_matrix_LU) / normFrob(dense_matrix_LU) << std::endl;

    //////////////////////////////////////
    // std::cout << "autr test pour etre sure :" << std::endl;
    // HMatrix<double, double> Lh0(root_cluster_1, root_cluster_1);
    // HMatrix<double, double> Uh0(root_cluster_1, root_cluster_1);
    // Lh0.copy_zero(root_hmatrix_LU);
    // Lh0.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // Lh0.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // Uh0.copy_zero(root_hmatrix_LU);
    // Uh0.set_low_rank_generator(root_hmatrix_LU.get_low_rank_generator());
    // Uh0.set_admissibility_condition(root_hmatrix_LU.get_admissibility_condition());
    // hmatrix_LU(root_hmatrix_LU, *root_cluster_1, Lh0, Uh0);
    // auto lu0 = Lh0.hmatrix_product(Uh0);
    // Matrix<double> lu0_dense(size, size);
    // copy_to_dense(lu0, lu0_dense.data());
    // std::cout << normFrob(dense_matrix_LU - lu0_dense) / normFrob(dense_matrix_LU) << std::endl;
}