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

    // std::cout << "HLU factortisation , size =" << size << std::endl;
    ///////////////////////
    ////// GENERATION MAILLAGE  :  size points en 3 dimensions
    std::vector<double> p1(3 * size);
    create_disk(3, 0.0, size, p1.data());

    // std::cout << "création cluster ok" << std::endl;
    //////////////////////

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    //////////////////////
    // Clustering
    ClusterTreeBuilder<double> recursive_build_strategy_1;
    std::shared_ptr<Cluster<double>> root_cluster_1 = std::make_shared<Cluster<double>>(recursive_build_strategy_1.create_cluster_tree(size, 3, p1.data(), 2, 2));
    // std::cout << "Cluster tree ok" << std::endl;
    //////////////////////

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    ///////////////////
    // Generator: donné en argument de la méthode
    auto permutation = root_cluster_1->get_permutation();
    // GeneratorTestType generator;

    Matrix<double> reference_num_htool(size, size);
    GeneratorTestType.copy_submatrix(size, size, 0, 0, reference_num_htool.data());

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    ///////////////////
    // HMATRIX
    // std::cout << "HMATRIX assemblage" << std::endl;
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');

    // build
    auto start_time_asmbl = std::chrono::high_resolution_clock::now();
    auto root_hmatrix     = hmatrix_tree_builder.build(generator);
    auto end_time_asmbl   = std::chrono::high_resolution_clock::now();
    auto duration_asmbl   = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_asmbl - start_time_asmbl).count();
    Matrix<double> ref(size, size);
    copy_to_dense(root_hmatrix, ref.data());
    auto comprr = root_hmatrix.get_compression();

    // auto er_b = normFrob(reference_num_htool - ref) / normFrob(reference_num_htool);
    // // std::cout << "assemblage ok, compression:" << root_hmatrix.get_compression() << "  , time ;  " << duration_asmbl << std::endl;

    // ////////////////////

    // //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    // /// Pour l'instant pas de pivot : il faut appliquer la permutaétion de dgetrf a la matrice

    // //////////////
    // /// Hmatrix in the good permutation
    // /////////////
    // // std::cout << " Facrotisation dgetrf pourt avoir le pivot" << std::endl;
    // Matrix<double> L(size, size);
    // Matrix<double> U(size, size);
    // std::vector<int> ipiv(size, 0.0);
    // int info = -1;
    // auto A   = reference_num_htool;
    // Lapack<double>::getrf(&size, &size, A.data(), &size, ipiv.data(), &info);

    // for (int i = 0; i < size; ++i) {
    //     L(i, i) = 1;
    //     U(i, i) = A(i, i);

    //     for (int j = 0; j < i; ++j) {
    //         L(i, j) = A(i, j);
    //         U(j, i) = A(j, i);
    //     }
    // }
    // // std::cout << " Pivot obtenue" << std::endl;

    // // // Matrice Lh et Uh
    // /// Warning : on peut pas juste donné L et U parce que Htool fais des permutations et la matrice qu'il compresserait serait pas forcemment triangtulaire
    // /// IL faut mettre L et U dans la permutation inverse de clusters
    // // std::cout << " Assemblage Lh et Uh " << std::endl;
    // // std::cout << "L et U doivent être formatées pour que htool produise des matrices triangulaires" << std::endl;
    // auto ll = get_unperm_mat(L, *root_cluster_1, *root_cluster_1);
    // Matricegenerator<double, double> l_generator(ll, *root_cluster_1, *root_cluster_1);
    // HMatrixTreeBuilder<double, double> lh_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    // auto Lh = lh_builder.build(l_generator);
    // Matrix<double> ldense(size, size);
    // copy_to_dense(Lh, ldense.data());
    // // std::cout << "Lh assembled, error relative Lh-L  :" << normFrob(L - ldense) / normFrob(L) << std::endl;

    // auto uu = get_unperm_mat(U, *root_cluster_1, *root_cluster_1);
    // Matricegenerator<double, double> u_generator(uu, *root_cluster_1, *root_cluster_1);
    // HMatrixTreeBuilder<double, double> uh_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    // auto Uh = uh_builder.build(u_generator);
    // Matrix<double> udense(size, size);
    // copy_to_dense(Uh, udense.data());
    // // std::cout << "Uh asembled, error relative Uh-U : " << normFrob(U - udense) / normFrob(U) << std::endl;

    // // Hmatrice dans la bonne permutation = Lh*Uh
    // // on appelle HLU sur (Lh*Uh) et on aimerait retomber sur Lh et Uh
    // Lh.set_epsilon(epsilon);
    // Uh.set_epsilon(epsilon);
    // auto produit_LU = Lh.hmatrix_triangular_product(Uh, 'L', 'U');
    // Matrix<double> temp(size, size);
    // Matrix<double> tempp(size, size);
    // Matrix<double> ref_lu(size, size);
    // copy_to_dense(produit_LU, ref_lu.data());
    // auto ref_compr = produit_LU.get_compression();
    // // std::cout << "Compression Hmatrix dans la bonne numérotation (Lh*Uh ) : " << produit_LU.get_compression() << "   et sans le pivot :    " << root_hmatrix.get_compression() << std::endl;

    // /// On initialise les matrices Llh et Uuh avec le bon block cluster tree
    // produit_LU.set_epsilon(epsilon);
    // HMatrix<double, double> Llh(root_cluster_1, root_cluster_1);
    // Llh.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
    // Llh.set_epsilon(epsilon);
    // Llh.set_low_rank_generator(root_hmatrix.get_low_rank_generator());

    // HMatrix<double, double> Uuh(root_cluster_1, root_cluster_1);
    // Uuh.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
    // Uuh.set_epsilon(epsilon);
    // Uuh.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
    // Uuh.copy_zero(produit_LU);
    // Llh.copy_zero(produit_LU);
    // auto start_time_lu = std::chrono::high_resolution_clock::now();

    // ///////////////////////
    // ////// HLU no pivot
    // //////////////////////
    // HLU_noperm(produit_LU, *root_cluster_1, Llh, Uuh);
    // auto end_time_lu = std::chrono::high_resolution_clock::now();
    // auto duration_lu = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_lu - start_time_lu).count();

    // Matrix<double> uuu(size, size);
    // Matrix<double> lll(size, size);
    // // std::cout << "Passage en dense pour tester l'erreur" << std::endl;
    // // copy_to_dense(Llh, lll.data());
    // // std::cout << "l ok" << std::endl;
    // // copy_to_dense(Uuh, uuu.data());
    // // std::cout << "u ok " << std::endl;
    // // auto erlu = normFrob(lll * uuu - L * U) / normFrob(L * U);
    // // std::cout << "erreur dense(HLU[0])*dense(HLU[1]) -reference  :  : " << erlu << std::endl;
    // // auto lhuh = Llh.hmatrix_triangular_product(Uuh, 'L', 'U');
    // // Matrix<double> lluu(size, size);
    // // copy_to_dense(lhuh, lluu.data());
    // // std::cout << "erreur dense(HLU[0]*HLU[1]) -reference  :  : " << normFrob(lluu - (L * U)) / normFrob(L * U) << std::endl;

    // // auto compru = Uuh.get_compression();
    // // auto comprl = Llh.get_compression();
    // // std::cout << "compression des L et U de dgetrf au format hiérarchique     " << Lh.get_compression() << ',' << Uh.get_compression() << std::endl;
    // // std::cout << std::endl;
    // // std::cout << std::endl;
    // // std::cout << "Time HLU ------------------------------>" << duration_lu << std::endl;
    // // std::cout << "compression des L et U de  HLU :------------------------>" << comprl + compru << " sur une Hmatrix compressée de : " << ref_compr << std::endl;

    // // std::cout << "stabilité avec HMAT profdd , erreur relative perm_reference-dense (Lh*Uh):" << normFrob(lluu - ref_lu) / normFrob(ref_lu) << std::endl;

    // auto rand = generate_random_vector(size);
    // // auto ytempl = L * rand;
    // // std::vector<double> resl(L.nb_rows());
    // // Llh.hmatrix_vector_triangular_L(resl, ytempl, Llh.get_target_cluster(), 0);
    // // // std::cout << "erreur forward : " << norm2(rand - resl) / norm2(resl) << std::endl;

    // // auto ytempu = U * rand;
    // // std::vector<double> resu(U.nb_rows());
    // // Uuh.hmatrix_vector_triangular_U(resu, ytempu, Llh.get_target_cluster(), 0);
    // // // std::cout << "erreur backward: " << norm2(rand - resu) / norm2(resl) << std::endl;

    // // std::vector<double> test_lu(size);
    // // Llh.solve_LU(Llh, Uuh, ytemp, test_lu);
    // // std::cout << "erreur solve LU :  ---------------------------------> " << norm2(rand - test_lu) / norm2(rand) << std::endl;
    // auto ytemp      = (L * U) * rand;
    // auto res_triang = Llh.solve_LU_triangular(Llh, Uuh, ytemp);
    // std::cout << "_________________________________________" << std::endl;
    // std::cout << " Size : -------------------------------> " << size << std::endl;
    // std::cout << " Time : -------------------------------> " << duration_lu << std::endl;
    // std::cout << " Error :-------------------------------> " << norm2(rand - res_triang) / norm2(rand) << std::endl;
    // std::cout << " Compression :-------------------------> " << Lh.get_compression() + Uh.get_compression() << std::endl;
    // std::cout << "_________________________________________" << std::endl;
    // std::cout << "_________________________________________" << std::endl;
    // double compr1, compr0;
    // for (auto &l : Lh.get_leaves()) {
    //     if (l->get_target_cluster().get_offset() >= l->get_source_cluster().get_offset()) {
    //         if (l->is_dense()) {
    //             compr0 = compr0 + 1.0 * (l->get_target_cluster().get_size() * l->get_source_cluster().get_size());
    //         } else {
    //             compr0 = compr0 + l->get_low_rank_data()->rank_of() * 1.0 * (l->get_target_cluster().get_size() + l->get_source_cluster().get_size());
    //         }
    //     }
    // }

    // compr0 = compr0 / (0.5 * (Lh.get_target_cluster().get_size() * Lh.get_source_cluster().get_size()));
    // compr1 = 1.000 - compr0;
    // std::cout << "vraiment ??" << compr1 << std::endl;

    // ///////////////////// tests sans pivot ?///
    HMatrix<double, double> L0(root_cluster_1, root_cluster_1);
    L0.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
    L0.set_epsilon(epsilon);
    L0.set_low_rank_generator(root_hmatrix.get_low_rank_generator());

    HMatrix<double, double> U0(root_cluster_1, root_cluster_1);
    U0.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
    U0.set_epsilon(epsilon);
    U0.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
    U0.copy_zero(root_hmatrix);
    L0.copy_zero(root_hmatrix);
    auto start_time_lu0 = std::chrono::high_resolution_clock::now();
    HLU_noperm(root_hmatrix, *root_cluster_1, L0, U0);
    auto end_time_lu0 = std::chrono::high_resolution_clock::now();
    auto duration_lu0 = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_lu0 - start_time_lu0).count();

    // HLU_noperm(root_hmatrix, *root_cluster_1, L0, U0);
    auto rand   = generate_random_vector(size);
    auto y0     = reference_num_htool * rand;
    auto res0   = L0.solve_LU_triangular(L0, U0, y0);
    auto err    = norm2(rand - res0) / norm2(rand);
    double cpt1 = 0;
    double cpt2 = 0;
    for (auto &l : L0.get_leaves()) {
        if (l->get_target_cluster().get_offset() >= l->get_source_cluster().get_offset()) {
            if (l->is_dense()) {
                cpt1 = 1.0 * cpt1 + 1.0 * (l->get_target_cluster().get_size() * l->get_source_cluster().get_size());
            } else {
                // cpt1 = cpt1 + l->get_low_rank_data()->Get_U().nb_cols() * 1.0 * (l->get_target_cluster().get_size() + l->get_source_cluster().get_size());
                if (l->get_low_rank_data()->rank_of() > 0) {
                    cpt1 = cpt1 + l->get_low_rank_data()->rank_of() * 1.0 * (l->get_target_cluster().get_size() + l->get_source_cluster().get_size());
                } else {
                    cpt1 = cpt1 + 1.0 * (l->get_target_cluster().get_size() * l->get_source_cluster().get_size());
                }
            }
        }
    }
    for (auto &l : U0.get_leaves()) {
        if (l->get_target_cluster().get_offset() <= l->get_source_cluster().get_offset()) {
            if (l->is_dense()) {
                cpt1 = 1.0 * cpt1 + 1.0 * (l->get_target_cluster().get_size() * l->get_source_cluster().get_size());
            } else {
                // cpt1 = cpt1 + l->get_low_rank_data()->Get_U().nb_cols() * 1.0 * (l->get_target_cluster().get_size() + l->get_source_cluster().get_size());
                if (l->get_low_rank_data()->rank_of() > 0) {
                    cpt1 = cpt1 + l->get_low_rank_data()->rank_of() * 1.0 * (l->get_target_cluster().get_size() + l->get_source_cluster().get_size());
                } else {
                    cpt1 = cpt1 + 1.0 * (l->get_target_cluster().get_size() * l->get_source_cluster().get_size());
                }
            }
        }
    }
    cpt1 = cpt1 / ((L0.get_target_cluster().get_size() * L0.get_source_cluster().get_size()));
    cpt2 = 1.000 - cpt1;

    // double cpt11 = 0;
    // double cpt21 = 0;
    // for (auto &l : U0.get_leaves()) {
    //     if (l->get_target_cluster().get_offset() <= l->get_source_cluster().get_offset()) {
    //         if (l->is_dense()) {
    //             cpt11 = cpt11 + 1.0 * (l->get_target_cluster().get_size() * l->get_source_cluster().get_size());
    //         } else {
    //             if (l->get_low_rank_data()->rank_of() > 0) {
    //                 cpt11 = cpt11 + l->get_low_rank_data()->rank_of() * 1.0 * (l->get_target_cluster().get_size() + l->get_source_cluster().get_size());
    //             } else {
    //                 std::cout << "!" << std::endl;
    //                 cpt11 = cpt11 + 1.0 * (l->get_target_cluster().get_size() * l->get_source_cluster().get_size());
    //             }
    //             // cpt11 = cpt11 + l->get_low_rank_data()->Get_U().nb_cols() * 1.0 * (l->get_target_cluster().get_size() + l->get_source_cluster().get_size());
    //         }
    //     }
    // }

    // cpt11 = cpt11 / (2 * (U0.get_target_cluster().get_size() * U0.get_source_cluster().get_size()));
    // cpt21 = 1.000 - cpt11;
    // // double compression = L0.get_compression() + U0.get_compression();
    // double cc = cpt2 + cpt21;
    std::cout << "compression de A " << comprr << std::endl;
    std::cout << "____________________________________________" << std::endl;
    std::cout << " Size : -------------------------------> " << size << std::endl;
    std::cout << " Time : -------------------------------> " << duration_lu0 << std::endl;
    std::cout << " Compression :-------------------------> " << cpt2 << std::endl;
    std::cout << " Error :-------------------------------> " << err << std::endl;
    std::cout << "____________________________________________" << std::endl;
    std::cout << std::endl;

    // std::vector<double> res_vec(4);
    // res_vec[0] = size;
    // res_vec[1] = duration_lu0;
    // res_vec[2] = cpt2;
    // res_vec[3] = err;
    // return res_vec;

    return is_error;
}
