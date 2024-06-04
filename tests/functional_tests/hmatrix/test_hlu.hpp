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

/// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% /////

template <typename T, typename GeneratorTestType>
bool test_hlu(int size, htool::underlying_type<T> epsilon, htool::underlying_type<T> eta) {

    bool is_error = false;

    std::cout << "_________________________________________" << std::endl;
    std::cout << "_________________________________________" << std::endl;
    std::cout << "_________________________________________" << std::endl;

    std::cout << "HLU factortisation , size =" << size << std::endl;
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
    ///////////////////
    // HMATRIX
    std::cout << "HMATRIX assemblage" << std::endl;
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');

    // build
    auto start_time_asmbl = std::chrono::high_resolution_clock::now();
    auto root_hmatrix     = hmatrix_tree_builder.build(generator);
    auto end_time_asmbl   = std::chrono::high_resolution_clock::now();
    auto duration_asmbl   = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_asmbl - start_time_asmbl).count();
    Matrix<double> ref(size, size);
    copy_to_dense(root_hmatrix, ref.data());
    auto comprr = root_hmatrix.get_compression();

    auto er_b = normFrob(reference_num_htool - ref) / normFrob(reference_num_htool);
    std::cout << "assemblage ok, compression:" << root_hmatrix.get_compression() << "  , time ;  " << duration_asmbl << std::endl;

    ////////////////////

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    /// Pour l'instant pas de pivot : il faut appliquer la permutaétion de dgetrf a la matrice

    //////////////
    /// Hmatrix in the good permutation
    /////////////
    std::cout << " Facrotisation dgetrf pourt avoir le pivot" << std::endl;
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
    std::cout << "Uh asembled, error relative Uh-U : " << normFrob(U - udense) / normFrob(U) << std::endl;

    // Hmatrice dans la bonne permutation = Lh*Uh
    // on appelle HLU sur (Lh*Uh) et on aimerait retomber sur Lh et Uh
    Lh.set_epsilon(epsilon);
    Uh.set_epsilon(epsilon);
    auto produit_LU = Lh.hmatrix_triangular_product(Uh, 'L', 'U');
    Matrix<double> temp(size, size);
    Matrix<double> tempp(size, size);
    Matrix<double> ref_lu(size, size);
    copy_to_dense(produit_LU, ref_lu.data());
    auto ref_compr = produit_LU.get_compression();
    std::cout << "Compression Hmatrix dans la bonne numérotation (Lh*Uh ) : " << produit_LU.get_compression() << "   et sans le pivot :    " << root_hmatrix.get_compression() << std::endl;

    /// On initialise les matrices Llh et Uuh avec le bon block cluster tree
    produit_LU.set_epsilon(epsilon);
    HMatrix<double, double> Llh(root_cluster_1, root_cluster_1);
    Llh.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
    Llh.set_epsilon(epsilon);
    Llh.set_low_rank_generator(root_hmatrix.get_low_rank_generator());

    HMatrix<double, double> Uuh(root_cluster_1, root_cluster_1);
    Uuh.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
    Uuh.set_epsilon(epsilon);
    Uuh.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
    Uuh.copy_zero(produit_LU);
    Llh.copy_zero(produit_LU);
    auto start_time_lu = std::chrono::high_resolution_clock::now();

    ///////////////////////
    ////// HLU no pivot
    //////////////////////
    HLU_noperm(produit_LU, *root_cluster_1, Llh, Uuh);
    auto end_time_lu = std::chrono::high_resolution_clock::now();
    auto duration_lu = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_lu - start_time_lu).count();

    Matrix<double> uuu(size, size);
    Matrix<double> lll(size, size);
    std::cout << "Passage en dense pour tester l'erreur" << std::endl;
    copy_to_dense(Llh, lll.data());
    std::cout << "l ok" << std::endl;
    copy_to_dense(Uuh, uuu.data());
    std::cout << "u ok " << std::endl;
    auto erlu = normFrob(lll * uuu - L * U) / normFrob(L * U);
    std::cout << "erreur dense(HLU[0])*dense(HLU[1]) -reference  :  : " << erlu << std::endl;
    auto lhuh = Llh.hmatrix_triangular_product(Uuh, 'L', 'U');
    Matrix<double> lluu(size, size);
    copy_to_dense(lhuh, lluu.data());
    std::cout << "erreur dense(HLU[0]*HLU[1]) -reference  :  : " << normFrob(lluu - (L * U)) / normFrob(L * U) << std::endl;

    auto compru = Uuh.get_compression();
    auto comprl = Llh.get_compression();
    std::cout << "compression des L et U de dgetrf au format hiérarchique     " << Lh.get_compression() << ',' << Uh.get_compression() << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Time HLU ------------------------------>" << duration_lu << std::endl;
    std::cout << "compression des L et U de  HLU :------------------------>" << comprl + compru << " sur une Hmatrix compressée de : " << ref_compr << std::endl;

    // std::cout << "stabilité avec HMAT profdd , erreur relative perm_reference-dense (Lh*Uh):" << normFrob(lluu - ref_lu) / normFrob(ref_lu) << std::endl;

    auto rand   = generate_random_vector(size);
    auto ytempl = L * rand;
    std::vector<double> resl(L.nb_rows());
    Llh.hmatrix_vector_triangular_L(resl, ytempl, Llh.get_target_cluster(), 0);
    std::cout << "erreur forward : " << norm2(rand - resl) / norm2(resl) << std::endl;

    auto ytempu = U * rand;
    std::vector<double> resu(U.nb_rows());
    Uuh.hmatrix_vector_triangular_U(resu, ytempu, Llh.get_target_cluster(), 0);
    std::cout << "erreur backward: " << norm2(rand - resu) / norm2(resl) << std::endl;

    // std::vector<double> test_lu(size);
    // Llh.solve_LU(Llh, Uuh, ytemp, test_lu);
    // std::cout << "erreur solve LU :  ---------------------------------> " << norm2(rand - test_lu) / norm2(rand) << std::endl;
    auto ytemp      = (L * U) * rand;
    auto res_triang = Llh.solve_LU_triangular(Llh, Uuh, ytemp);
    std::cout << "erreur solve LU avec Lapack:  ---------------------------------> " << norm2(rand - res_triang) / norm2(rand) << std::endl;

    ////////////////////////////////////////////////
    // mini test pour triangular solver
    /// la matrice qu'on cherche a retrouver :
    auto aa     = reference_num_htool * reference_num_htool;
    auto output = L * aa;
    auto alpha  = 1.0;
    triangular_matrix_matrix_solve('L', 'L', 'N', 'U', alpha, L, output);
    std::cout << "erreur triagulaire : " << normFrob(aa - output) / normFrob(aa) << std::endl;
    return is_error;
}
