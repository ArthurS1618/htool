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

std::vector<double> generate_random_vector(int n) {
    // Initialiser le générateur de nombres aléatoires avec une graine basée sur l'horloge système
    std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());

    // Définir une distribution uniforme entre 0 et 1
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    // Générer les nombres aléatoires et les stocker dans un vecteur
    std::vector<double> random_vector(n);
    for (int i = 0; i < n; i++) {
        random_vector[i] = distribution(generator);
    }
    return random_vector;
}

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
int main() {
    std::cout << "_________________________________________" << std::endl;
    double epsilon = 1e-3;
    double eta     = 10;
    // int size       = 2048;
    int size = std::pow(2, 9);

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

    GeneratorTestDouble generator_21(3, size, size, p2, p1, root_cluster_2, root_cluster_1);
    GeneratorTestDouble generator_32(3, size, size, p3, p2, root_cluster_3, root_cluster_2);

    // References
    // Elles sont donnée dans la numérotation Htool -> erreur c'est hmat - reference et pas hmat -perm(reference)
    Matrix<double> dense_matrix_21(root_cluster_2->get_size(), root_cluster_1->get_size()), dense_matrix_32(root_cluster_3->get_size(), root_cluster_2->get_size());
    generator_21.copy_submatrix(dense_matrix_21.nb_rows(), dense_matrix_21.nb_cols(), 0, 0, dense_matrix_21.data());
    generator_32.copy_submatrix(dense_matrix_32.nb_rows(), dense_matrix_32.nb_cols(), 0, 0, dense_matrix_32.data());
    std::cout << "generator ok" << std::endl;

    // HMatrix

    HMatrixTreeBuilder<double, double> hmatrix_tree_builder_21(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder_32(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');

    HMatrixTreeBuilder<double, double> hmatrix_tree_builder_L(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder_U(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');

    // build
    auto root_hmatrix_21 = hmatrix_tree_builder_21.build(generator_21);
    auto root_hmatrix_32 = hmatrix_tree_builder_32.build(generator_32);
    std::cout << "hmat build ok " << std::endl;

    // mise au format dense
    Matrix<double> dense_32(size, size);
    Matrix<double> dense_21(size, size);
    copy_to_dense(root_hmatrix_21, dense_21.data());
    copy_to_dense(root_hmatrix_32, dense_32.data());

    // info compression et erreur
    std::cout << "info 32 " << std::endl;
    std::cout << "copression " << root_hmatrix_32.get_compression() << std::endl;
    std::cout << "erreur : " << normFrob(dense_32 - dense_matrix_32) << std::endl;
    std::cout << "info 21 " << std::endl;
    std::cout << "compression " << root_hmatrix_21.get_compression() << std::endl;
    std::cout << "erreur" << normFrob(dense_21 - dense_matrix_21) << std::endl;

    std::vector<int> ipiv(size, 0.0);
    int info = -1;
    Matrix<double> test(size, size);
    for (int k = 0; k < size; ++k) {
        test(k, k) = 1;
    }
    Lapack<double>::getrf(&size, &size, test.data(), &size, ipiv.data(), &info);
    std::cout << "info " << info << std::endl;
    for (int k = 0; k < 10; ++k) {
        for (int l = 0; l < 10; ++l) {
            std::cout << test(k, l) << ',';
        }
        std::cout << std::endl;
    }

    /////////////////
    // TEST LU AVEC LE "VRAI" LU
    ////////////////////

    // std::cout << "___________________________" << std::endl;
    // std::cout << "test lu " << std::endl;
    // auto lu = get_lu(LU(dense_matrix_21));
    // auto L  = lu.first;
    // auto U  = lu.second;
    // std::cout << normFrob(L * U - dense_matrix_21) / normFrob(dense_matrix_21) << std::endl;
    // std::cout << L.nb_rows() << ',' << L.nb_cols() << std::endl;
    // std::cout << U.nb_rows() << ',' << U.nb_cols() << std::endl;

    // DenseGenerator<double> lh(L);
    // DenseGenerator<double> uh(U);
    // auto Lh = hmatrix_tree_builder_L.build(lh);
    // auto Uh = hmatrix_tree_builder_U.build(uh);

    // Matrix<double> ll(size, size);
    // copy_to_dense(Lh, ll.data());
    // Matrix<double> uu(size, size);
    // copy_to_dense(Uh, uu.data());

    // std::cout << "info lh et uh" << std::endl;
    // std::cout << "compression:" << std::endl;
    // std::cout << "Lh : " << Lh.get_compression() << " Uh : " << Uh.get_compression() << std::endl;
    // std::cout << "erreur " << std::endl;
    // std::cout << "Lh : " << normFrob(ll - L) / normFrob(L) << " Uh : " << normFrob(uu - U) / normFrob(U) << std::endl;

    // auto x = generate_random_vector(size);
    // std::vector<double> y(size, 0.0);
    // std::vector<double> x_forward(size, 0.0);
    // std::cout << "11 " << norm2(x) << std::endl;
    // Lh.add_vector_product('N', 1.0, x.data(), 0.0, y.data());
    // std::cout << "22" << norm2(x) << std::endl;
    // auto yy = L * x;
    // std::cout << "?" << norm2(yy - y) / norm2(yy) << std::endl;
    // Lh.forward_substitution_extract(*root_cluster_1, x_forward, y);

    // std::cout << norm2(x_forward - x) / norm2(x) << std::endl;
    // std::cout << norm2(L * x_forward - L * x) / norm2(L * x) << std::endl;

    /////////////////////////////////////////////////////////////////
    ///// C'est quand même super biizard que pour une maatirce qui est sencé être inversible Lx = Ly et x !=  y
    ///// Du coup ca doit venir de décomposition LU

    //////////////////////////////////////////////////////////////////
    // On peut esssayer d'assembler un matrice traingulaire a la main: -> on copupe une hmat en 2
    //////////////////////////////////////////////////////////////////

    ////////////////////////////////
    //// On initialise une Hmatrice nulle

    // Matrix<double> zero(root_cluster_1->get_size(), root_cluster_1->get_size());
    // DenseGenerator<double> Z(zero);

    // // //////////////////
    // // // FORWARD   L x = y
    // // /////////////////
    // HMatrixTreeBuilder<double, double> hmatrix_tree_builder_Z(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    // auto Lprime = hmatrix_tree_builder_Z.build(Z);
    // root_hmatrix_21.Get_L(Lprime);
    // Matrix<double> dl(root_cluster_1->get_size(), root_cluster_1->get_size());
    // copy_to_dense(Lprime, dl.data());
    // auto x_prime = generate_random_vector(size);
    // std::vector<double> y_prime(size, 0.0);
    // std::vector<double> x_forward_prime(size, 0.0);
    // Lprime.add_vector_product('N', 1.0, x_prime.data(), 0.0, y_prime.data());
    // Lprime.forward_substitution_extract(*root_cluster_1, x_forward_prime, y_prime);
    // std::cout << "forward " << std::endl;
    // std::cout << "erreur relative de x_forward - x " << norm2(x_forward_prime - x_prime) / norm2(x_prime) << std::endl;
    // std::cout << " erreur relative de L*x-L*x_forward " << norm2(dl * x_forward_prime - dl * x_prime) / norm2(dl * x_prime) << std::endl;

    // //////////////////
    // // BACKWARD  Ux = y
    // /////////////////
    // HMatrixTreeBuilder<double, double> hmatrix_tree_builder_Zu(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    // auto Uprime = hmatrix_tree_builder_Zu.build(Z);
    // root_hmatrix_21.Get_U(Uprime);
    // Matrix<double> dd(root_cluster_1->get_size(), root_cluster_1->get_size());
    // copy_to_dense(Uprime, dd.data());
    // auto x_prime2 = generate_random_vector(size);
    // std::vector<double> y_prime2(size, 0.0);
    // std::vector<double> x_backward_prime(size, 0.0);
    // Uprime.add_vector_product('N', 1.0, x_prime2.data(), 0.0, y_prime2.data());
    // std::cout << norm2(y_prime2) << std::endl;
    // std::cout << norm2(dd * x_prime2) << std::endl;

    // Uprime.backward_substitution_extract(*root_cluster_1, x_backward_prime, y_prime2);

    // std::cout << "backward " << std::endl;
    // std::cout << "erreur relative de x_backward - x " << norm2(x_backward_prime - x_prime2) / norm2(x_prime2) << std::endl;
    // std::cout << " erreur relative de U*x-U*x_backward " << norm2(dd * x_backward_prime - dd * x_prime2) / norm2(dd * x_prime2) << std::endl;

    // ///////////////////
    // //// Forward_T xU = y
    // //////////////////
    // auto xprime_3 = generate_random_vector(size);
    // std::vector<double> y3(size, 0.0);
    // Uprime.add_vector_product('T', 1.0, xprime_3.data(), 0.0, y3.data());
    // std::cout << "prod ok" << ',' << norm2(y3) << std::endl;
    // std::vector<double> x_forward_t(size, 0.0);
    // Uprime.forward_substitution_extract_T(*root_cluster_1, x_forward_t, y3);
    // std::cout << "forward_T" << ',' << norm2(x_forward_t) << std::endl;
    // std::cout << "erreur relative de x_ft - x :" << norm2(x_forward_t - xprime_3) / norm2(xprime_3) << std::endl;

    // ///////////////////////
    // // FORWARD_BACKWARD  LUx = y
    // //////////////////////
    // auto x_ref               = generate_random_vector(size);
    // std::vector<double> y_lu = dl * dd * x_ref;
    // std::vector<double> x_lu(size, 0.0);
    // forward_backward(Lprime, Uprime, *root_cluster_1, x_lu, y_lu);
    // // auto test = dl*dd*x_lu
    // std::cout << "forward backward " << std::endl;
    // std::cout << "erreur Lux = y  " << norm2(x_ref - x_lu) / norm2(x_ref) << std::endl;

    // /////////////////////////////////////////////
    // /// LA MEME CHOSE MAIS AVEC DES MATRICES
    // //////////////////////////////////////
    // std::cout << "equations matricielles " << std::endl;
    /////////////////////////
    /// FORWARD_M L x = H ( tout en Hmat)
    ///////////////////////
    // auto X        = Lprime.hmatrix_product(Uprime);
    // auto ref_prod = dl * dd;
    // Matrix<double> ref_21(size, size);
    // copy_to_dense(root_hmatrix_21, ref_21.data());
    // auto L_hmat = Lprime.hmatrix_product(root_hmatrix_21);

    // auto ref_prod = dl * ref_21;

    // Matrix<double> dense_prod(size, size);

    // copy_to_dense(L_hmat, dense_prod.data());
    // std::cout << "norme ref prod " << normFrob(ref_prod) << std::endl;
    // std::cout << " erreur entre lh uh et (lu)h" << normFrob(dense_prod - ref_prod) / normFrob(ref_prod) << std::endl;
    // HMatrix<double, double> hmatrix_Z(root_cluster_1, root_cluster_1);

    // std::cout << "hmult ?" << std::endl;
    // auto prod = root_hmatrix_21.hmatrix_product(root_hmatrix_21);
    // Matrix<double> pp(size, size);
    // copy_to_dense(prod, pp.data());
    // std::cout << normFrob(pp - ref_21 * ref_21) / normFrob(ref_21 * ref_21) << std::endl;
    ///////////////////
    /// FORWARD_M -> LX =Z
    // forward marche mais pas dans < lu donc la c'est forward new (juste enlevé le new pour tester forward)
    /////////////////
    // hmatrix_Z.copy_zero(L_hmat);

    // Matrix<double> res(size, size);
    // Forward_M_new(Lprime, L_hmat, *root_cluster_1, *root_cluster_1, hmatrix_Z);

    // copy_to_dense(hmatrix_Z, res.data());
    // std::cout << "erreur ref - dense ( res) " << normFrob(res - ref_21) / normFrob(ref_21) << std::endl;

    /// FORWARD_M_T -> XU =Z
    // auto hmat_U = root_hmatrix_21.hmatrix_product(Uprime);

    // hmatrix_Z.copy_zero(hmat_U);
    // hmatrix_Z.set_low_rank_generator(root_hmatrix_21.get_lr());

    // hmatrix_Z.set_admissibility_condition(root_hmatrix_21.get_adm());
    // Matrix<double> ttest(size, size);
    // copy_to_dense(hmat_U, ttest.data());
    // std::cout << normFrob(ttest) << std::endl;
    // Matrix<double> res(size, size);
    // Forward_M_T_new(Uprime, hmat_U, *root_cluster_1, *root_cluster_1, hmatrix_Z);
    // copy_to_dense(hmatrix_Z, res.data());
    // std::cout << "erreur forward_M_T " << normFrob(res - ref_21) / normFrob(ref_21) << std::endl;

    ////////////////////////////////
    // H_LU
    ////////////////////////////////

    // root_hmatrix_21.Get_Block(10, 10, 10, 10);

    // auto z1 = root_hmatrix_21.get_block(32, 32, 32, 0);
    // std::cout << z1->get_children().size() << std::endl;
    // HMatrix<double, double> Uh(root_cluster_1, root_cluster_1);
    // HMatrix<double, double> Lh(root_cluster_1, root_cluster_1);
    // Uh.copy_zero(root_hmatrix_21);
    // Lh.copy_zero(root_hmatrix_21);
    // Uh.set_admissibility_condition(root_hmatrix_21.get_adm());
    // Lh.set_admissibility_condition(root_hmatrix_21.get_adm());
    // Uh.set_low_rank_generator(root_hmatrix_21.get_lr());
    // Lh.set_low_rank_generator(root_hmatrix_21.get_lr());
    // std::cout << "!!" << std::endl;

    // // H_LU_new(Lh, Uh, root_hmatrix_21, *root_cluster_1);
    // my_hlu(root_hmatrix_21, Lh, Uh, *root_cluster_1);

    // auto lu = Lh.hmatrix_product(Uh);
    // Matrix<double> lu_dense(size, size);
    // Matrix<double> ll(size, size);
    // Matrix<double> uu(size, size);
    // copy_to_dense(Lh, ll.data());
    // copy_to_dense(Uh, uu.data());
    // std::cout << "=====================================" << std::endl;
    // std::cout << normFrob(ll) << ',' << normFrob(uu) << ',' << normFrob(ref_21) << std::endl;
    // std::cout << normFrob(ll * uu - ref_21) / normFrob(ref_21) << std::endl;

    // copy_to_dense(lu, lu_dense.data());
    // std::cout << "!!" << std::endl;
    // std::cout << normFrob(lu_dense - ref_21) / normFrob(ref_21) << std::endl;

    ///////////////////////:
    /// HLU DENSE
    //////////////////
    // Matrix<double> ldense(size, size);
    // Matrix<double> udense(size, size);
    // auto reff = ref_21;
    // H_LU_dense(Lh, Uh, root_hmatrix_21, *root_cluster_1, reff, ldense, udense);
    // Matrix<double> ll(size, size);
    // Matrix<double> uu(size, size);
    // copy_to_dense(Lh, ll.data());
    // copy_to_dense(Uh, uu.data());
    // std::cout << normFrob(ll * uu - ref_21) / normFrob(ref_21) << std::endl;
    // for (int k = 0; k < 15; ++k) {
    //     for (int l = 0; l < 15; ++l) {
    //         std::cout << uu(k, l) << ',';
    //     }
    //     std::cout << std::endl;
    // }

    ////////////////////////////////////////////////////////////////////
    // forward_substitution_dense(Lprime, L_hmat, dense_prod, Lprime.get_target_cluster(), Lprime.get_source_cluster(), res, hmatrix_Z);
    // // hmatrix_Z.forward_substitution_this(Lprime, L_hmat, dense_prod, Lprime.get_target_cluster(), Lprime.get_source_cluster(), res);

    // std::cout << "erreur sur le dense" << normFrob(res - ref_21) / normFrob(ref_21) << std::endl;

    // Matrix<double> res_h(size, size);
    // copy_to_dense(hmatrix_Z, res_h.data());
    // std::cout << " erreur sur le Hmat" << normFrob(res_h - ref_21) / normFrob(ref_21) << std::endl;

    // auto test_h = Lprime.hmatrix_product(hmatrix_Z);
    // Matrix<double> densep(size, size);
    // copy_to_dense(test_h, densep.data());
    // std::cout << "erreur aprés produit " << normFrob(densep - ref_prod) / normFrob(ref_prod) << std::endl;

    // auto bloc = hmatrix_Z.get_block(32, 32, 480, 448);
    // std::cout << "est ce qu'il a écrit sur la feuille ?" << normFrob(*bloc->get_dense_data()) << std::endl;
    ////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////
    /////TEST ANI
    /////////////////////////////

    // Matrix<double> A11(size / 2, size / 2);
    // Matrix<double> A12(size / 2, size / 2);
    // Matrix<double> A21(size / 2, size / 2);
    // Matrix<double> A22(size / 2, size / 2);
    // Matrix<double> reference(size, size);
    // Matrix<double> li(size, size);
    // for (int k = 0; k < size / 2; ++k) {
    //     for (int l = 0; l < size / 2; ++l) {
    //         std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
    //         std::uniform_real_distribution<double> distribution(0.0, 1.0);
    //         A11(k, l) = distribution(generator);
    //         A12(k, l) = distribution(generator);
    //         A21(k, l) = distribution(generator);
    //         A22(k, l) = distribution(generator);
    //         li(k, l)  = 0;
    //     }
    // }
    // // // A11.assign(size / 2, size / 2, A11.data(), false);

    // HMatrix<double, double> hmat(root_cluster_1, root_cluster_1);
    // auto a11 = hmat.add_child(root_cluster_1->get_children()[0].get(), root_cluster_1->get_children()[0].get());
    // a11->set_dense_data(A11);
    // auto a12 = hmat.add_child(root_cluster_1->get_children()[0].get(), root_cluster_1->get_children()[1].get());
    // a12->set_dense_data(A12);
    // auto a21 = hmat.add_child(root_cluster_1->get_children()[1].get(), root_cluster_1->get_children()[0].get());
    // a21->set_dense_data(A21);
    // auto a22 = hmat.add_child(root_cluster_1->get_children()[1].get(), root_cluster_1->get_children()[1].get());
    // a22->set_dense_data(A22);
    // copy_to_dense(hmat, reference.data());

    // HMatrixTreeBuilder<double, double> hmatrix_tree_builder_test(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    // auto X_h = hmatrix_tree_builder_test.build(Z);
    // Matrix<double> X(size, size);

    // // HMatrix<double, double> lmat(root_cluster_1, root_cluster_1);
    // //  lmat.get_block(size / 2, size / 2, 0, 0)->set_dense_data(A11);
    // //  lmat.get_block(size / 2, size / 2, size / 2, 0)->set_dense_data(A21);
    // //  lmat.get_block(size / 2, size / 2, size / 2, size / 2)->set_dense_data(A22);
    // // HMatrixTreeBuilder<double, double> hmatrix_tree_builder_l(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    // // auto lmat = hmatrix_tree_builder_l.build(Z);
    // // std::cout << lmat.get_leaves().size() << std::endl;

    // // hmat.Get_L(lmat);
    // // std::cout << "nb son " << lmat.get_children().size() << std::endl;
    // // std::cout << "info son 1 " << std::endl;
    // // std::cout << lmat.get_children()[0]->get_children().size() << std::endl;
    // // std::cout << "norme " << normFrob(*lmat.get_children()[0]->get_dense_data()) << std::endl;
    // // std::cout << "target" << lmat.get_children()[0]->get_target_cluster().get_size() << ',' << lmat.get_children()[0]->get_target_cluster().get_offset() << std::endl;
    // // std::cout << "source" << lmat.get_children()[0]->get_source_cluster().get_size() << ',' << lmat.get_children()[0]->get_source_cluster().get_offset() << std::endl;
    // // std::cout << "info son 2 " << std::endl;
    // // std::cout << lmat.get_children()[1]->get_children().size() << std::endl;
    // // std::cout << "norme " << normFrob(*lmat.get_children()[1]->get_dense_data()) << std::endl;
    // // std::cout << "target" << lmat.get_children()[1]->get_target_cluster().get_size() << ',' << lmat.get_children()[1]->get_target_cluster().get_offset() << std::endl;
    // // std::cout << "source" << lmat.get_children()[1]->get_source_cluster().get_size() << ',' << lmat.get_children()[1]->get_source_cluster().get_offset() << std::endl;

    // // std::cout << lmat.get_leaves().size() << std::endl;
    // HMatrix<double, double> lmat(root_cluster_1, root_cluster_1);

    // // Matrix<double> ldense(size, size);
    // // lmat.To_dense(ldense);
    // auto l11 = lmat.add_child(root_cluster_1->get_children()[0].get(), root_cluster_1->get_children()[0].get());
    // l11->set_dense_data(A11);
    // auto l12 = lmat.add_child(root_cluster_1->get_children()[0].get(), root_cluster_1->get_children()[1].get());
    // l12->set_dense_data(li);
    // auto l21 = lmat.add_child(root_cluster_1->get_children()[1].get(), root_cluster_1->get_children()[0].get());
    // l21->set_dense_data(A21);
    // auto l22 = lmat.add_child(root_cluster_1->get_children()[1].get(), root_cluster_1->get_children()[1].get());
    // l22->set_dense_data(A22);
    // lmat.set_admissibility_condition(root_hmatrix_21.get_adm());

    // Matrix<double> l_dense(size, size);
    // copy_to_dense(lmat, l_dense.data());
    // std::cout << normFrob(l_dense) << ',' << normFrob(reference) << std::endl;
    // std::cout << "prod begin " << std::endl;
    // auto prod = lmat.hmatrix_product(hmat);
    // std::cout << "prod ok " << std::endl;
    // Matrix<double> prod_to_dense(size, size);
    // copy_to_dense(prod, prod_to_dense.data());
    // std::cout << "alors ?" << normFrob(prod_to_dense - l_dense * reference) / normFrob(prod_to_dense) << std::endl;
    // std::cout << "nb sons de prod :" << std::endl;
    // std::cout << prod.get_block(size / 2, size / 2, 0, 0)->get_children().size() << std::endl;
    // std::cout << prod.get_block(size / 2, size / 2, size / 2, 0)->get_children().size() << std::endl;
    // std::cout << prod.get_block(size / 2, size / 2, 0, size / 2)->get_children().size() << std::endl;
    // std::cout << prod.get_block(size / 2, size / 2, size / 2, size / 2)->get_children().size() << std::endl;
    // std::cout << "nb sons de l :" << std::endl;
    // std::cout << lmat.get_block(size / 2, size / 2, 0, 0)->get_children().size() << std::endl;
    // std::cout << lmat.get_block(size / 2, size / 2, size / 2, 0)->get_children().size() << std::endl;
    // std::cout << lmat.get_block(size / 2, size / 2, 0, size / 2)->get_children().size() << std::endl;
    // std::cout << lmat.get_block(size / 2, size / 2, size / 2, size / 2)->get_children().size() << std::endl;

    // forward_substitution_dense(lmat, prod, prod_to_dense, *root_cluster_1, *root_cluster_1, X, X_h);
    // std::cout << "erreur sur le dense " << normFrob(reference - X) / normFrob(reference) << std::endl;
    // Matrix<double> Xdense(size, size);
    // copy_to_dense(X_h, Xdense.data());
    // std::cout << "edrreur sur ref - dense(hmat) " << normFrob(Xdense - reference) / normFrob(reference) << std::endl;
    // auto test_prod = lmat.hmatrix_product(X_h);
    // Matrix<double> test_prod_dense(size, size);
    // copy_to_dense(test_prod, test_prod_dense.data());
    // std::cout << normFrob(test_prod_dense - prod_to_dense) / normFrob(prod_to_dense) << std::endl;
    //////////////////////////////////////////////////:
    /// TEST FORWARD BUILD
    ///////////////////////////////////////////////////////

    // HMatrix<double, double> xforward(root_cluster_1, root_cluster_1);
    // xforward.forward_substitution_build(Lprime, L_hmat, dense_prod, Lprime.get_target_cluster(), Lprime.get_source_cluster(), res, xforward);

    // Matrix<double> res_h(size, size);
    // copy_to_dense(hmatrix_Z, res_h.data());
    // std::cout << normFrob(res_h - ref_21) / normFrob(ref_21) << std::endl;
    // TEST SUR SOUS BLOCS
    // auto X11 = hmatrix_Z.get_block(size / 2, size / 2, 0, 0);
    // auto L11 = Lprime.get_block(size / 2, size / 2, 0, 0);
    // auto Z11 = L_hmat.get_block(size / 2, size / 2, 0, 0);
    // Matrix<double> z11(size / 2, size / 2);
    // copy_to_dense(*Z11, z11.data());

    // Forward_M_extract(*L11, *Z11, L11->get_target_cluster(), L11->get_source_cluster(), *X11);

    // Matrix<double> x11(size / 2, size / 2);
    // Matrix<double> l11(size / 2, size / 2);

    // copy_to_dense(*X11, x11.data());
    // copy_to_dense(*L11, l11.data());
    // std::cout << normFrob(l11 * x11 - z11) / normFrob(z11) << std::endl;

    // std::cout << "test forwartd_M avec les mains" << std::endl;
    // auto Lts = Lprime.get_block(64, 64, 448, 448);
    // auto Zts = X.get_block(64, 64, 448, 320);

    // Matrix<double> ldense(64, 64);
    // copy_to_dense(*Lts, ldense.data());
    // Matrix<double> zdense(64, 64);
    // copy_to_dense(*Zts, zdense.data());

    // std::vector<double> xt(size, 0.0);
    // auto zj = zdense.get_col(3);
    // for (int k = 0; k < 64; ++k) {
    //     std::cout << zj[k] << std::endl;
    // }
    // std::vector<double> zt(size, 0.0);
    // for (int k = 0; k < 64; ++k) {
    //     zt[k + 448] = zj[k];
    // }
    // for (int k = 0; k < 64; ++k) {
    //     std::cout << "!" << zt[k + 448] << std::endl;
    // }
    // Lprime.forward_substitution(Lts->get_target_cluster(), xt, zt);

    // std::vector<double> xres(64, 0.0);
    // for (int k = 0; k < 64; ++k) {
    //     xres[k] = xt[k + 448];
    // }
    // std::cout << "alors ?" << norm2(ldense * xres - zj) / norm2(zj) << std::endl;
}
