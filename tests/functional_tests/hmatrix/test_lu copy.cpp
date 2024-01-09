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
    double epsilon = 2e-4;
    double eta     = 20;
    int size       = std::pow(2, 9);

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

    //// On initialise une Hmatrice nulle

    Matrix<double> zero(root_cluster_1->get_size(), root_cluster_1->get_size());
    DenseGenerator<double> Z(zero);

    // //////////////////
    // // FORWARD   L x = y
    // /////////////////
    std::cout << "____________________________________" << std::endl;
    std::cout << "++++++++++++++TEST FORWARD  Lx=y+++++++++++++++++" << std::endl;
    std::cout << "____________________________________" << std::endl;
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder_Z(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    auto Lprime = hmatrix_tree_builder_Z.build(Z);
    root_hmatrix_21.Get_L(Lprime);
    Matrix<double> dl(root_cluster_1->get_size(), root_cluster_1->get_size());
    copy_to_dense(Lprime, dl.data());
    auto x_prime = generate_random_vector(size);
    std::vector<double> y_prime(size, 0.0);
    std::vector<double> x_forward_prime(size, 0.0);
    Lprime.add_vector_product('N', 1.0, x_prime.data(), 0.0, y_prime.data());
    // Lprime.forward_substitution_extract(*root_cluster_1, x_forward_prime, y_prime);
    Lprime.forward_substitution(*root_cluster_1, x_forward_prime, y_prime, 0);

    std::cout << "forward " << std::endl;
    std::cout << "erreur relative de x_forward - x " << norm2(x_forward_prime - x_prime) / norm2(x_prime) << std::endl;
    std::cout << " erreur relative de L*x-L*x_forward " << norm2(dl * x_forward_prime - dl * x_prime) / norm2(dl * x_prime) << std::endl;
    std::cout << "____________________________________" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    //////////////////
    // BACKWARD  xU = y
    /////////////////
    std::cout << "____________________________________" << std::endl;
    std::cout << "++++++++++++++TEST BACKWARD  x U=y+++++++++++++++++" << std::endl;
    std::cout << "____________________________________" << std::endl;
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder_Zu(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    auto Uprime = hmatrix_tree_builder_Zu.build(Z);
    root_hmatrix_21.Get_U(Uprime);
    Matrix<double> dd(root_cluster_1->get_size(), root_cluster_1->get_size());
    copy_to_dense(Uprime, dd.data());
    auto x_prime2 = generate_random_vector(size);
    // std::vector<double> y_prime2(size, 0.0);
    std::vector<double> x_backward_prime(size, 0.0);
    // Uprime.add_vector_product('N', 1.0, x_prime2.data(), 0.0, y_prime2.data());
    auto y_prime2 = dd.transp(dd) * x_prime2;
    Uprime.forward_substitution_T(*root_cluster_1, x_backward_prime, y_prime2, 0);
    std::cout << "erreur relative x_backward " << norm2(x_prime2 - x_backward_prime) / norm2(x_prime2) << std::endl;
    std::cout << "erreur relative xU -y " << norm2(dd * x_prime2 - dd * x_backward_prime) / norm2(dd * x_prime2) << std::endl;
    std::cout << "____________________________________" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    ///////////////////
    // FORWARD_M LX =Y
    ///////////////////
    std::cout << "____________________________________" << std::endl;
    std::cout << "++++++++++++++TEST FORWARD_M  LX=Y+++++++++++++++++" << std::endl;
    std::cout << "____________________________________" << std::endl;

    HMatrix<double, double> Xl(root_cluster_1, root_cluster_1);

    auto LX = Lprime.hmatrix_product(root_hmatrix_21);
    Matrix<double> pp(size, size);
    copy_to_dense(LX, pp.data());
    std::cout << "reference L*21 : erreur hmult" << normFrob(pp - dl * dense_21) / normFrob(dl * dense_21) << std::endl;
    Xl.copy_zero(LX);

    Matrix<double> res(size, size);
    Forward_M_new(Lprime, LX, *root_cluster_1, *root_cluster_1, Xl);

    copy_to_dense(Xl, res.data());
    std::cout << "erreur ref - dense ( res) " << normFrob(res - dense_21) / normFrob(dense_21) << std::endl;
    std::cout << "info de X" << std::endl;
    print_hmatrix_information(Xl, std::cout);
    save_leaves_with_rank(Xl, "hmatrix_structure.csv");
    std::cout << "____________________________________" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    /////////////////
    // Matrix<double> utest(size, size);
    // copy_to_dense(Uprime, utest.data());
    // Matrix<double> moins = dense_matrix_21 - dense_matrix_21 * utest;
    // std::cout << normFrob(moins) << std::endl;
    // auto XU = root_hmatrix_21.hmatrix_product(Uprime);
    // Matrix<double> mtest(size, size);
    // root_hmatrix_21.Moins(XU);
    // copy_to_dense(root_hmatrix_21, mtest.data());
    // std::cout << normFrob(mtest - moins) / normFrob(moins) << std::endl;
    ///////////////////
    // FORWARD_M^T LX =Y
    ///////////////////
    std::cout << "____________________________________" << std::endl;
    std::cout << "++++++++++++++ TEST FORWARD_M^T  XU=Y +++++++++++++++++" << std::endl;
    std::cout << "____________________________________" << std::endl;
    auto XU = root_hmatrix_21.hmatrix_product(Uprime);

    HMatrix<double, double> Xu(root_cluster_1, root_cluster_1);
    Xu.copy_zero(XU);
    Xu.set_low_rank_generator(root_hmatrix_21.get_low_rank_generator());
    Xu.set_admissibility_condition(root_hmatrix_21.get_admissibility_condition());

    Matrix<double> ttest(size, size);
    copy_to_dense(Uprime, ttest.data());
    std::cout << normFrob(ttest) << std::endl;
    Matrix<double> ress(size, size);
    Forward_M_T_new(Uprime, XU, *root_cluster_1, *root_cluster_1, Xu);
    copy_to_dense(Xu, ress.data());
    std::cout << "erreur forward_M_T " << normFrob(ress - dense_21) / normFrob(dense_21) << std::endl;
    std::cout << normFrob(dense_21 * ttest - ress * ttest) / normFrob(dense_21 * ttest) << std::endl;
    Matrix<double> xuu(size, size);
    copy_to_dense(XU, xuu.data());
    std::cout << normFrob(xuu - dense_21 * ttest) / normFrob(dense_21) << std::endl;
    std::cout << "norm ttest :" << normFrob(ttest) << std::endl;
    std::cout << "norm dense :" << normFrob(dense_21) << std::endl;

    ////////////////////////////////////////////////
    /// TEST POUR COMPRENDRE COMMENT MARCHE LES PIVOT DE GETRF
    ///////////////////////////////////////////////
    // std::cout << "_______________________________" << std::endl;
    // size = 5;
    // Matrix<double> temp(size, size);
    // for (int k = 0; k < size; ++k) {
    //     for (int l = 0; l < size; ++l) {
    //         temp(k, l) = dense_21(k, l);
    //     }
    // }

    // std::vector<int> ipiv(size, 0.0);
    // int info = -1;
    // auto LU  = temp;
    // /////////////////////////////////////
    // /// GETRF RENVOIE A=P*L*U , L et U triangulaires et ipiv les permuation:
    // /// ipiv -> au départ on a le cycle identité (1, 2, ..., size) . On parcours ipiv , et on change le k-ème avec le ipiv[k]_ème]
    // Lapack<double>::getrf(&size, &size, LU.data(), &size, ipiv.data(), &info);
    // std::cout << "info =0 :" << info << std::endl;
    // Matrix<double> LL(size, size);
    // Matrix<double> UU(size, size);
    // for (int i = 0; i < size; ++i) {
    //     UU(i, i) = LU(i, i);
    //     LL(i, i) = 1;
    //     for (int j = 0; j < i; ++j) {
    //         LL(i, j) = LU(i, j);
    //         UU(j, i) = LU(j, i);
    //     }
    // }
    // std::cout << "L et U récupéré" << std::endl;

    // auto tewst = LL * UU;

    // std::cout << "erreur avant permutation " << normFrob(tewst - temp) / normFrob(temp) << std::endl;
    // std::cout << std::endl;
    // int k1      = 1;
    // int incx    = 1;
    // auto untemp = temp;
    // std::cout << "___________________________________" << std::endl;
    // std::cout << "avant permutation " << std::endl;
    // for (int k = 0; k < size; ++k) {
    //     for (int l = 0; l < size; ++l) {
    //         std::cout << temp(k, l) << ',';
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "ipiv:" << std::endl;
    // for (int k = 0; k < size; ++k) {
    //     std::cout << ipiv[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "matrice de permutation" << std::endl;
    // std::vector<int> vrai_piv(size);
    // vrai_piv[0] = 1;
    // vrai_piv[1] = 2;
    // vrai_piv[2] = 5;
    // vrai_piv[3] = 3;
    // vrai_piv[4] = 4;
    // Matrix<double> permutation(size, size);
    // for (int k = 0; k < size; ++k) {
    //     permutation(vrai_piv[k] - 1, k) = 1.0;
    // }
    // std::vector<int> cycle(size);
    // for (int k = 0; k < size; ++k) {
    //     cycle[k] = k + 1;
    // }
    // for (int k = 0; k < size; ++k) {
    //     if (ipiv[k] - 1 != k) {
    //         int temp           = cycle[k];
    //         cycle[k]           = cycle[ipiv[k] - 1];
    //         cycle[ipiv[k] - 1] = temp;
    //     }
    //     for (int kk = 0; kk < size; ++kk) {
    //         std::cout << "k=" << k << ':' << cycle[kk] << ',';
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "cycle" << std::endl;

    // for (int k = 0; k < size; ++k) {
    //     for (int l = 0; l < size; ++l) {
    //         std::cout << permutation(k, l) << ',';
    //     }
    //     std::cout << std::endl;
    // }
    // Matrix<double> cyclmat(size, size);
    // std::cout << "vrai matrice de permutaion ?" << std::endl;
    // std::cout << normFrob(untemp - permutation * tewst) / normFrob(untemp) << std::endl;
    // for (int k = 0; k < size; ++k) {
    //     cyclmat(cycle[k] - 1, k) = 1.0;
    // }
    // std::cout << "routine ok ?" << std::endl;
    // std::cout << normFrob(untemp - cyclmat * tewst) / normFrob(untemp) << std::endl;

    ///////////////////////////////////////////////////////
    // Matrix<double> inv_ipiv(size, size);
    // for (int k = 0; k < size; ++k) {
    //     inv_ipiv(k, ipiv[k] - 1) = 1.0;
    // }
    // // for (int k = 0; k < size; ++k) {
    // //     std::cout << iipiv[k] << ',';
    // // }
    // std::cout << std::endl;
    // Matrix<double> uperm = UU;
    // Matrix<double> lperm = inv_ipiv * LL;
    // incx                 = -1;
    // // Lapack<double>::laswp(&size, uperm.data(), &size, &k1, &size, iipiv.data(), &incx);
    // // Lapack<double>::laswp(&size, tewst.data(), &size, &k1, &size, iipiv.data(), &incx);
    // std::cout << "erreur permutation inverse" << std::endl;
    // std::cout << normFrob(untemp - lperm * uperm) / normFrob(untemp) << std::endl;

    // std::cout << "Aperm:" << std::endl;
    // for (int k = 0; k < size; ++k) {
    //     for (int l = 0; l < size; ++l) {
    //         std::cout << tewst(k, l) << ',';
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "ipiv" << std::endl;
    // for (int k = 0; k < size; ++k) {
    //     if (ipiv[k] - 1 != k) {
    //         std::cout << "k=" << k << ',' << "piv[k]=" << ipiv[k] << std::endl;
    //     }
    // }
    // // HMatrix<double, double> Xu(root_cluster_1, root_cluster_1);
    // Xu.set_low_rank_generator(root_hmatrix_21.get_low_rank_generator());

    // auto XU = root_hmatrix_21.hmatrix_product(Uprime);
    // Matrix<double> ppp(size, size);
    // copy_to_dense(XU, ppp.data());
    // std::cout << "reference 21 * U : erreur hmult" << normFrob(ppp - dense_21 * dd) / normFrob(dense_21 * dd) << std::endl;
    // Xu.copy_zero(XU);

    // Matrix<double> ress(size, size);
    // Forward_M_T_new(Uprime, XU, *root_cluster_1, *root_cluster_1, Xu);

    // copy_to_dense(Xu, ress.data());
    // std::cout << "erreur ref - dense ( res) " << normFrob(ress - dense_21) / normFrob(dense_21) << std::endl;
    // std::cout << "____________________________________" << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;

    ///////////////////
    // H LU
    ///////////////////
    std::cout << "____________________________________" << std::endl;
    std::cout << "++++++++++++++ TEST HLU  LU=M +++++++++++++++++" << std::endl;
    std::cout << "____________________________________" << std::endl;
    HMatrix<double, double> Lh(root_cluster_1, root_cluster_1);
    Lh.copy_zero(root_hmatrix_21);
    Lh.set_low_rank_generator(root_hmatrix_21.get_low_rank_generator());
    Lh.set_admissibility_condition(root_hmatrix_21.get_admissibility_condition());
    HMatrix<double, double> Uh(root_cluster_1, root_cluster_1);
    Uh.copy_zero(root_hmatrix_21);
    Uh.set_low_rank_generator(root_hmatrix_21.get_low_rank_generator());
    Uh.set_admissibility_condition(root_hmatrix_21.get_admissibility_condition());

    H_LU(root_hmatrix_21, *root_cluster_1, Lh, Uh);
    std::cout << "lu ok " << std::endl;
    Matrix<double> ldense(root_cluster_1->get_size(), root_cluster_1->get_size());
    Matrix<double> udense(root_cluster_1->get_size(), root_cluster_1->get_size());
    copy_to_dense(Lh, ldense.data());
    copy_to_dense(Uh, udense.data());
    std::cout << "erreur HLU " << normFrob(dense_21 - ldense * udense) / normFrob(dense_21) << std::endl;

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
