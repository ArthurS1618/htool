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
    double epsilon = 1e-5;
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

    Matrix<double> zero(root_cluster_1->get_size(), root_cluster_1->get_size());
    DenseGenerator<double> Z(zero);

    // //////////////////
    // // FORWARD   L x = y
    // /////////////////
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder_Z(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    auto Lprime = hmatrix_tree_builder_Z.build(Z);
    root_hmatrix_21.Get_L(Lprime);
    Matrix<double> dl(root_cluster_1->get_size(), root_cluster_1->get_size());
    copy_to_dense(Lprime, dl.data());
    auto x_prime = generate_random_vector(size);
    std::vector<double> y_prime(size, 0.0);
    std::vector<double> x_forward_prime(size, 0.0);
    Lprime.add_vector_product('N', 1.0, x_prime.data(), 0.0, y_prime.data());
    Lprime.forward_substitution_extract(*root_cluster_1, x_forward_prime, y_prime);
    std::cout << "forward " << std::endl;
    std::cout << "erreur relative de x_forward - x " << norm2(x_forward_prime - x_prime) / norm2(x_prime) << std::endl;
    std::cout << " erreur relative de L*x-L*x_forward " << norm2(dl * x_forward_prime - dl * x_prime) / norm2(dl * x_prime) << std::endl;

    //////////////////
    // BACKWARD  Ux = y
    /////////////////
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder_Zu(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    auto Uprime = hmatrix_tree_builder_Zu.build(Z);
    root_hmatrix_21.Get_U(Uprime);
    Matrix<double> dd(root_cluster_1->get_size(), root_cluster_1->get_size());
    copy_to_dense(Uprime, dd.data());
    auto x_prime2 = generate_random_vector(size);
    std::vector<double> y_prime2(size, 0.0);
    std::vector<double> x_backward_prime(size, 0.0);
    Uprime.add_vector_product('N', 1.0, x_prime2.data(), 0.0, y_prime2.data());
    std::cout << norm2(y_prime2) << std::endl;
    std::cout << norm2(dd * x_prime2) << std::endl;

    Uprime.backward_substitution_extract(*root_cluster_1, x_backward_prime, y_prime2);

    std::cout << "backward " << std::endl;
    std::cout << "erreur relative de x_backward - x " << norm2(x_backward_prime - x_prime2) / norm2(x_prime2) << std::endl;
    std::cout << " erreur relative de U*x-U*x_backward " << norm2(dd * x_backward_prime - dd * x_prime2) / norm2(dd * x_prime2) << std::endl;

    ///////////////////////
    // FORWARD_BACKWARD  LUx = y
    //////////////////////
    auto x_ref               = generate_random_vector(size);
    std::vector<double> y_lu = dl * dd * x_ref;
    std::vector<double> x_lu(size, 0.0);
    forward_backward(Lprime, Uprime, *root_cluster_1, x_lu, y_lu);
    // auto test = dl*dd*x_lu
    std::cout << "forward backward " << std::endl;
    std::cout << "erreur Lux = y  " << norm2(x_ref - x_lu) / norm2(x_ref) << std::endl;

    /////////////////////////////////////////////
    /// LA MEME CHOSE MAIS AVEC DES MATRICES
    //////////////////////////////////////
    std::cout << "equations matricielles " << std::endl;
    /////////////////////////
    /// FORWARD_M L x = H ( tout en Hmat)
    ///////////////////////
    // auto X        = Lprime.hmatrix_product(Uprime);
    // auto ref_prod = dl * dd;
    Matrix<double> ref_21(size, size);
    copy_to_dense(root_hmatrix_21, ref_21.data());
    auto X = Lprime.hmatrix_product(root_hmatrix_21);

    auto ref_prod = dl * ref_21;

    Matrix<double> dense_prod(size, size);

    copy_to_dense(X, dense_prod.data());
    std::cout << "norme ref prod " << normFrob(ref_prod) << std::endl;
    std::cout << " erreur entre lh uh et (lu)h" << normFrob(dense_prod - ref_prod) / normFrob(ref_prod) << std::endl;

    // // On cherche Z tq L xx =  X ;
    // // on initialise la hmat nulle
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder_Z_m(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    auto hmatrix_Z = hmatrix_tree_builder_Z_m.build(Z);
    std::cout << "!" << std::endl;

    Forward_M_extract(Lprime, X, *root_cluster_1, *root_cluster_1, hmatrix_Z);

    Matrix<double> test_hprod(size, size);
    copy_to_dense(hmatrix_Z, test_hprod.data());
    std::cout << "erreur L Z = X : ||Z-Z_ref|| =" << normFrob(test_hprod - ref_21) / normFrob(ref_21) << std::endl;
}
