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

void saveVectorToFile(const std::vector<double> &data, const std::string &filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        // Écrit tous les éléments sur une seule ligne avec des virgules comme séparateurs
        for (size_t i = 0; i < data.size() - 1; ++i) {
            file << data[i];
            file << ",";
        }
        file << data[data.size() - 1];

        file.close();
        std::cout << "Vecteur enregistré dans le fichier : " << filename << std::endl;
    } else {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << " pour l'écriture." << std::endl;
    }
}

///////////////////////////////////////////////////////////////////
int main() {
    std::vector<double> time_assmbl;
    std::vector<double> time_vect;
    std::vector<double> time_mat;
    std::vector<double> time_lu;

    std::vector<double> err_asmbl;
    std::vector<double> err_vect;
    std::vector<double> err_mat;
    std::vector<double> err_lu;

    std::vector<double> compr_ratio;

    std::cout << "____________________________" << std::endl;
    for (int k = 11; k < 30; ++k) {

        int size = 600 + std::pow(2.0, k / 30.0 * 15.0);
        // 600 + 133 * k;

        std::cout << "_________________________________________" << std::endl;
        std::cout << "HMATRIX" << std::endl;
        double epsilon = 1e-4;
        double eta     = 1;
        std::cout << "size =" << size << std::endl;
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
        GeneratorDouble generator(3, size, size, p1, p1, root_cluster_1, root_cluster_1);

        Matrix<double> reference_num_htool(size, size);
        generator.copy_submatrix(size, size, 0, 0, reference_num_htool.data());
        // HMATRIX
        HMatrixTreeBuilder<double, double> hmatrix_tree_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');

        // build
        auto start_time_asmbl = std::chrono::high_resolution_clock::now();
        auto root_hmatrix     = hmatrix_tree_builder.build(generator);
        auto end_time_asmbl   = std::chrono::high_resolution_clock::now();
        auto duration_asmbl   = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_asmbl - start_time_asmbl).count();
        time_assmbl.push_back(duration_asmbl);
        Matrix<double> ref(size, size);
        copy_to_dense(root_hmatrix, ref.data());

        auto er_b = normFrob(reference_num_htool - ref) / normFrob(reference_num_htool);
        err_asmbl.push_back(er_b);
        std::cout << "asmbl ok " << std::endl;

        // Matrice vecteur
        // auto x               = generate_random_vector(size);
        // auto start_time_vect = std::chrono::high_resolution_clock::now();
        // std::vector<double> y(size, 0.0);
        // root_hmatrix.add_vector_product('N', 1.0, x.data(), 1.0, y.data());
        // auto end_time_vect = std::chrono::high_resolution_clock::now();
        // auto duration_vect = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_vect - start_time_vect).count();
        // time_vect.push_back(duration_vect);
        // auto temp = reference_num_htool * x;
        // auto er_v = norm2(y - temp) / norm2(temp);
        // err_vect.push_back(er_v);
        // std::cout << "vect ok " << std::endl;

        // Matrice Matrice
        auto start_time_mat = std::chrono::high_resolution_clock::now();
        std::cout << "mat " << std::endl;
        auto prod = root_hmatrix.hmatrix_product_new(root_hmatrix);
        std::cout << "mat ok" << std::endl;
        auto end_time_mat = std::chrono::high_resolution_clock::now();
        auto duration_mat = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_mat - start_time_mat).count();
        time_mat.push_back(duration_mat);
        auto tempm = reference_num_htool * reference_num_htool;
        Matrix<double> dense_prod(size, size);
        copy_to_dense(prod, dense_prod.data());
        auto er_m = normFrob(dense_prod - tempm) / normFrob(tempm);
        err_mat.push_back(er_m);
        auto compr = prod.get_compression();
        compr_ratio.push_back(compr);
        std::cout << "erreur :" << er_m << "    , compr" << compr << std::endl;
        // /// HLU
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
        // std::cout << "________________________" << std::endl;
        // std::cout << "Assemblage de matrice hiérarchique triangulaire pour nos tests : " << std::endl;
        // // Matrice Lh et Uh
        // auto ll = get_unperm_mat(L, *root_cluster_1, *root_cluster_1);
        // Matricegenerator<double, double> l_generator(ll, *root_cluster_1, *root_cluster_1);
        // HMatrixTreeBuilder<double, double> lh_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
        // auto Lh = lh_builder.build(l_generator);
        // Matrix<double> ldense(size, size);
        // copy_to_dense(Lh, ldense.data());
        // std::cout << "erreur Lh :" << normFrob(L - ldense) / normFrob(L) << std::endl;

        // auto uu = get_unperm_mat(U, *root_cluster_1, *root_cluster_1);
        // Matricegenerator<double, double> u_generator(uu, *root_cluster_1, *root_cluster_1);
        // HMatrixTreeBuilder<double, double> uh_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
        // auto Uh = uh_builder.build(u_generator);
        // Matrix<double> udense(size, size);
        // copy_to_dense(Uh, udense.data());
        // std::cout << "erreur uh : " << normFrob(U - udense) / normFrob(U) << std::endl;
        // auto prod_LU = Lh.hmatrix_product(Uh);

        // HMatrix<double, double> Llh(root_cluster_1, root_cluster_1);
        // Llh.set_admissibility_condition(prod_LU.get_admissibility_condition());
        // Llh.set_epsilon(epsilon);
        // Llh.set_low_rank_generator(prod_LU.get_low_rank_generator());

        // HMatrix<double, double> Uuh(root_cluster_1, root_cluster_1);
        // Uuh.set_admissibility_condition(prod_LU.get_admissibility_condition());
        // Uuh.set_epsilon(epsilon);
        // Uuh.set_low_rank_generator(prod_LU.get_low_rank_generator());
        // Uuh.copy_zero(prod_LU);
        // Llh.copy_zero(prod_LU);
        // Matrix<double> prod_dense(size, size);
        // copy_to_dense(prod_LU, prod_dense.data());

        // HMatrix<double, double> Permutation(root_cluster_1, root_cluster_1);
        // Permutation.set_admissibility_condition(prod_LU.get_admissibility_condition());
        // Permutation.set_epsilon(epsilon);
        // Permutation.set_low_rank_generator(prod_LU.get_low_rank_generator());
        // Permutation.copy_zero(prod_LU);
        // // Matrix<double> ref_lu(size, size);
        // // copy_to_dense(produit_LU, ref_lu.data());
        // Matrix<double> dense_perm(size, size);
        // // std::cout << "norme de Lh*Uh la matrice que l'on factorise  : " << normFrob(ref_lu) << "!!" << normFrob(ldense * udense) << std::endl;
        // HMatrix<double, double> unperm(root_cluster_1, root_cluster_1);

        // Permutation.set_admissibility_condition(prod_LU.get_admissibility_condition());
        // unperm.set_admissibility_condition(prod_LU.get_admissibility_condition());
        // Permutation.set_low_rank_generator(prod_LU.get_low_rank_generator());
        // unperm.set_low_rank_generator(prod_LU.get_low_rank_generator());
        // unperm.copy_zero(prod_LU);

        // std::cout << "_________________________________________" << std::endl;
        // auto start_time_lu = std::chrono::high_resolution_clock::now();

        // HMatrix_PLU(prod_LU, *root_cluster_1, Llh, Uuh, Permutation, unperm);
        // auto end_time_lu = std::chrono::high_resolution_clock::now();
        // auto duration_lu = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_lu - start_time_lu).count();
        // time_lu.push_back(duration_lu);
        // copy_to_dense(Permutation, dense_perm.data());
        // Matrix<double> denseunperm(size, size);
        // copy_to_dense(unperm, denseunperm.data());

        // auto test_lu = Llh.hmatrix_product(Uuh);
        // Matrix<double> lu_dense(size, size);
        // copy_to_dense(test_lu, lu_dense.data());
        // double erlu = normFrob(prod_dense - dense_perm * lu_dense) / normFrob(reference_num_htool);
        // err_lu.push_back(erlu);
        // std::cout << erlu << std::endl;
        // Matrix<double> lldense(size, size);
        // copy_to_dense(Llh, lldense.data());
        // Matrix<double> uudense(size, size);
        // copy_to_dense(Uuh, uudense.data());
        // std::cout << "erreur lu " << normFrob(prod_dense - dense_perm * (lldense * uudense)) / normFrob(prod_dense) << " L : " << normFrob(L - lldense) / normFrob(L) << "             U: " << normFrob(U - uudense) / normFrob(udense) << std::endl;

        // HMatrix<double, double> lh(root_cluster_1, root_cluster_1);
        // lh.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
        // lh.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
        // lh.copy_zero(root_hmatrix);

        // HMatrix<double, double> uh(root_cluster_1, root_cluster_1);
        // uh.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
        // uh.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
        // uh.copy_zero(root_hmatrix);

        // HMatrix<double, double> Permutation(root_cluster_1, root_cluster_1);
        // Permutation.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
        // Permutation.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
        // Permutation.copy_zero(root_hmatrix);

        // HMatrix<double, double> unperm(root_cluster_1, root_cluster_1);
        // unperm.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
        // unperm.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
        // unperm.copy_zero(root_hmatrix);

        // HMatrix_PLU(root_hmatrix, *root_cluster_1, lh, uh, Permutation, unperm);
        // Matrix<double> lldense(size, size);
        // copy_to_dense(lh, lldense.data());
        // Matrix<double> uudense(size, size);
        // copy_to_dense(uh, uudense.data());
        // Matrix<double> dense_perm(size, size);
        // copy_to_dense(Permutation, dense_perm.data());

        // std::cout << "erreur lu " << normFrob(dense_perm * ref - (lldense * uudense)) / normFrob(ref) << " L : " << normFrob(L - lldense) / normFrob(L) << "             U: " << normFrob(U - uudense) / normFrob(uudense) << std::endl;
    }

    // saveVectorToFile(time_assmbl, "time_assemble_eps4.txt");
    // saveVectorToFile(time_vect, "time_vect_eps4.txt");
    saveVectorToFile(time_mat, "time_mat_eps4.txt");
    // saveVectorToFile(time_lu, "time_lu_eps4.txt");

    // saveVectorToFile(err_asmbl, "error_assemble_eps4.txt");
    // saveVectorToFile(err_vect, "error_vect_eps4.txt");
    saveVectorToFile(err_mat, "error_mat_eps4.txt");
    // saveVectorToFile(err_lu, "error_lu_eps4.txt");

    // saveVectorToFile(compr_ratio, "compression_ratio_eps4.txt");
}