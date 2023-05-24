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

#include <ctime>
#include <iostream>
using namespace htool;

std::vector<double> to_row_major(const Matrix<double> &M) {
    int nr = M.nb_rows();
    int nc = M.nb_cols();
    std::vector<double> conv;
    for (int k = 0; k < nc; ++k) {
        for (int l = 0; l < nr; ++l) {
            conv.push_back(M(l, k));
        }
    }
    return conv;
}
int main() {
    int size       = 512;
    double eta     = 10;
    double epsilon = 1e-5;
    //////////////////////
    /// nuage de points
    //////////////////////
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
    std::cout << "copression " << root_hmatrix_21.get_compression() << std::endl;
    // std::cout << "erreur : " << normFrob(dense_21 - dense_matrix_21) << std::endl;

    // vrai mliplication
    clock_t start_time = std::clock(); // temps de départ
    auto ref           = dense_matrix_32 * dense_matrix_21;
    clock_t end_time   = clock(); // temps de fin
    double time_taken  = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "temps vrai mult :" << time_taken << std::endl;
    // Hmat* lr
    std::vector<double> dense_row(size * size, 0.0);
    std::vector<double> prod(size * size, 0.0);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dense_row[i * size + j] = dense_21.data()[j * size + i];
        }
    }
    start_time = std::clock();
    root_hmatrix_32.add_matrix_product_row_major('N', 1.0, dense_row.data(), 0.0, prod.data(), size);
    end_time           = clock(); // temps de fin
    double time_taken0 = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "temps hmat lr :" << time_taken0 << std::endl;
    std::vector<double> tt(size * size, 0.0), ss;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            tt[i * size + j] = ref.data()[j * size + i];
        }
    }
    Matrix<double> tempo(size, size);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            tempo.data()[i * size + j] = prod.data()[j * size + i];
        }
    }

    // std::cout << "erreur hmat lr " << norm2(prod - tt) / norm2(tt) << std::endl;
    std::cout << "errer hmat lr  " << normFrob(tempo - ref) / normFrob(ref) << std::endl;

    //  lr *Hmat
    std::vector<double>
        dense_row2(size * size, 0.0);
    std::vector<double> prod2(size * size, 0.0);
    Matrix<double> dd(size, size);
    for (int k = 0; k < size; ++k) {
        for (int l = 0; l < size; ++l) {
            dd(l, k) = dense_32(k, l);
        }
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dense_row2[i * size + j] = dd.data()[j * size + i];
        }
    }
    start_time = std::clock();
    root_hmatrix_21.add_matrix_product_row_major('T', 1.0, dense_row2.data(), 0.0, prod2.data(), size);
    end_time    = clock();
    time_taken0 = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "temps  lr hmat :" << time_taken0 << std::endl;
    std::vector<double> temp(size * size, 0.0);
    for (int k = 0; k < size; ++k) {
        for (int l = 0; l < size; ++l) {
            temp[k * size + l] = prod2[l * size + k];
        }
    }
    std::cout << "!" << std::endl;
    std::cout << temp.size() << ',' << tt.size() << std::endl;
    std::cout << norm2(temp) << ',' << norm2(tt) << std::endl;
    std::cout << "erreur lr hmat :" << norm2(temp - tt) / norm2(tt) << std::endl;

    std::cout << "!!" << std::endl;
    ////////////////////////////////////////
    ///////// Hmult
    ////////////////////////////////////

    start_time           = std::clock(); // temps de départ
    auto root_hmatrix_31 = root_hmatrix_32.hmatrix_product(root_hmatrix_21);
    end_time             = clock(); // temps de fin
    time_taken0          = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "temps  hmat hmat :" << time_taken0 << std::endl;
    Matrix<double> to_dense(size, size);
    copy_to_dense(root_hmatrix_31, to_dense.data());
    std::cout << normFrob(to_dense - ref) / normFrob(ref) << std::endl;

    // Matrix<double> temp(size, size);
    // temp.assign(size, size, prod2.data(), true);
    // Matrix<double> rr(size, size);
    // std::vector<double> rrr(size * size);
    // for (int k = 0; k < size; ++k) {
    //     for (int l = 0; l < size; ++l) {
    //         rrr[k * size + l] = prod2[l * size + l];
    //     }
    // }

    // std::cout << rrr.size() << ',' << tt.size() << std::endl;
    // std::cout << norm2(tt) << std::endl;

    // Matrix<double> refT(size, size);
    // Matrix<double> XT(size, size);
    // XT.assign(size, size, xT_row_major.data(), true);
    // refT.assign(size, size, refT_row_major.data(), true);
    // start_time = std::clock(); // temps de départ
    // root_hmatrix_21.add_matrix_product_row_major('T', 1.0, XT.data(), 0.0, prodT.data(), 1);
    // end_time    = clock(); // temps de fin
    // time_taken0 = double(end_time - start_time) / CLOCKS_PER_SEC;
    // std::cout << "temps  lr hmat:" << time_taken0 << std::endl;
}