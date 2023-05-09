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
int main() {
    std::vector<double> true_time;
    std::vector<double> my_time;
    for (int i = 1; i < 100; ++i) {
        int size       = i * 100;
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
        std::cout << "erreur : " << normFrob(dense_21 - dense_matrix_21) << std::endl;

        // Hmat* lr
        clock_t start_time = std::clock(); // temps de départ
        auto test_dense    = dense_matrix_32 * dense_matrix_21;
        clock_t end_time   = clock(); // temps de fin
        double time_taken  = double(end_time - start_time) / CLOCKS_PER_SEC;
        std::cout << "temps vrai mult :" << time_taken << std::endl;

        start_time         = std::clock(); // temps de départ
        auto testm         = root_hmatrix_32.hmat_lr(dense_matrix_21);
        end_time           = clock(); // temps de fin
        double time_taken0 = double(end_time - start_time) / CLOCKS_PER_SEC;
        std::cout << "temps hmat lr :" << time_taken0 << std::endl;
        std::cout << "erreur hmat lr " << normFrob(testm - test_dense) / normFrob(test_dense) << std::endl;
        true_time.push_back(time_taken);
        my_time.push_back(time_taken0);
    }

    std ::string true_file = " true_time.txt";
    std::string my_file    = " my_time.txt";
    std::ofstream file0(true_file);
    if (!file0.is_open()) {
        std::cerr << "Error opening file: " << true_file << std::endl;
    }
    for (const auto &x : true_time) {
        file0 << x << '\n';
    }
    file0.close();
    std::cout << "Vector written to file: " << true_file << std::endl;
    std::ofstream file1(true_file);
    if (!file1.is_open()) {
        std::cerr << "Error opening file: " << my_file << std::endl;
    }
    for (const auto &x : true_time) {
        file1 << x << '\n';
    }
    file1.close();
    std::cout << "Vector written to file: " << my_file << std::endl;
}