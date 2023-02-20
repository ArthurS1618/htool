#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_hmatrix_product(int size_1, int size_2, int size_3, htool::underlying_type<T> epsilon) {

    srand(1);
    bool is_error = false;

    // Geometry
    vector<double> p1(3 * size_1), p2(3 * size_2), p3(3 * size_3);
    create_disk(3, 0., size_1, p1.data());
    create_disk(3, 0.5, size_2, p2.data());
    create_disk(3, 1., size_3, p3.data());

    // Clustering
    ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> recursive_build_strategy_1(size_1, 3, p1.data(), 2, 2);
    ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> recursive_build_strategy_2(size_2, 3, p2.data(), 2, 2);
    ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> recursive_build_strategy_3(size_3, 3, p3.data(), 2, 2);

    std::shared_ptr<Cluster<htool::underlying_type<T>>> root_cluster_1 = make_shared<Cluster<htool::underlying_type<T>>>(recursive_build_strategy_1.create_cluster_tree());
    std::shared_ptr<Cluster<htool::underlying_type<T>>> root_cluster_2 = make_shared<Cluster<htool::underlying_type<T>>>(recursive_build_strategy_2.create_cluster_tree());
    std::shared_ptr<Cluster<htool::underlying_type<T>>> root_cluster_3 = make_shared<Cluster<htool::underlying_type<T>>>(recursive_build_strategy_3.create_cluster_tree());

    // Generator
    GeneratorTestType generator_21(3, size_2, size_1, p2, p1, root_cluster_2, root_cluster_1);
    GeneratorTestType generator_32(3, size_3, size_2, p3, p2, root_cluster_3, root_cluster_2);

    // References
    Matrix<T> dense_matrix_21(root_cluster_2->get_size(), root_cluster_1->get_size()), dense_matrix_32(root_cluster_3->get_size(), root_cluster_2->get_size());
    generator_21.copy_submatrix(dense_matrix_21.nb_rows(), dense_matrix_21.nb_cols(), 0, 0, dense_matrix_21.data());
    generator_32.copy_submatrix(dense_matrix_32.nb_rows(), dense_matrix_32.nb_cols(), 0, 0, dense_matrix_32.data());

    Matrix<T> reference_dense_matrix = dense_matrix_32 * dense_matrix_21;

    // HMatrix
    double eta = 10;

    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_21(root_cluster_2, root_cluster_1, epsilon, eta, 'N', 'N');
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_32(root_cluster_3, root_cluster_2, epsilon, eta, 'N', 'N');

    // // build
    auto root_hmatrix_21 = hmatrix_tree_builder_21.build(generator_21);
    auto root_hmatrix_32 = hmatrix_tree_builder_32.build(generator_32);
    // Matrix<T> hmatrix_to_dense_21(root_hmatrix_21.get_target_cluster().get_size(), root_hmatrix_21.get_source_cluster().get_size()), hmatrix_to_dense_32(root_hmatrix_32.get_target_cluster().get_size(), root_hmatrix_32.get_source_cluster().get_size());
    // copy_to_dense(root_hmatrix_21, hmatrix_to_dense_21.data());
    // copy_to_dense(root_hmatrix_32, hmatrix_to_dense_32.data());
    // std::cout << normFrob(dense_matrix_21 - hmatrix_to_dense_21) / normFrob(dense_matrix_21) << "\n";
    // std::cout << normFrob(dense_matrix_32 - hmatrix_to_dense_32) / normFrob(dense_matrix_32) << "\n";

    // HMatrix product
    // HMatrix result_hmatrix = root_hmatrix_32.prod(root_hmatrix_21);
    // HMatrix result_hmatrix = prod(root_hmatrix_32,root_hmatrix_21);

    return is_error;
}
