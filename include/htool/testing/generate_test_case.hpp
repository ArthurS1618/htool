
#ifndef HTOOL_TESTING_GENERATE_TEST_CASE_HPP
#define HTOOL_TESTING_GENERATE_TEST_CASE_HPP
#include "../clustering/cluster_node.hpp"
#include "../clustering/tree_builder/tree_builder.hpp"
#include "geometry.hpp"
#include "partition.hpp"
namespace htool {

template <typename T, typename GeneratorTestType>
class TestCaseProduct {
  public:
    char transa;
    char transb;
    std::vector<underlying_type<T>> x1;
    std::vector<underlying_type<T>> x2;
    std::vector<underlying_type<T>> x3;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_1 = nullptr;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_2 = nullptr;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_3 = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_A_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_A_output    = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_B_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_B_output    = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_C_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_C_output    = nullptr;
    std::unique_ptr<GeneratorTestType> operator_A               = nullptr;
    std::unique_ptr<GeneratorTestType> operator_B               = nullptr;
    std::unique_ptr<GeneratorTestType> operator_C               = nullptr;
    int ni_A;
    int no_A;
    int ni_B;
    int no_B;
    int ni_C;
    int no_C;

    TestCaseProduct(char transa0, char transb0, int n1, int n2, int n3, underlying_type<T> z_distance_A, underlying_type<T> z_distance_B, int number_of_partition = -1) : transa(transa0), transb(transb0), x1(3 * n1), x2(3 * n2), x3(3 * n3) {

        // Input sizes
        ni_A = (transa == 'T' || transa == 'C') ? n1 : n2;
        no_A = (transa == 'T' || transa == 'C') ? n2 : n1;
        ni_B = (transb == 'T' || transb == 'C') ? n2 : n3;
        no_B = (transb == 'T' || transb == 'C') ? n3 : n2;
        ni_C = n3;
        no_C = n1;

        ClusterTreeBuilder<underlying_type<T>> recursive_build_strategy;

        // First geometry
        create_disk(3, 0., n1, x1.data());
        if (number_of_partition > 0) {
            std::vector<int> partition;
            test_partition(3, n1, x1, number_of_partition, partition);
            root_cluster_1 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n1, 3, x1.data(), 2, number_of_partition, partition.data()));
        } else {
            root_cluster_1 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n1, 3, x1.data(), 2, 2));
        }

        // Second geometry
        create_disk(3, z_distance_A, n2, x2.data());
        if (number_of_partition > 0) {
            std::vector<int> partition;
            test_partition(3, n2, x2, number_of_partition, partition);
            root_cluster_2 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n2, 3, x2.data(), 2, number_of_partition, partition.data()));
        } else {

            root_cluster_2 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n2, 3, x2.data(), 2, 2));
        }

        // Third geometry
        create_disk(3, z_distance_B, n3, x3.data());
        if (number_of_partition > 0) {
            std::vector<int> partition;
            test_partition(3, n3, x3, number_of_partition, partition);
            root_cluster_3 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n3, 3, x3.data(), 2, number_of_partition, partition.data()));
        } else {
            root_cluster_3 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n3, 3, x3.data(), 2, 2));
        }

        // Operators
        if (transa == 'N') {
            operator_A            = std::make_unique<GeneratorTestType>(3, no_A, ni_A, x1, x2, *root_cluster_1, *root_cluster_2, true, true);
            root_cluster_A_output = root_cluster_1.get();
            root_cluster_A_input  = root_cluster_2.get();
        } else {
            operator_A            = std::make_unique<GeneratorTestType>(3, no_A, ni_A, x2, x1, *root_cluster_2, *root_cluster_1, true, true);
            root_cluster_A_output = root_cluster_2.get();
            root_cluster_A_input  = root_cluster_1.get();
        }
        if (transb == 'N') {
            operator_B            = std::make_unique<GeneratorTestType>(3, no_B, ni_B, x2, x3, *root_cluster_2, *root_cluster_3, true, true);
            root_cluster_B_output = root_cluster_2.get();
            root_cluster_B_input  = root_cluster_3.get();
        } else {
            operator_B            = std::make_unique<GeneratorTestType>(3, no_B, ni_B, x3, x2, *root_cluster_3, *root_cluster_2, true, true);
            root_cluster_B_output = root_cluster_3.get();
            root_cluster_B_input  = root_cluster_2.get();
        }
        operator_C            = std::make_unique<GeneratorTestType>(3, no_C, ni_C, x1, x3, *root_cluster_1, *root_cluster_3, true, true);
        root_cluster_C_input  = root_cluster_3.get();
        root_cluster_C_output = root_cluster_1.get();
    }
};

template <typename T, typename GeneratorTestType>
class TestCaseSymmetricProduct {
  public:
    char side;
    char symmetry;
    char UPLO;
    std::vector<underlying_type<T>> x1;
    std::vector<underlying_type<T>> x2;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_1 = nullptr;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_2 = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_A_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_A_output    = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_B_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_B_output    = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_C_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_C_output    = nullptr;
    std::unique_ptr<GeneratorTestType> operator_A               = nullptr;
    std::unique_ptr<GeneratorTestType> operator_B               = nullptr;
    std::unique_ptr<GeneratorTestType> operator_C               = nullptr;
    int ni_A;
    int no_A;
    int ni_B;
    int no_B;
    int ni_C;
    int no_C;

    TestCaseSymmetricProduct(int n1, int n2, underlying_type<T> z_distance_A, char side0, char symmetry0, char UPLO0, int number_of_partition = -1) : side(side0), symmetry(symmetry0), UPLO(UPLO0), x1(3 * n1), x2(3 * n1) {

        // Input sizes
        ni_A = n1;
        no_A = n1;
        ni_B = (side == 'L') ? n2 : n1;
        no_B = (side == 'L') ? n1 : n2;
        ni_C = (side == 'L') ? n2 : n1;
        no_C = (side == 'L') ? n1 : n2;

        ClusterTreeBuilder<underlying_type<T>> recursive_build_strategy;

        // First geometry
        create_disk(3, 0., n1, x1.data());
        if (number_of_partition > 0) {
            std::vector<int> partition;
            test_partition(3, n1, x1, number_of_partition, partition);
            root_cluster_1 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n1, 3, x1.data(), 2, number_of_partition, partition.data()));
        } else {
            root_cluster_1 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n1, 3, x1.data(), 2, 2));
        }

        // Second geometry
        create_disk(3, z_distance_A, n2, x2.data());
        if (number_of_partition > 0) {
            std::vector<int> partition;
            test_partition(3, n2, x2, number_of_partition, partition);
            root_cluster_2 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n2, 3, x2.data(), 2, number_of_partition, partition.data()));
        } else {

            root_cluster_2 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n2, 3, x2.data(), 2, 2));
        }

        // Operators
        operator_A            = std::make_unique<GeneratorTestType>(3, no_A, ni_A, x1, x1, *root_cluster_1, *root_cluster_1, true, true);
        root_cluster_A_output = root_cluster_1.get();
        root_cluster_A_input  = root_cluster_1.get();

        if (side == 'L') {
            operator_B            = std::make_unique<GeneratorTestType>(3, no_B, ni_B, x1, x2, *root_cluster_1, *root_cluster_2, true, true);
            root_cluster_B_output = root_cluster_1.get();
            root_cluster_B_input  = root_cluster_2.get();

            operator_C            = std::make_unique<GeneratorTestType>(3, no_C, ni_C, x1, x2, *root_cluster_1, *root_cluster_2, true, true);
            root_cluster_C_input  = root_cluster_1.get();
            root_cluster_C_output = root_cluster_2.get();
        } else {
            operator_B            = std::make_unique<GeneratorTestType>(3, no_B, ni_B, x2, x1, *root_cluster_2, *root_cluster_1, true, true);
            root_cluster_B_output = root_cluster_2.get();
            root_cluster_B_input  = root_cluster_1.get();

            operator_C            = std::make_unique<GeneratorTestType>(3, no_C, ni_C, x2, x1, *root_cluster_2, *root_cluster_1, true, true);
            root_cluster_C_input  = root_cluster_2.get();
            root_cluster_C_output = root_cluster_1.get();
        }
    }
};

template <typename T, typename GeneratorTestType>
class TestCaseSymmetricRankUpdate {
  public:
    char trans;
    char symmetry;
    char UPLO;
    std::vector<underlying_type<T>> x1;
    std::vector<underlying_type<T>> x2;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_1 = nullptr;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_2 = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_A_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_A_output    = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_C_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_C_output    = nullptr;
    std::unique_ptr<GeneratorTestType> operator_A               = nullptr;
    std::unique_ptr<GeneratorTestType> operator_C               = nullptr;
    int ni_A;
    int no_A;
    int ni_C;
    int no_C;

    TestCaseSymmetricRankUpdate(int n1, int n2, underlying_type<T> z_distance_A, char trans0, char symmetry0, char UPLO0, int number_of_partition = -1) : trans(trans0), symmetry(symmetry0), UPLO(UPLO0), x1(3 * n1), x2(3 * n1) {

        // Input sizes
        ni_C = n1;
        no_C = n1;
        ni_A = (trans == 'N') ? n1 : n2;
        no_A = (trans == 'N') ? n2 : n1;

        ClusterTreeBuilder<underlying_type<T>> recursive_build_strategy;

        // First geometry
        create_disk(3, 0., n1, x1.data());
        if (number_of_partition > 0) {
            std::vector<int> partition;
            test_partition(3, n1, x1, number_of_partition, partition);
            root_cluster_1 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n1, 3, x1.data(), 2, number_of_partition, partition.data()));
        } else {
            root_cluster_1 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n1, 3, x1.data(), 2, 2));
        }

        // Second geometry
        create_disk(3, z_distance_A, n2, x2.data());
        if (number_of_partition > 0) {
            std::vector<int> partition;
            test_partition(3, n2, x2, number_of_partition, partition);
            root_cluster_2 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n2, 3, x2.data(), 2, number_of_partition, partition.data()));
        } else {

            root_cluster_2 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n2, 3, x2.data(), 2, 2));
        }

        // Operators
        operator_C            = std::make_unique<GeneratorTestType>(3, no_C, ni_C, x1, x1, *root_cluster_1, *root_cluster_1, true, true);
        root_cluster_C_output = root_cluster_1.get();
        root_cluster_C_input  = root_cluster_1.get();

        if (trans == 'N') {
            operator_A            = std::make_unique<GeneratorTestType>(3, no_A, ni_A, x1, x2, *root_cluster_1, *root_cluster_2, true, true);
            root_cluster_A_output = root_cluster_1.get();
            root_cluster_A_input  = root_cluster_2.get();
        } else {
            operator_A            = std::make_unique<GeneratorTestType>(3, no_A, ni_A, x2, x1, *root_cluster_2, *root_cluster_1, true, true);
            root_cluster_A_output = root_cluster_2.get();
            root_cluster_A_input  = root_cluster_1.get();
        }
    }
};

template <typename T, typename GeneratorTestType>
class TestCaseSolve {
  public:
    char side;
    char trans;
    std::vector<underlying_type<T>> x1;
    std::vector<underlying_type<T>> x2;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_1 = nullptr;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_2 = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_A_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_A_output    = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_X_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_X_output    = nullptr;
    std::unique_ptr<GeneratorTestType> operator_A               = nullptr;
    std::unique_ptr<GeneratorTestType> operator_X               = nullptr;
    int ni_A;
    int no_A;
    int ni_X;
    int no_X;

    TestCaseSolve(char side0, char trans0, int n1, int n2, underlying_type<T> z_distance_A, int number_of_partition = -1) : side(side0), trans(trans0), x1(3 * n1), x2(3 * n2) {

        // Input sizes
        ni_A = n1;
        no_A = n1;
        ni_X = (side == 'L') ? n1 : n2;
        no_X = (side == 'L') ? n2 : n1;

        ClusterTreeBuilder<underlying_type<T>> recursive_build_strategy;

        // First geometry
        create_disk(3, 0., n1, x1.data());
        if (number_of_partition > 0) {
            std::vector<int> partition;
            test_partition(3, n1, x1, number_of_partition, partition);
            root_cluster_1 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n1, 3, x1.data(), 2, number_of_partition, partition.data()));
        } else {
            root_cluster_1 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n1, 3, x1.data(), 2, 2));
        }

        // Second geometry
        create_disk(3, z_distance_A, n2, x2.data());
        if (number_of_partition > 0) {
            std::vector<int> partition;
            test_partition(3, n2, x2, number_of_partition, partition);
            root_cluster_2 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n2, 3, x2.data(), 2, number_of_partition, partition.data()));
        } else {
            root_cluster_2 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n2, 3, x2.data(), 2, 2));
        }

        // Operators
        operator_A            = std::make_unique<GeneratorTestType>(3, no_A, ni_A, x1, x1, *root_cluster_1, *root_cluster_1, true, true);
        root_cluster_A_output = root_cluster_1.get();
        root_cluster_A_input  = root_cluster_1.get();

        if (side == 'L') {
            operator_X            = std::make_unique<GeneratorTestType>(3, no_X, ni_X, x1, x2, *root_cluster_1, *root_cluster_2, true, true);
            root_cluster_X_output = root_cluster_1.get();
            root_cluster_X_input  = root_cluster_2.get();
        } else {
            operator_X            = std::make_unique<GeneratorTestType>(3, no_X, ni_X, x2, x1, *root_cluster_2, *root_cluster_1, true, true);
            root_cluster_X_output = root_cluster_2.get();
            root_cluster_X_input  = root_cluster_1.get();
        }
    }
};

template <typename T, typename GeneratorTestType>
class TestCaseAddition {
  public:
    std::vector<underlying_type<T>> x1;
    std::vector<underlying_type<T>> x2;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_1 = nullptr;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_2 = nullptr;

    const Cluster<underlying_type<T>> *root_cluster_A_input  = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_A_output = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_B_input  = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_B_output = nullptr;
    std::unique_ptr<GeneratorTestType> operator_A            = nullptr;
    std::unique_ptr<GeneratorTestType> operator_B            = nullptr;
    int ni_A;
    int no_A;
    int ni_B;
    int no_B;

    TestCaseAddition(int n1, int n2, underlying_type<T> z_distance_A, int number_of_partition = -1) : x1(3 * n1), x2(3 * n2) {

        // Input sizes
        ni_A = n1;
        no_A = n2;

        ClusterTreeBuilder<underlying_type<T>> recursive_build_strategy;

        // First geometry
        create_disk(3, 0., n1, x1.data());
        if (number_of_partition > 0) {
            std::vector<int> partition;
            test_partition(3, n1, x1, number_of_partition, partition);
            root_cluster_1 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n1, 3, x1.data(), 2, number_of_partition, partition.data()));
        } else {
            root_cluster_1 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n1, 3, x1.data(), 2, 2));
        }

        // Second geometry
        create_disk(3, z_distance_A, n2, x2.data());
        if (number_of_partition > 0) {
            std::vector<int> partition;
            test_partition(3, n2, x2, number_of_partition, partition);
            root_cluster_2 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n2, 3, x2.data(), 2, number_of_partition, partition.data()));
        } else {

            root_cluster_2 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n2, 3, x2.data(), 2, 2));
        }

        // Operators
        operator_A            = std::make_unique<GeneratorTestType>(3, no_A, ni_A, x1, x2, *root_cluster_1, *root_cluster_2, true, true);
        root_cluster_A_output = root_cluster_1.get();
        root_cluster_A_input  = root_cluster_2.get();

        // Sub lrmat two level deep
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, 1);
        int first_target_index  = dist(gen);
        int second_target_index = dist(gen);
        int first_source_index  = dist(gen);
        int second_source_index = dist(gen);

        const Cluster<htool::underlying_type<T>> &sub_target_cluster = *root_cluster_A_output->get_children()[first_target_index]->get_children()[second_target_index];
        const Cluster<htool::underlying_type<T>> &sub_source_cluster = *root_cluster_A_input->get_children()[first_source_index]->get_children()[second_source_index];

        no_B = sub_target_cluster.get_size();
        ni_B = sub_source_cluster.get_size();

        root_cluster_B_input  = &sub_source_cluster;
        root_cluster_B_output = &sub_target_cluster;
    }
};
} // namespace htool
#endif
