#ifndef HTOOL_HMATRIX_LINALG_FACTORIZATION_HPP
#define HTOOL_HMATRIX_LINALG_FACTORIZATION_HPP

#include "../../matrix/linalg/factorization.hpp"
#include "add_hmatrix_hmatrix_product.hpp"
#include "triangular_hmatrix_hmatrix_solve.hpp"

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision>
void lu_factorization(HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    if (hmatrix.is_hierarchical()) {

        bool block_tree_not_consistent = (hmatrix.get_target_cluster().get_rank() < 0 || hmatrix.get_source_cluster().get_rank() < 0);
        std::vector<const Cluster<CoordinatePrecision> *> clusters;
        const Cluster<CoordinatePrecision> &cluster = hmatrix.get_target_cluster();

        if (cluster.is_leaf() || (block_tree_not_consistent and cluster.get_rank() >= 0)) {
            clusters.push_back(&cluster);
        } else if (block_tree_not_consistent) {
            for (auto &output_cluster_child : cluster.get_clusters_on_partition()) {
                clusters.push_back(output_cluster_child);
            }
        } else {
            for (auto &output_cluster_child : cluster.get_children()) {
                clusters.push_back(output_cluster_child.get());
            }
        }

        for (auto &cluster_child : clusters) {
            HMatrix<CoefficientPrecision, CoordinatePrecision> *pivot = hmatrix.get_sub_hmatrix(*cluster_child, *cluster_child);
            // Compute pivot block
            lu_factorization(*pivot);

            // Apply pivot block to row and column
            for (auto &other_cluster_child : clusters) {
                if (other_cluster_child->get_offset() > cluster_child->get_offset()) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *U = hmatrix.get_sub_hmatrix(*cluster_child, *other_cluster_child);
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*other_cluster_child, *cluster_child);
                    triangular_hmatrix_hmatrix_solve('L', 'L', 'N', 'U', CoefficientPrecision(1), *pivot, *U);
                    triangular_hmatrix_hmatrix_solve('R', 'U', 'N', 'N', CoefficientPrecision(1), *pivot, *L);
                }
            }

            // Update Schur complement
            for (auto &output_cluster_child : clusters) {
                for (auto &input_cluster_child : clusters) {
                    if (output_cluster_child->get_offset() > cluster_child->get_offset() && input_cluster_child->get_offset() > cluster_child->get_offset()) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = hmatrix.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);
                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *U = hmatrix.get_sub_hmatrix(*cluster_child, *input_cluster_child);
                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*output_cluster_child, *cluster_child);

                        add_hmatrix_hmatrix_product('N', 'N', CoefficientPrecision(-1), *L, *U, CoefficientPrecision(1), *A_child);
                    }
                }
            }
        }
    } else if (hmatrix.is_dense()) {
        lu_factorization(*hmatrix.get_dense_data());
    } else {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for lu_factorization (hmatrix is low-rank)"); // LCOV_EXCL_LINE
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void lu_solve(char trans, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &X) {

    if (trans == 'N') {
        triangular_hmatrix_matrix_solve('L', 'L', 'N', 'U', CoefficientPrecision(1), A, X);
        triangular_hmatrix_matrix_solve('L', 'U', 'N', 'N', CoefficientPrecision(1), A, X);
    } else {
        triangular_hmatrix_matrix_solve('L', 'U', trans, 'N', CoefficientPrecision(1), A, X);
        triangular_hmatrix_matrix_solve('L', 'L', trans, 'U', CoefficientPrecision(1), A, X);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void cholesky_factorization(char UPLO, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    if (hmatrix.is_hierarchical()) {

        bool block_tree_not_consistent = (hmatrix.get_target_cluster().get_rank() < 0 || hmatrix.get_source_cluster().get_rank() < 0);
        std::vector<const Cluster<CoordinatePrecision> *> clusters;
        const Cluster<CoordinatePrecision> &cluster = hmatrix.get_target_cluster();

        if (cluster.is_leaf() || (block_tree_not_consistent and cluster.get_rank() >= 0)) {
            clusters.push_back(&cluster);
        } else if (block_tree_not_consistent) {
            for (auto &output_cluster_child : cluster.get_clusters_on_partition()) {
                clusters.push_back(output_cluster_child);
            }
        } else {
            for (auto &output_cluster_child : cluster.get_children()) {
                clusters.push_back(output_cluster_child.get());
            }
        }

        for (auto &cluster_child : clusters) {
            HMatrix<CoefficientPrecision, CoordinatePrecision> *pivot = hmatrix.get_sub_hmatrix(*cluster_child, *cluster_child);
            // Compute pivot block
            cholesky_factorization(UPLO, *pivot);

            // Apply pivot block to row and column
            for (auto &other_cluster_child : clusters) {
                if (other_cluster_child->get_offset() > cluster_child->get_offset()) {
                    if (UPLO == 'L') {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*other_cluster_child, *cluster_child);
                        triangular_hmatrix_hmatrix_solve('R', UPLO, 'C', 'N', CoefficientPrecision(1), *pivot, *L);
                    } else {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *U = hmatrix.get_sub_hmatrix(*cluster_child, *other_cluster_child);
                        triangular_hmatrix_hmatrix_solve('L', UPLO, 'C', 'N', CoefficientPrecision(1), *pivot, *U);
                    }
                }
            }

            // Update Schur complement
            for (auto &output_cluster_child : clusters) {
                for (auto &input_cluster_child : clusters) {
                    if (UPLO == 'L' && output_cluster_child->get_offset() > cluster_child->get_offset() && input_cluster_child->get_offset() > cluster_child->get_offset() && output_cluster_child->get_offset() >= input_cluster_child->get_offset()) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = hmatrix.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);
                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*output_cluster_child, *cluster_child);

                        // if (*output_cluster_child == *input_cluster_child) {
                        //     symmetric_rank_k_update(UPLO, 'N', CoefficientPrecision(-1), *L, CoefficientPrecision(1), *A_child);
                        // } else {
                        add_hmatrix_hmatrix_product('N', 'C', CoefficientPrecision(-1), *L, *L, CoefficientPrecision(1), *A_child);
                        // }
                    } else if (UPLO == 'U' && output_cluster_child->get_offset() > cluster_child->get_offset() && input_cluster_child->get_offset() > cluster_child->get_offset() && input_cluster_child->get_offset() >= output_cluster_child->get_offset()) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = hmatrix.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);
                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *U = hmatrix.get_sub_hmatrix(*cluster_child, *input_cluster_child);
                        // if (*output_cluster_child == *input_cluster_child) {
                        //     symmetric_rank_k_update(UPLO, 'C', CoefficientPrecision(-1), *U, CoefficientPrecision(1), *A_child);
                        // } else {
                        add_hmatrix_hmatrix_product('C', 'N', CoefficientPrecision(-1), *U, *U, CoefficientPrecision(1), *A_child);
                        // }
                    }
                }
            }
        }
    } else if (hmatrix.is_dense()) {
        cholesky_factorization(UPLO, *hmatrix.get_dense_data());
    } else {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for cholesky_factorization (hmatrix is low-rank)"); // LCOV_EXCL_LINE
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void cholesky_solve(char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &X) {
    if (UPLO == 'L') {
        triangular_hmatrix_matrix_solve('L', 'L', 'N', 'N', CoefficientPrecision(1), A, X);
        triangular_hmatrix_matrix_solve('L', 'L', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), A, X);
    } else {
        triangular_hmatrix_matrix_solve('L', 'U', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), A, X);
        triangular_hmatrix_matrix_solve('L', 'U', 'N', 'N', CoefficientPrecision(1), A, X);
    }
}

} // namespace htool

#endif
