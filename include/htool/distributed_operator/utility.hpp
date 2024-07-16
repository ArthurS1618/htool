#ifndef HTOOL_DISTRIBUTED_OPERATOR_UTILITY_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_UTILITY_HPP

#include "../hmatrix/hmatrix_distributed_output.hpp"
#include "../local_operators/local_hmatrix.hpp"
#include "distributed_operator.hpp"
#include "implementations/partition_from_cluster.hpp"
namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision>
class DistributedOperatorFromHMatrix {
  private:
    const PartitionFromCluster<CoefficientPrecision, CoordinatePrecision> target_partition, source_partition;
    std::function<int(MPI_Comm)> get_rankWorld = [](MPI_Comm comm) {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld); 
    return rankWorld; };

  public:
    const HMatrix<CoefficientPrecision, CoordinatePrecision> hmatrix;

  private:
    const LocalHMatrix<CoefficientPrecision, CoordinatePrecision> local_hmatrix;

  public:
    DistributedOperator<CoefficientPrecision> distributed_operator;
    const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix{nullptr};

    DistributedOperatorFromHMatrix(const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision> &hmatrix_builder, MPI_Comm communicator) : target_partition(target_cluster), source_partition(source_cluster), hmatrix(hmatrix_builder.build(generator)), local_hmatrix(hmatrix, hmatrix_builder.get_target_cluster().get_cluster_on_partition(get_rankWorld(communicator)), hmatrix_builder.get_source_cluster(), hmatrix_builder.get_symmetry(), hmatrix_builder.get_UPLO(), false, false), distributed_operator(target_partition, source_partition, hmatrix_builder.get_symmetry(), hmatrix_builder.get_UPLO(), communicator) {
        distributed_operator.add_local_operator(&local_hmatrix);
        block_diagonal_hmatrix = hmatrix.get_sub_hmatrix(hmatrix_builder.get_target_cluster().get_cluster_on_partition(get_rankWorld(communicator)), hmatrix_builder.get_source_cluster().get_cluster_on_partition(get_rankWorld(communicator)));
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
class DefaultApproximationBuilder {
  private:
    std::function<int(MPI_Comm)> get_rankWorld = [](MPI_Comm comm) {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld); 
    return rankWorld; };
    DistributedOperatorFromHMatrix<CoefficientPrecision, CoordinatePrecision> distributed_operator_builder;

  public:
    const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix;

  public:
    DistributedOperator<CoefficientPrecision> &distributed_operator;
    const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix{nullptr};

    DefaultApproximationBuilder(const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, htool::underlying_type<CoefficientPrecision> epsilon, htool::underlying_type<CoefficientPrecision> eta, char symmetry, char UPLO, MPI_Comm communicator) : distributed_operator_builder(generator, target_cluster, source_cluster, HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>(target_cluster, source_cluster, epsilon, eta, symmetry, UPLO, -1, get_rankWorld(communicator), get_rankWorld(communicator)), communicator), hmatrix(distributed_operator_builder.hmatrix), distributed_operator(distributed_operator_builder.distributed_operator), block_diagonal_hmatrix(distributed_operator_builder.block_diagonal_hmatrix) {}
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
class DefaultLocalApproximationBuilder {
  private:
    std::function<int(MPI_Comm)> get_rankWorld = [](MPI_Comm comm) {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld); 
    return rankWorld; };
    DistributedOperatorFromHMatrix<CoefficientPrecision, CoordinatePrecision> distributed_operator_builder;

  public:
    const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix;

  public:
    DistributedOperator<CoefficientPrecision> &distributed_operator;
    const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix{nullptr};

  public:
    DefaultLocalApproximationBuilder(const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, htool::underlying_type<CoefficientPrecision> epsilon, htool::underlying_type<CoefficientPrecision> eta, char symmetry, char UPLO, MPI_Comm communicator) : distributed_operator_builder(generator, target_cluster, source_cluster, HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>(target_cluster.get_cluster_on_partition(get_rankWorld(communicator)), source_cluster.get_cluster_on_partition(get_rankWorld(communicator)), epsilon, eta, symmetry, UPLO, -1, get_rankWorld(communicator), get_rankWorld(communicator)), communicator), hmatrix(distributed_operator_builder.hmatrix), distributed_operator(distributed_operator_builder.distributed_operator), block_diagonal_hmatrix(distributed_operator_builder.block_diagonal_hmatrix) {}
};

} // namespace htool
#endif
