#ifndef HTOOL_HMATRIX_HPP
#define HTOOL_HMATRIX_HPP

#if defined(_OPENMP)
#    include <omp.h>
#endif
#include "../basic_types/tree.hpp"
#include "../clustering/cluster_node.hpp"
#include "../matrix/linalg/add_matrix_matrix_product.hpp"
#include "../matrix/linalg/factorization.hpp"
#include "../misc/logger.hpp"
#include "hmatrix_tree_data.hpp"
#include "interfaces/virtual_admissibility_condition.hpp"
#include "interfaces/virtual_dense_blocks_generator.hpp"
#include "interfaces/virtual_generator.hpp"
#include "lrmat/lrmat.hpp"
#include "sumexpr.hpp"

#include <queue>

namespace htool {

// Class
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class HMatrix : public TreeNode<HMatrix<CoefficientPrecision, CoordinatePrecision>, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>> {
  public:
    enum class StorageType {
        Dense,
        LowRank,
        Hierarchical
    };

  private:
    // Data members
    const Cluster<CoordinatePrecision> *m_target_cluster, *m_source_cluster; // child's clusters are non owning
    char m_symmetry{'N'};
    char m_UPLO{'N'};

    std::unique_ptr<Matrix<CoefficientPrecision>> m_dense_data{nullptr};
    std::unique_ptr<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> m_low_rank_data{nullptr};

    // Cached leaves
    mutable std::vector<const HMatrix *> m_leaves{};
    mutable std::vector<const HMatrix *> m_leaves_for_symmetry{};
    mutable char m_symmetry_type_for_leaves{'N'};
    mutable char m_UPLO_for_leaves{'N'};

    StorageType m_storage_type{StorageType::Hierarchical};

    void set_leaves_in_cache() const {
        if (m_leaves.empty()) {
            std::stack<std::pair<const HMatrix<CoefficientPrecision, CoordinatePrecision> *, bool>> hmatrix_stack;
            hmatrix_stack.push(std::pair<const HMatrix<CoefficientPrecision, CoordinatePrecision> *, bool>(this, m_symmetry != 'N'));

            while (!hmatrix_stack.empty()) {
                auto &current_element = hmatrix_stack.top();
                hmatrix_stack.pop();
                const HMatrix<CoefficientPrecision, CoordinatePrecision> *current_hmatrix = current_element.first;
                bool has_symmetric_ancestor                                               = current_element.second;

                if (current_hmatrix->is_leaf()) {
                    m_leaves.push_back(current_hmatrix);

                    if (has_symmetric_ancestor && current_hmatrix->get_target_cluster().get_offset() != current_hmatrix->get_source_cluster().get_offset()) {
                        m_leaves_for_symmetry.push_back(current_hmatrix);
                    }
                }

                if (m_symmetry_type_for_leaves == 'N' && current_hmatrix->get_symmetry() != 'N') {
                    m_symmetry_type_for_leaves = current_hmatrix->get_symmetry();
                    m_UPLO_for_leaves          = current_hmatrix->get_UPLO();
                }

                for (const auto &child : current_hmatrix->get_children()) {
                    hmatrix_stack.push(std::pair<const HMatrix<CoefficientPrecision, CoordinatePrecision> *, bool>(child.get(), current_hmatrix->get_symmetry() != 'N' || has_symmetric_ancestor));
                }
            }
        }
    }

    void
    threaded_hierarchical_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const;
    void threaded_hierarchical_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const;
    void recursive_build_hmatrix_product_fast(const SumExpression_fast<CoefficientPrecision, CoordinatePrecision> &sumepr, std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> *adm);
    void recursive_build_hmatrix_product_fast_triangulaire(char transa, char transb, const SumExpression_fast<CoefficientPrecision, CoordinatePrecision> &sumepr, std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> *adm);

  public:
    // Root constructor
    HMatrix(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster) : TreeNode<HMatrix, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(), m_target_cluster(&target_cluster), m_source_cluster(&source_cluster) {
    }

    HMatrix(std::shared_ptr<const Cluster<CoordinatePrecision>> target_cluster, std::shared_ptr<const Cluster<CoordinatePrecision>> source_cluster) : TreeNode<HMatrix, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(), m_target_cluster(target_cluster.get()), m_source_cluster(source_cluster.get()) {
        this->m_tree_data->m_target_cluster_tree = target_cluster;
        this->m_tree_data->m_source_cluster_tree = source_cluster;
    }

    // Child constructor
    HMatrix(const HMatrix &parent, const Cluster<CoordinatePrecision> *target_cluster, const Cluster<CoordinatePrecision> *source_cluster) : TreeNode<HMatrix, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(parent), m_target_cluster(target_cluster), m_source_cluster(source_cluster) {}

    HMatrix(const HMatrix &rhs) : TreeNode<HMatrix<CoefficientPrecision, CoordinatePrecision>, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(rhs), m_target_cluster(rhs.m_target_cluster), m_source_cluster(rhs.m_source_cluster), m_symmetry(rhs.m_symmetry), m_UPLO(rhs.m_UPLO), m_leaves(), m_leaves_for_symmetry(), m_symmetry_type_for_leaves(), m_storage_type(rhs.m_storage_type) {
        Logger::get_instance().log(LogLevel::INFO, "Deep copy of HMatrix");
        this->m_depth     = rhs.m_depth;
        this->m_is_root   = rhs.m_is_root;
        this->m_tree_data = std::make_shared<HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(*rhs.m_tree_data);
        this->m_children.clear();
        for (auto &child : rhs.m_children) {
            this->m_children.emplace_back(std::make_unique<HMatrix<CoefficientPrecision, CoordinatePrecision>>(*child));
        }
        if (rhs.m_dense_data) {
            m_dense_data = std::make_unique<Matrix<CoefficientPrecision>>(*rhs.m_dense_data);
        }
        if (rhs.m_low_rank_data) {
            m_low_rank_data = std::make_unique<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(*rhs.m_low_rank_data);
        }
    }
    HMatrix &operator=(const HMatrix &rhs) {
        Logger::get_instance().log(LogLevel::INFO, "Deep copy of HMatrix");
        if (&rhs == this) {
            return *this;
        }
        this->m_depth     = rhs.m_depth;
        this->m_is_root   = rhs.m_is_root;
        this->m_tree_data = std::make_shared<HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(*rhs.m_tree_data);
        this->m_children.clear();
        for (auto &child : rhs.m_children) {
            this->m_children.emplace_back(std::make_unique<HMatrix<CoefficientPrecision, CoordinatePrecision>>(*child));
        }
        m_target_cluster = rhs.m_target_cluster;
        m_source_cluster = rhs.m_source_cluster;
        m_symmetry       = rhs.m_symmetry;
        m_UPLO           = rhs.m_UPLO;
        m_storage_type   = rhs.m_storage_type;

        if (rhs.m_dense_data) {
            m_dense_data = std::make_unique<Matrix<CoefficientPrecision>>(*rhs.m_dense_data);
        }
        if (rhs.m_low_rank_data) {
            m_low_rank_data = std::make_unique<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(*rhs.m_low_rank_data);
        }
        m_leaves.clear();
        m_leaves_for_symmetry.clear();
        return *this;
    }

    HMatrix(HMatrix &&) noexcept            = default;
    HMatrix &operator=(HMatrix &&) noexcept = default;
    virtual ~HMatrix()                      = default;

    // HMatrix getters
    const Cluster<CoordinatePrecision> &get_target_cluster() const { return *m_target_cluster; }
    const Cluster<CoordinatePrecision> &get_source_cluster() const { return *m_source_cluster; }
    int nb_cols() const { return m_source_cluster->get_size(); }
    int nb_rows() const { return m_target_cluster->get_size(); }
    htool::underlying_type<CoefficientPrecision> get_epsilon() const { return this->m_tree_data->m_epsilon; }
    void set_low_rank_data(const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> &lrmat) {
        m_low_rank_data = std::unique_ptr<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(new LowRankMatrix<CoefficientPrecision, CoordinatePrecision>(lrmat));
        m_storage_type  = StorageType::LowRank;
    }
    HMatrix<CoefficientPrecision, CoordinatePrecision> *get_child_or_this(const Cluster<CoordinatePrecision> &required_target_cluster, const Cluster<CoordinatePrecision> &required_source_cluster) {
        if (*m_target_cluster == required_target_cluster and *m_source_cluster == required_source_cluster) {
            return this;
        }
        for (auto &child : this->m_children) {
            if (child->get_target_cluster() == required_target_cluster and child->get_source_cluster() == required_source_cluster) {
                return child.get();
            }
        }
        return nullptr;
    }

    const HMatrix<CoefficientPrecision, CoordinatePrecision> *get_child_or_this(const Cluster<CoordinatePrecision> &required_target_cluster, const Cluster<CoordinatePrecision> &required_source_cluster) const {
        if (*m_target_cluster == required_target_cluster and *m_source_cluster == required_source_cluster) {
            return this;
        }
        for (auto &child : this->m_children) {
            if (child->get_target_cluster() == required_target_cluster and child->get_source_cluster() == required_source_cluster) {
                return child.get();
            }
        }
        return nullptr;
    }
    void save_plot(const std::string &outputname) const;

    int get_rank() const {
        return m_storage_type == StorageType::LowRank ? m_low_rank_data->rank_of() : -1;
    }
    const std::vector<const HMatrix *> &get_leaves() const {
        set_leaves_in_cache();
        return m_leaves;
    }
    const std::vector<const HMatrix *> &get_leaves_for_symmetry() const {
        set_leaves_in_cache();
        return m_leaves_for_symmetry;
    }
    const Matrix<CoefficientPrecision> *get_dense_data() const { return m_dense_data.get(); }
    Matrix<CoefficientPrecision> *get_dense_data() { return m_dense_data.get(); }
    const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> *get_low_rank_data() const { return m_low_rank_data.get(); }
    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> *get_low_rank_data() { return m_low_rank_data.get(); }
    char get_symmetry() const { return m_symmetry; }
    char get_UPLO() const { return m_UPLO; }
    const HMatrixTreeData<CoefficientPrecision, CoordinatePrecision> *get_hmatrix_tree_data() const { return this->m_tree_data.get(); }
    const HMatrix<CoefficientPrecision> *get_sub_hmatrix(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster) const {
        std::queue<const HMatrix<CoefficientPrecision> *> hmatrix_queue;
        hmatrix_queue.push(this);

        while (!hmatrix_queue.empty()) {
            const HMatrix<CoefficientPrecision> *current_hmatrix = hmatrix_queue.front();
            hmatrix_queue.pop();

            if (target_cluster == current_hmatrix->get_target_cluster() && source_cluster == current_hmatrix->get_source_cluster()) {
                return current_hmatrix;
            }

            const auto &children = current_hmatrix->get_children();
            for (auto &child : children) {
                hmatrix_queue.push(child.get());
            }
        }
        return nullptr;
    }
    HMatrix<CoefficientPrecision> *get_sub_hmatrix(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster) {
        std::queue<HMatrix<CoefficientPrecision> *> hmatrix_queue;
        hmatrix_queue.push(this);

        while (!hmatrix_queue.empty()) {
            HMatrix<CoefficientPrecision> *current_hmatrix = hmatrix_queue.front();
            hmatrix_queue.pop();

            if (target_cluster == current_hmatrix->get_target_cluster() && source_cluster == current_hmatrix->get_source_cluster()) {
                return current_hmatrix;
            }

            auto &children = current_hmatrix->get_children();
            for (auto &child : children) {
                hmatrix_queue.push(child.get());
            }
        }
        return nullptr;
    }
    StorageType get_storage_type() const { return m_storage_type; }

    // HMatrix node setters
    void set_symmetry(char symmetry) { m_symmetry = symmetry; }
    void set_UPLO(char UPLO) { m_UPLO = UPLO; }
    void set_target_cluster(const Cluster<CoordinatePrecision> *new_target_cluster) { m_target_cluster = new_target_cluster; }

    // Test properties
    bool is_dense() const { return m_storage_type == StorageType::Dense; }
    bool is_low_rank() const { return m_storage_type == StorageType::LowRank; }
    bool is_hierarchical() const { return m_storage_type == StorageType::Hierarchical; }

    // HMatrix Tree setters
    void set_eta(CoordinatePrecision eta) { this->m_tree_data->m_eta = eta; }
    void set_epsilon(underlying_type<CoefficientPrecision> epsilon) { this->m_tree_data->m_epsilon = epsilon; }
    void set_low_rank_generator(std::shared_ptr<VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>> ptr) { this->m_tree_data->m_low_rank_generator = ptr; }
    void set_admissibility_condition(std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> ptr) { this->m_tree_data->m_admissibility_condition = ptr; }
    void set_maximal_block_size(int maxblock_size) { this->m_tree_data->m_maxblocksize = maxblock_size; }
    void set_minimal_target_depth(unsigned int minimal_target_depth) { this->m_tree_data->m_minimal_target_depth = minimal_target_depth; }
    void set_minimal_source_depth(unsigned int minimal_source_depth) { this->m_tree_data->m_minimal_source_depth = minimal_source_depth; }

    // HMatrix Tree setters
    char get_symmetry_for_leaves() const { return m_symmetry_type_for_leaves; }

    bool compute_admissibility(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) {
        return this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(t, s, this->m_tree_data->m_eta);
    }
    std::shared_ptr<VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>> get_low_rank_generator() { return this->m_tree_data->m_low_rank_generator; }

    std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> get_admissibility_condition() { return this->m_tree_data->m_admissibility_condition; }

    char get_UPLO_for_leaves() const { return m_UPLO_for_leaves; }

    // Data computation
    void compute_dense_data(const VirtualGenerator<CoefficientPrecision> &generator) {
        m_dense_data = std::make_unique<Matrix<CoefficientPrecision>>(m_target_cluster->get_size(), m_source_cluster->get_size());
        generator.copy_submatrix(m_target_cluster->get_size(), m_source_cluster->get_size(), m_target_cluster->get_offset(), m_source_cluster->get_offset(), m_dense_data->data());
        m_storage_type = StorageType::Dense;
    }

    void compute_low_rank_data(const VirtualGenerator<CoefficientPrecision> &generator, const VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision> &low_rank_generator, int reqrank, underlying_type<CoefficientPrecision> epsilon) {
        m_low_rank_data = std::make_unique<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(generator, low_rank_generator, *m_target_cluster, *m_source_cluster, reqrank, epsilon);
        m_storage_type  = StorageType::LowRank;
    }
    void clear_low_rank_data() { m_low_rank_data.reset(); }

    void set_dense_data(std::unique_ptr<Matrix<CoefficientPrecision>> dense_matrix_ptr) {
        this->delete_children();
        m_leaves.clear();
        m_leaves_for_symmetry.clear();
        m_dense_data = std::move(dense_matrix_ptr);
        // m_dense_data = std::make_unique<Matrix<CoefficientPrecision>>();
        // m_dense_data->assign(dense_matrix.nb_rows(), dense_matrix.nb_cols(), dense_matrix.release(), true);
        m_storage_type = StorageType::Dense;
    }
    // je rajoute ca pour faire du coarsenning
    // void ACA(const int &size_t, const int &size_s, const int &offset_t, const int &offset_s, const double &epsilon, bool flag_out) {
    //     auto bloc   = *this->get_block(size_t, size_s, offset_t, offset_s);
    //     double flag = 1e20;
    //     int rk      = 0;
    //     int i0      = 0;
    //     int j0      = 0;
    //     std::vector<int> index;
    //     std::vector<int> index_left;
    //     index.push_back(0);
    //     double val = 0;
    //     std::vector<int> index;
    //     while ((flag > (size_t - rk) * epsilon) && (rk < size_t / 2)) {
    //         index.push_back(i0);
    //         std::vector<CoefficientPrecision> ei(size_t);
    //         ei[rk] = 1.0;
    //         std::vector<CoefficientPrecision> row_i(size_s);
    //         bloc.add_vector_product('T', 1.0, ei.data(), 1.0, row_i.data());
    //         auto max_row = std::max_element(row_i.begin(), row_i.end());
    //         j0           = std::distance(row_i.begin(), max_row);
    //         std::vector<CoefficientPrecision> ej(size_t);
    //         ej[j0] = 1.0;
    //         std::vector<CoefficientPrecision> col_j(size_t);
    //         bloc.add_vector_product('N', 1.0, ej.data(), 1.0, col_j.data());
    //         max_col = std::max_element(col_j.begin(), col_j.end());
    //         i0      = std::distance(col_j.begin(), max_col);
    //         val     = col_j[i0];
    //         if (std::abs(val) < 1e-20) {
    //             flag = 0;
    //         } else {
    //         }
    //     }
    // }

    // void random_ACA(const int &size_t, const int &size_s, const int &off_t, const int &off_s, Matrix<CoefficientPrecision> &L, Matrix<CoefficientPrecision> &Ut, double tol, bool flag_out) {
    //     auto bloc    = this->get_block(size_t, size_s, off_t, off_s);
    //     double norme = 1e20;
    //     int rk       = 0;
    //     Matrix<CoefficientPrecision> bloc_dense(size_t, size_s);
    //     copy_to_dense(*bloc, bloc_dense.data());
    //     Matrix<CoefficientPrecision> temp(size_t, size_t);
    //     for (int k = 0; k < size_t; ++k) {
    //         temp(k, k) = 1.0;
    //     }
    //     while (norme > (tol * (size_s - rk)) && (rk < std::min(size_t / 2, size_s / 2))) {
    //         std::cout << "norme = " << norme << std::endl;
    //         auto w = generateGaussianVector(size_s);
    //         std::vector<CoefficientPrecision> lrtemp(size_t);
    //         bloc->add_vector_product('N', 1.0, w.data(), 1.0, lrtemp.data());
    //         std::vector<CoefficientPrecision> lr(size_t);
    //         temp.add_vector_product('N', 1.0, lrtemp.data(), 1.0, lr.data());
    //         norme = norm2(lr);
    //         lr    = mult(1.0 / norme, lr);
    //         rk += 1;
    //         std::vector<CoefficientPrecision> new_Ldata(size_t * rk);
    //         std::copy(L.data(), L.data() + (size_t * (rk - 1)), new_Ldata.data());
    //         L.assign(size_t, rk, new_Ldata.data(), false);
    //         L.set_col(rk - 1, lr);
    //         std::vector<CoefficientPrecision> new_Udata(size_s * rk);
    //         std::copy(Ut.data(), Ut.data() + (size_s * (rk - 1)), new_Udata.data());
    //         Ut.assign(size_s, rk, new_Udata.data(), false);
    //         std::vector<CoefficientPrecision> ur(size_s);
    //         bloc->add_vector_product('T', 1.0, lr.data(), 1.0, ur.data());
    //         Ut.set_col(rk - 1, ur);
    //         Matrix<CoefficientPrecision> ltemp(size_t, 1);
    //         ltemp.assign(size_t, 1, lr.data(), false);
    //         // temp = temp- l* l^T
    //         double alpha = -1.0;
    //         double beta  = 1.0;
    //         int kk       = 1;
    //         int ld       = size_t;
    //         std::cout << "aprés L et U : " << normFrob(bloc_dense - L * transp(Ut)) << std::endl;

    //         Blas<CoefficientPrecision>::gemm("N", "T", &ld, &ld, &kk, &alpha, ltemp.data(), &ld, ltemp.data(), &ld, &beta, temp.data(), &ld);
    //     }
    //     if (rk == std::min(size_t / 2, size_s / 2)) {
    //         flag_out = false;
    //     } else {
    //         flag_out = true;
    //         Matrix<CoefficientPrecision> U(rk, size_s);
    //         for (int k = 0; k < rk; ++k) {
    //             for (int l = 0; l < size_s; ++l) {
    //                 U(k, l) = Ut(l, k);
    //             }
    //         }
    //         bloc->delete_children();
    //         LowRankMatrix<CoefficientPrecision, CoordinatePrecision> LU(L, U);
    //         bloc->set_low_rank_data(LU);
    //         Ut = U;
    //     }
    // // }
    // void random_ACA(const int &size_t, const int &size_s, const int &off_t, const int &off_s, double tol, bool flag_out) {
    //     auto bloc    = this->get_block(size_t, size_s, off_t, off_s);
    //     double norme = 1e20;
    //     int rk       = 1;
    //     Matrix<CoefficientPrecision> L(size_t, 1);
    //     Matrix<CoefficientPrecision> Ut(size_s, 1);
    //     Matrix<CoefficientPrecision> bloc_dense(size_t, size_s);
    //     copy_to_dense(*bloc, bloc_dense.data());
    //     Matrix<CoefficientPrecision> temp(size_t, size_t);
    //     for (int k = 0; k < size_t; ++k) {
    //         temp(k, k) = 1.0;
    //     }
    //     while (norme > (tol * (size_s - rk)) && (rk < std::min(size_t / 2, size_s / 2))) {
    //         std::cout << "norme = " << norme << std::endl;
    //         auto w = generateGaussianVector(size_s);
    //         std::vector<CoefficientPrecision> lrtemp(size_t);
    //         bloc->add_vector_product('N', 1.0, w.data(), 1.0, lrtemp.data());
    //         std::vector<CoefficientPrecision> lr(size_t);
    //         temp.add_vector_product('N', 1.0, lrtemp.data(), 1.0, lr.data());
    //         norme = norm2(lr);
    //         lr    = mult(1.0 / norme, lr);
    //         std::vector<CoefficientPrecision> new_Ldata(size_t * rk);
    //         std::copy(L.data(), L.data() + (size_t * (rk - 1)), new_Ldata.data());
    //         L.assign(size_t, rk, new_Ldata.data(), false);
    //         L.set_col(rk - 1, lr);
    //         std::vector<CoefficientPrecision> new_Udata(size_s * rk);
    //         std::copy(Ut.data(), Ut.data() + (size_s * (rk - 1)), new_Udata.data());
    //         Ut.assign(size_s, rk, new_Udata.data(), false);
    //         std::vector<CoefficientPrecision> ur(size_s);
    //         bloc->add_vector_product('T', 1.0, lr.data(), 1.0, ur.data());
    //         Ut.set_col(rk - 1, ur);
    //         Matrix<CoefficientPrecision> ltemp(size_t, 1);
    //         ltemp.assign(size_t, 1, lr.data(), false);
    //         // temp = temp- l* l^T
    //         double alpha = -1.0;
    //         double beta  = 1.0;
    //         int kk       = 1;
    //         int ld       = size_t;
    //         std::cout << "aprés L et U : " << normFrob(bloc_dense - L * transp(Ut)) << std::endl;

    //         Blas<CoefficientPrecision>::gemm("N", "T", &ld, &ld, &kk, &alpha, ltemp.data(), &ld, ltemp.data(), &ld, &beta, temp.data(), &ld);
    //         rk += 1;
    //     }
    //     if (rk == std::min(size_t / 2, size_s / 2)) {
    //         flag_out = false;
    //     } else {
    //         flag_out = true;
    //         // Matrix<CoefficientPrecision> U(rk, size_s);
    //         // for (int k = 0; k < rk; ++k) {
    //         //     for (int l = 0; l < size_s; ++l) {
    //         //         U(k, l) = Ut(l, k);
    //         //     }
    //         // }
    //         auto U = transp(Ut);
    //         bloc->delete_children();
    //         LowRankMatrix<CoefficientPrecision, CoordinatePrecision> LU(L, U);
    //         bloc->set_low_rank_data(LU);
    //         // Ut = U;
    //     }
    // }
    // il faut renvoyer LU , on ravaill aavec UT pour toujours rajouter de colonnes
    void random_ACA(const int &size_t, const int &size_s, const int &oft, const int &ofs, const double &tolerance, bool &flagout) {
        int rk = 1;
        Matrix<CoefficientPrecision> L(size_t, 1);
        Matrix<CoefficientPrecision> Ut(size_s, 1);
        Matrix<CoefficientPrecision> temp(size_t, size_t);
        for (int k = 0; k < size_t; ++k) {
            temp(k, k) = 1.0;
        }
        double norme_LU = 0.0;
        auto bloc       = this->get_block(size_t, size_s, oft, ofs);
        auto w          = gaussian_vector(size_s, 0.0, 1.0);
        while ((rk < size_t / 2) && (rk < size_s / 2)) {
            std::vector<CoefficientPrecision> Aw(size_t, 0.0);
            bloc->add_vector_product('N', 1.0, w.data(), 1.0, Aw.data());
            std::vector<CoefficientPrecision> lr(size_t, 0.0);
            temp.add_vector_product('N', 1.0, Aw.data(), 1.0, lr.data());
            auto norm_lr = norm2(lr);
            if (norm_lr < 1e-20) {
                flagout = false;
                break;
            } else {
                lr = mult(1.0 / norm_lr, lr);
            }
            std::vector<CoefficientPrecision> ur(size_s, 0.0);
            bloc->add_vector_product('T', 1.0, lr.data(), 1.0, ur.data());
            auto norm_ur = norm2(ur);
            // std::cout << "norm2 ur : " << norm_ur << std::endl;
            Matrix<CoefficientPrecision> lr_ur(size_t, size_t);
            double alpha = 1.0;
            double beta  = -1.0;
            int kk       = 1;
            Blas<CoefficientPrecision>::gemm("N", "T", &size_t, &size_t, &kk, &alpha, lr.data(), &size_t, ur.data(), &size_t, &alpha, lr_ur.data(), &size_t);
            // Matrix<CoefficientPrecision> llr(size_t, 1);
            // llr.assign(size_t, 1, lr.data(), false);
            // Matrix<CoefficientPrecision> uur(size_s, 1);
            // uur.assign(size_t, 1, ur.data(), false);
            // Matrix<CoefficientPrecision> uurt(1, size_s);
            // Matrix<CoefficientPrecision> llrt(1, size_t);
            // transpose(uurt, uur);
            // lr_ur = llr * uurt;
            std::vector<CoefficientPrecision> Llr(L.nb_cols(), 0.0);
            L.add_vector_product('T', 1.0, lr.data(), 1.0, Llr.data());
            std::vector<CoefficientPrecision> ULlr(size_s, 0.0);
            Ut.add_vector_product('N', 1.0, Llr.data(), 1.0, ULlr.data());
            double trace = 0.0;
            for (int k = 0; k < std::min(size_t, size_s); ++k) {
                trace += (ULlr[k] * ur[k]);
            }
            auto nr_lrur = normFrob(lr_ur);
            auto nr      = norme_LU + std::pow(nr_lrur, 2.0) + 2 * trace;
            if (rk > 1 && (std::pow(norm_lr * norm_ur * (size_s - rk), 2.0) <= std::pow(tolerance, 2.0) * nr)) {
                break;
            }
            if (rk == 1) {
                std::copy(lr.data(), lr.data() + size_t, L.data());
                std::copy(ur.data(), ur.data() + size_s, Ut.data());
                // std::cout << "!!!!!! aprés affectation " << normFrob(L) << ',' << normFrob(Ut) << "! " << norm_lr << ',' << norm_ur << std::endl;
            } else {
                Matrix<CoefficientPrecision> new_L(size_t, rk);
                std::copy(L.data(), L.data() + L.nb_rows() * L.nb_cols(), new_L.data());
                std::copy(lr.data(), lr.data() + size_t, new_L.data() + (rk - 1) * size_t);
                Matrix<CoefficientPrecision> new_U(size_s, rk);
                std::copy(Ut.data(), Ut.data() + Ut.nb_rows() * Ut.nb_cols(), new_U.data());
                std::copy(ur.data(), ur.data() + size_s, new_U.data() + (rk - 1) * size_s);
                L  = new_L;
                Ut = new_U;
            }
            w = gaussian_vector(size_s, 0.0, 1.0);
            Blas<CoefficientPrecision>::gemm("N", "T", &size_t, &size_t, &kk, &beta, lr.data(), &size_t, lr.data(), &size_t, &alpha, temp.data(), &size_t);
            norme_LU += std::pow(nr_lrur, 2.0);
            rk += 1;
        }
        if (rk >= std::min(size_t / 2, size_s / 2)) {
            flagout = false;
        }
        if (flagout) {
            Matrix<CoefficientPrecision> U(rk - 1, size_s);
            transpose(Ut, U);
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_mat(L, U);
            bloc->delete_children();
            bloc->set_low_rank_data(lr_mat);
        } else {
            std::cerr << "aucune approximation trouvée , rk=" << rk << "sur un bloxc de taille " << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << std::endl;
        }
        // std::cout << flagout << std::endl;
    }

    std::vector<Matrix<CoefficientPrecision>> random_ACA_vect(const int &size_t, const int &size_s, const int &oft, const int &ofs, const double &tolerance, bool &flagout) {
        int rk = 1;
        Matrix<CoefficientPrecision> L(size_t, 1);
        Matrix<CoefficientPrecision> Ut(size_s, 1);
        Matrix<CoefficientPrecision> temp(size_t, size_t);
        std::vector<Matrix<CoefficientPrecision>> res;
        for (int k = 0; k < size_t; ++k) {
            temp(k, k) = 1.0;
        }
        double norme_LU = 0.0;
        auto bloc       = this->get_block(size_t, size_s, oft, ofs);
        auto w          = gaussian_vector(size_s, 0.0, 1.0);
        while ((rk < size_t / 2) && (rk < size_s / 2)) {
            std::vector<CoefficientPrecision> Aw(size_t, 0.0);
            bloc->add_vector_product('N', 1.0, w.data(), 1.0, Aw.data());
            std::vector<CoefficientPrecision> lr(size_t, 0.0);
            temp.add_vector_product('N', 1.0, Aw.data(), 1.0, lr.data());
            auto norm_lr = norm2(lr);
            if (norm_lr < 1e-20) {
                // flagout = false;
                // matrice vide
                break;
            } else {
                lr = mult(1.0 / norm_lr, lr);
            }
            std::vector<CoefficientPrecision> ur(size_s, 0.0);
            bloc->add_vector_product('T', 1.0, lr.data(), 1.0, ur.data());
            auto norm_ur = norm2(ur);
            // std::cout << "norm2 ur : " << norm_ur << std::endl;
            Matrix<CoefficientPrecision> lr_ur(size_t, size_t);
            double alpha = 1.0;
            double beta  = -1.0;
            int kk       = 1;
            Blas<CoefficientPrecision>::gemm("N", "T", &size_t, &size_t, &kk, &alpha, lr.data(), &size_t, ur.data(), &size_t, &alpha, lr_ur.data(), &size_t);
            // Matrix<CoefficientPrecision> llr(size_t, 1);
            // llr.assign(size_t, 1, lr.data(), false);
            // Matrix<CoefficientPrecision> uur(size_s, 1);
            // uur.assign(size_t, 1, ur.data(), false);
            // Matrix<CoefficientPrecision> uurt(1, size_s);
            // Matrix<CoefficientPrecision> llrt(1, size_t);
            // transpose(uurt, uur);
            // lr_ur = llr * uurt;
            std::vector<CoefficientPrecision> Llr(L.nb_cols(), 0.0);
            L.add_vector_product('T', 1.0, lr.data(), 1.0, Llr.data());
            std::vector<CoefficientPrecision> ULlr(size_s, 0.0);
            Ut.add_vector_product('N', 1.0, Llr.data(), 1.0, ULlr.data());
            double trace = 0.0;
            for (int k = 0; k < std::min(size_t, size_s); ++k) {
                trace += (ULlr[k] * ur[k]);
            }
            auto nr_lrur = normFrob(lr_ur);
            auto nr      = norme_LU + std::pow(nr_lrur, 2.0) + 2 * trace;
            if (rk > 1 && (std::pow(norm_lr * norm_ur * (size_s - rk), 2.0) <= std::pow(tolerance, 2.0) * nr)) {
                break;
            }
            if (rk == 1) {
                std::copy(lr.data(), lr.data() + size_t, L.data());
                std::copy(ur.data(), ur.data() + size_s, Ut.data());
                // std::cout << "!!!!!! aprés affectation " << normFrob(L) << ',' << normFrob(Ut) << "! " << norm_lr << ',' << norm_ur << std::endl;
            } else {
                Matrix<CoefficientPrecision> new_L(size_t, rk);
                std::copy(L.data(), L.data() + L.nb_rows() * L.nb_cols(), new_L.data());
                std::copy(lr.data(), lr.data() + size_t, new_L.data() + (rk - 1) * size_t);
                Matrix<CoefficientPrecision> new_U(size_s, rk);
                std::copy(Ut.data(), Ut.data() + Ut.nb_rows() * Ut.nb_cols(), new_U.data());
                std::copy(ur.data(), ur.data() + size_s, new_U.data() + (rk - 1) * size_s);
                L  = new_L;
                Ut = new_U;
            }
            w = gaussian_vector(size_s, 0.0, 1.0);
            Blas<CoefficientPrecision>::gemm("N", "T", &size_t, &size_t, &kk, &beta, lr.data(), &size_t, lr.data(), &size_t, &alpha, temp.data(), &size_t);
            norme_LU += std::pow(nr_lrur, 2.0);
            rk += 1;
        }
        if (rk >= std::min(size_t / 2, size_s / 2)) {
            flagout = false;
        }
        if (flagout) {
            if (rk == 1) {
                Matrix<CoefficientPrecision> U(1, size_s);
                res.push_back(L);
                res.push_back(U);
            } else {
                Matrix<CoefficientPrecision> U(rk - 1, size_s);
                transpose(Ut, U);
                // LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_mat(L, U);
                // bloc->delete_children();
                // bloc->set_low_rank_data(lr_mat);
                res.push_back(L);
                res.push_back(U);
            }
        } else {
            Matrix<CoefficientPrecision> dense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
            copy_to_dense(*this, dense.data());
            Matrix<CoefficientPrecision> U(rk - 1, size_s);
            transpose(Ut, U);

            std::cerr << "aucune approximation trouvée , rk=" << rk << "sur un bloc de taille " << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << " et de norme : " << normFrob(dense) << std::endl;
            std::cerr << " l'erreur a la fin est " << normFrob(dense - L * U) << std::endl;
        }
        return res;
        // std::cout << flagout << std::endl;
    }

    // void iterate_ACA(const double &tolerance) {
    //     auto adm = this->compute_admissibility(this->get_target_cluster(), this->get_source_cluster());
    //     if (adm) {
    //         bool flag = true;
    //         Matrix<CoefficientPrecision> dense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
    //         copy_to_dense(*this, dense.data());
    //         this->random_ACA(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset(), tolerance, flag);
    //         // std::cout << "flag : " << flag;
    //         if (flag) {

    //             std::cout << "norme du_ bloc : " << normFrob(dense) << std::endl;
    //             std::cout << "erreur de l'approximation sur le bloc : " << normFrob(this->get_low_rank_data()->get_U() * this->get_low_rank_data()->get_V() - dense) << std::endl;
    //         }
    //         if (!flag && this->get_children().size() > 0) {
    //             for (auto &child : this->get_children()) {
    //                 child->iterate_ACA(tolerance);
    //             }
    //         }
    //     } else if (this->get_children().size() > 0) {
    //         for (auto &child : this->get_children()) {
    //             child->iterate_ACA(tolerance);
    //         }
    //     }
    // }

    void iterate_ACA(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, const double &tolerance) {
        auto adm = this->compute_admissibility(t, s);
        if (adm) {
            bool flag = true;
            // Matrix<CoefficientPrecision> dense(t.get_size(), s.get_size());
            // copy_to_dense(*this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset()), dense.data());
            this->random_ACA(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), tolerance, flag);
            // std::cout << "flag : " << flag;
            // if (flag) {

            //     std::cout << "norme du_ bloc : " << normFrob(dense) << std::endl;
            //     std::cout << "erreur de l'approximation sur le bloc : " << normFrob(this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_low_rank_data()->get_U() * this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_low_rank_data()->get_V() - dense) << std::endl;
            // }
            if (!flag && this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_children().size() > 0) {
                for (auto &t_son : t.get_children()) {
                    for (auto &s_son : s.get_children()) {
                        this->iterate_ACA(*t_son, *s_son, tolerance);
                    }
                }
            }
        } else if (this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_children().size() > 0) {
            for (auto &t_son : t.get_children()) {
                for (auto &s_son : s.get_children()) {
                    this->iterate_ACA(*t_son, *s_son, tolerance);
                }
            }
        }
    }

    void format_ACA(HMatrix *res, const double &tolerance) {
        auto &t    = this->get_target_cluster();
        auto &s    = this->get_source_cluster();
        int size_t = t.get_size();
        int size_s = s.get_size();
        int oft    = t.get_offset();
        int ofs    = s.get_offset();
        auto adm   = this->compute_admissibility(this->get_target_cluster(), this->get_source_cluster());
        if (adm) {
            bool flag                                        = true;
            std::vector<Matrix<CoefficientPrecision>> resACA = this->random_ACA_vect(size_t, size_s, oft, ofs, tolerance, flag);
            if (resACA.size() > 0) {
                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(resACA[0], resACA[1]);
                res->set_low_rank_data(lr);
            } else {
                if (this->get_children().size() > 0) {
                    for (auto &child : this->get_children()) {
                        auto sub_res = res->add_child(&child->get_target_cluster(), &child->get_source_cluster());
                        child->format_ACA(sub_res, tolerance);
                    }
                } else {
                    auto dense = this->get_dense_data();
                    // std::unique_ptr<Matrix<CoefficientPrecision>> ptr_dense(dense);
                    res->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(*dense));
                }
            }
        } else {
            if (this->get_children().size() > 0) {
                for (auto &child : this->get_children()) {
                    auto sub_res = res->add_child(&child->get_target_cluster(), &child->get_source_cluster());
                    child->format_ACA(sub_res, tolerance);
                }
            } else {
                auto dense = this->get_dense_data();
                res->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(*dense));
            }
        }
    }

    // Linear algebra
    void
    add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const;
    void add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const;
    void plus_egal(const CoefficientPrecision &apha, const HMatrix *B);

    // In case a dense matrix is assigned when n > minblocksize
    void assign(const Matrix<CoefficientPrecision> *M, const int &oft0, const int &ofs0) {
        auto adm = this->compute_admissibility(this->get_target_cluster(), this->get_source_cluster());
        if (adm) {
            // std::cout << "!!!!" << std::endl;
            SumExpression_fast<CoefficientPrecision, CoordinatePrecision> temp;
            temp.set_size(this->get_target_cluster(), this->get_source_cluster());
            auto m = copy_sub_matrix(*M, this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset() - oft0, this->get_source_cluster().get_offset() - ofs0);
            temp.set_dense(m);
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_test(temp, *this->get_low_rank_generator(), this->get_target_cluster(), this->get_source_cluster(), -1, this->get_epsilon());
            if (lr_test.get_U().nb_cols() > 0) {
                this->set_low_rank_data(lr_test);
            } else {
                if (this->get_target_cluster().get_children().size() > 0) {
                    for (auto &t_child : this->get_target_cluster().get_children()) {
                        for (auto &s_child : this->get_source_cluster().get_children()) {
                            auto child = this->add_child(t_child.get(), s_child.get());
                            child->assign(M, oft0, ofs0);
                        }
                    }
                } else {
                    // auto m                                            = copy_sub_matrix(*M, this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset() - oft0, this->get_source_cluster().get_offset() - ofs0);
                    // std::unique_ptr<Matrix<CoefficientPrecision>> mat = std::make_unique<Matrix<CoefficientPrecision>>(m);
                    // this->set_dense_data(std::move(mat));
                    this->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(m));
                }
            }
        } else {
            if (this->get_target_cluster().get_children().size() > 0) {
                for (auto &t_child : this->get_target_cluster().get_children()) {
                    for (auto &s_child : this->get_source_cluster().get_children()) {
                        auto child = this->add_child(t_child.get(), s_child.get());
                        child->assign(M, oft0, ofs0);
                    }
                }
            } else {
                auto m = copy_sub_matrix(*M, this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset() - oft0, this->get_source_cluster().get_offset() - ofs0);
                // std::unique_ptr<Matrix<CoefficientPrecision>> mat = std::make_unique<Matrix<CoefficientPrecision>>(m);
                // this->set_dense_data(std::move(mat));
                this->set_dense_data(std::move(std::make_unique<Matrix<CoefficientPrecision>>(m)));

                // std::cout << " on affect un bloc de taille " << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << std::endl;
            }
        }
    }

    void copy_format(HMatrix *res) {
        if (this->get_children().size() > 0) {
            for (auto &t_child : this->get_target_cluster().get_children()) {
                for (auto &s_child : this->get_source_cluster().get_children()) {
                    auto new_child = res->add_child(t_child.get(), s_child.get());
                    this->get_block(t_child->get_size(), s_child->get_size(), t_child->get_offset(), s_child->get_offset())->copy_format(new_child);
                }
            }
        } else {
            if (this->is_dense()) {
                if (this->get_target_cluster().get_children().size() == 0) {
                    // std::unique_ptr<Matrix<CoefficientPrecision>> m = std::make_unique<Matrix<CoefficientPrecision>>(*this->get_dense_data());
                    res->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(*this->get_dense_data()));
                } else {
                    // auto m = *M->get_dense_data();
                    res->assign(this->get_dense_data(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                    // res->set_data(*this->get_dense_data());
                }
            } else {
                res->set_low_rank_data(*this->get_low_rank_data());
            }
        }
    }

    HMatrix get_format() {
        HMatrix res(this->get_target_cluster(), this->get_source_cluster());
        res.set_admissibility_condition(this->get_admissibility_condition());
        res.set_low_rank_generator(this->get_low_rank_generator());
        res.set_epsilon(this->get_epsilon());
        res.set_eta(this->m_tree_data->m_eta);
        this->copy_format(&res);
        return res;
    }
    HMatrix hmatrix_product_fast(const HMatrix &H) const;
    HMatrix hmatrix_product_fast_triangulaire(char transa, char transb, const HMatrix &H) const;

    void add_hmatrix_product_fast(const CoefficientPrecision &alpha, SumExpression_fast<CoefficientPrecision, CoordinatePrecision> &sum_expr, std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> *adm);

    ///////////////////////////////////////////////////
    // Trouver x tel que Lx=y

    void hmatrix_vector_triangular_L(std::vector<CoefficientPrecision> &X, std::vector<CoefficientPrecision> &B, const Cluster<CoordinatePrecision> &t, const int &offset0) {
        auto Ltt = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Ltt->get_children().size() == 0) {
            if (Ltt->is_dense()) {
                auto ldense = *Ltt->get_dense_data();
                std::vector<CoefficientPrecision> target(t.get_size());
                std::copy(B.begin() + t.get_offset() - offset0, B.begin() + t.get_offset() - offset0 + t.get_size(), target.begin());
                triangular_matrix_vec_solve('L', 'L', 'N', 'U', 1.0, ldense, target);
                std::copy(target.begin(), target.end(), X.begin() + t.get_offset() - offset0);
            } else {
                std::cerr << "bloc no dense sur la diagonale" << std::endl;
            }
        } else {
            for (int j = 0; j < t.get_children().size(); ++j) {
                auto &tj = t.get_children()[j];
                Ltt->hmatrix_vector_triangular_L(X, B, *tj, offset0);
                for (int i = j + 1; i < t.get_children().size(); ++i) {
                    auto &ti = t.get_children()[i];
                    std::vector<CoefficientPrecision> bi(ti->get_size());
                    std::copy(B.begin() + ti->get_offset() - offset0, B.begin() + ti->get_offset() - offset0 + ti->get_size(), bi.begin());
                    std::vector<CoefficientPrecision> xj(tj->get_size());
                    std::copy(X.begin() + tj->get_offset() - offset0, X.begin() + tj->get_offset() - offset0 + tj->get_size(), xj.begin());
                    auto Ltemp = Ltt->get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset());

                    Ltemp->get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset())->add_vector_product('N', -1.0, xj.data(), 1.0, bi.data());
                    std::copy(bi.begin(), bi.end(), B.begin() + ti->get_offset() - offset0);
                }
            }
        }
    }

    // Trouver x tel que xU=y ou Ux =y
    void hmatrix_vector_triangular_U(char trans, std::vector<CoefficientPrecision> &X, std::vector<CoefficientPrecision> &B, const Cluster<CoordinatePrecision> &t, const int &offset0) {
        auto Utt = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());

        if (Utt->get_children().size() == 0) {
            if (Utt->is_dense()) {
                auto udense = Utt->get_dense_data();
                std::vector<CoefficientPrecision> target(t.get_size());
                std::copy(B.begin() + t.get_offset() - offset0, B.begin() + t.get_offset() + t.get_size() - offset0, target.begin());
                if (trans == 'N') {
                    triangular_matrix_vec_solve('L', 'U', 'N', 'N', 1.0, *udense, target);
                } else {
                    triangular_matrix_vec_solve('L', 'U', 'T', 'N', 1.0, *udense, target);
                }
                std::copy(target.begin(), target.end(), X.begin() + t.get_offset() - offset0);
            }
        }

        else {
            if (trans == 'N') {
                for (int j = t.get_children().size() - 1; j > -1; --j) {
                    auto &tj = t.get_children()[j];
                    Utt->hmatrix_vector_triangular_U(trans, X, B, *tj, offset0);
                    for (int i = 0; i < j; ++i) {
                        auto &ti = t.get_children()[i];
                        std::vector<CoefficientPrecision> bi(ti->get_size());
                        std::vector<CoefficientPrecision> xj(tj->get_size());
                        std::copy(B.begin() + ti->get_offset() - offset0, B.begin() + ti->get_offset() - offset0 + ti->get_size(), bi.begin());
                        std::copy(X.begin() + tj->get_offset() - offset0, X.begin() + tj->get_offset() - offset0 + tj->get_size(), xj.begin());
                        Utt->get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset())->add_vector_product('N', -1.0, xj.data(), 1.0, bi.data());
                        std::copy(bi.begin(), bi.end(), B.begin() + ti->get_offset() - offset0);
                    }
                }
            } else {
                for (int j = 0; j < t.get_children().size(); ++j) {
                    auto &tj = t.get_children()[j];
                    Utt->hmatrix_vector_triangular_U(trans, X, B, *tj, offset0);
                    for (int i = j + 1; i < t.get_children().size(); ++i) {
                        auto &ti = t.get_children()[i];
                        std::vector<CoefficientPrecision> bi(ti->get_size());
                        std::copy(B.begin() + ti->get_offset() - offset0, B.begin() + ti->get_offset() - offset0 + ti->get_size(), bi.begin());
                        std::vector<CoefficientPrecision> xj(tj->get_size());
                        std::copy(X.begin() + tj->get_offset() - offset0, X.begin() + tj->get_offset() - offset0 + tj->get_size(), xj.begin());
                        Utt->get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset())->add_vector_product('T', -1.0, xj.data(), 1.0, bi.data());
                        std::copy(bi.begin(), bi.end(), B.begin() + ti->get_offset() - offset0);
                    }
                }
            }
        }
    }

    /// trouver X tq LX=Y   avec X et Y matrices hiérarchiques

    void FM_fast_build(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix *X, HMatrix *bloc_courant) {
        if (bloc_courant->is_dense()) {
            if (bloc_courant->get_target_cluster().get_children().size() > 0) {
                Matrix<CoefficientPrecision> bloc_dense(bloc_courant->get_target_cluster().get_size(), bloc_courant->get_source_cluster().get_size());
                copy_to_dense(*bloc_courant, bloc_dense.data());
                Matrix<CoefficientPrecision> ldense(this->get_target_cluster().get_size(), this->get_target_cluster().get_size());
                copy_to_dense(*this, ldense.data());
                triangular_matrix_matrix_solve('L', 'L', 'N', 'U', 1.0, ldense, bloc_dense);
                X->assign(&bloc_dense, this->get_target_cluster().get_offset(), bloc_courant->get_source_cluster().get_offset());
            } else {
                Matrix<CoefficientPrecision> dense  = *bloc_courant->get_dense_data();
                auto dense0                         = dense;
                auto Lt                             = this->get_block(bloc_courant->get_target_cluster().get_size(), bloc_courant->get_target_cluster().get_size(), bloc_courant->get_target_cluster().get_offset(), bloc_courant->get_target_cluster().get_offset());
                Matrix<CoefficientPrecision> ldense = *Lt->get_dense_data();
                triangular_matrix_matrix_solve('L', 'L', 'N', 'U', 1.0, ldense, dense);
                // std::unique_ptr<Matrix<CoefficientPrecision>> mat = std::make_unique<Matrix<CoefficientPrecision>>(dense);
                X->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(dense));
            }
        } else if (bloc_courant->is_low_rank()) {
            Matrix<CoefficientPrecision> new_U(this->get_source_cluster().get_size(), bloc_courant->get_low_rank_data()->get_U().nb_cols());
            if (this->get_target_cluster().get_children().size() == 0) {
                Matrix<CoefficientPrecision> ldense0(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
                copy_to_dense(*this, ldense0.data());
                new_U = bloc_courant->get_low_rank_data()->get_U();
                triangular_matrix_matrix_solve('L', 'L', 'N', 'U', 1.0, ldense0, new_U);
            } else {
                for (int k = 0; k < bloc_courant->get_low_rank_data()->get_U().nb_cols(); ++k) {
                    std::vector<CoefficientPrecision> col(new_U.nb_rows(), 0.0);
                    auto v = bloc_courant->get_low_rank_data()->get_U().get_col(k);
                    this->hmatrix_vector_triangular_L(col, v, this->get_target_cluster(), this->get_target_cluster().get_offset());
                    new_U.set_col(k, col);
                }
            }
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_new(new_U, bloc_courant->get_low_rank_data()->get_V());
            X->set_low_rank_data(lr_new);
        } else {
            for (auto &ti : bloc_courant->get_target_cluster().get_children()) {
                for (auto &cluster_child : bloc_courant->get_source_cluster().get_children()) {
                    auto X_child = X->add_child(ti.get(), cluster_child.get());
                    this->get_block(ti->get_size(), ti->get_size(), ti->get_offset(), ti->get_offset())->FM_fast_build(*ti, *cluster_child, X_child, bloc_courant->get_block(ti->get_size(), cluster_child->get_size(), ti->get_offset(), cluster_child->get_offset()));
                    for (auto &tj : bloc_courant->get_target_cluster().get_children()) {
                        if (tj->get_offset() > ti->get_offset()) {
                            auto Lji  = this->get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
                            auto prod = Lji->hmatrix_product_fast(*X_child);
                            auto Zt   = bloc_courant->get_block(tj->get_size(), cluster_child->get_size(), tj->get_offset(), cluster_child->get_offset());
                            Zt->plus_egal(-1.0, &prod);
                        }
                    }
                }
            }
        }
    }

    // Trouver X tel que XU = Y avec X et Y matrices hiérarchiques
    void FMT_fast_build(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix *X, HMatrix *bloc_courant) {
        if (bloc_courant->is_dense()) {

            if (bloc_courant->get_target_cluster().get_children().size() > 0) {
                Matrix<CoefficientPrecision> bloc_dense(bloc_courant->get_target_cluster().get_size(), bloc_courant->get_source_cluster().get_size());
                copy_to_dense(*bloc_courant, bloc_dense.data());
                Matrix<CoefficientPrecision> ldense(this->get_target_cluster().get_size(), this->get_target_cluster().get_size());
                copy_to_dense(*this, ldense.data());
                triangular_matrix_matrix_solve('R', 'U', 'N', 'N', 1.0, ldense, bloc_dense);
                X->assign(&bloc_dense, bloc_courant->get_target_cluster().get_offset(), bloc_courant->get_source_cluster().get_offset());
            } else {
                Matrix<CoefficientPrecision> dense  = *bloc_courant->get_dense_data();
                auto dense0                         = dense;
                auto Ut                             = this->get_block(bloc_courant->get_source_cluster().get_size(), bloc_courant->get_source_cluster().get_size(), bloc_courant->get_source_cluster().get_offset(), bloc_courant->get_source_cluster().get_offset());
                Matrix<CoefficientPrecision> ldense = *Ut->get_dense_data();
                triangular_matrix_matrix_solve('R', 'U', 'N', 'N', 1.0, ldense, dense);
                // std::unique_ptr<Matrix<CoefficientPrecision>> mat = std::make_unique<Matrix<CoefficientPrecision>>(dense);
                X->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(dense));
            }
        } else if (bloc_courant->is_low_rank()) {
            Matrix<CoefficientPrecision> new_V(bloc_courant->get_low_rank_data()->get_V().nb_rows(), this->get_source_cluster().get_size());
            if (this->get_target_cluster().get_children().size() == 0) {
                Matrix<CoefficientPrecision> ldense0(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
                copy_to_dense(*this, ldense0.data());
                new_V = bloc_courant->get_low_rank_data()->get_V();
                triangular_matrix_matrix_solve('R', 'U', 'N', 'N', 1.0, ldense0, new_V);
            } else {
                for (int k = 0; k < bloc_courant->get_low_rank_data()->get_V().nb_rows(); ++k) {
                    std::vector<CoefficientPrecision> row(new_V.nb_cols(), 0.0);
                    auto v = bloc_courant->get_low_rank_data()->get_V().get_row(k);
                    this->hmatrix_vector_triangular_U('T', row, v, this->get_target_cluster(), this->get_target_cluster().get_offset());
                    new_V.set_row(k, row);
                }
            }
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_new(bloc_courant->get_low_rank_data()->get_U(), new_V);
            X->set_low_rank_data(lr_new);

        } else {
            for (auto &ti : bloc_courant->get_source_cluster().get_children()) {
                for (auto &cluster_child : bloc_courant->get_target_cluster().get_children()) {
                    auto X_child = X->add_child(cluster_child.get(), ti.get());
                    this->get_block(ti->get_size(), ti->get_size(), ti->get_offset(), ti->get_offset())->FMT_fast_build(*ti, *cluster_child, X_child, bloc_courant->get_block(cluster_child->get_size(), ti->get_size(), cluster_child->get_offset(), ti->get_offset()));
                    for (auto &tj : bloc_courant->get_source_cluster().get_children()) {
                        if (tj->get_offset() > ti->get_offset()) {
                            auto Uij  = this->get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset());
                            auto prod = X_child->hmatrix_product_fast(*Uij);
                            auto Zt   = bloc_courant->get_block(cluster_child->get_size(), tj->get_size(), cluster_child->get_offset(), tj->get_offset());
                            Zt->plus_egal(-1.0, &prod);
                        }
                    }
                }
            }
        }
    }

    // Trouver L et U tel que LU = M  , L et U hiérarchiques
    // (   L11     |   0    )   (   U11    |    U12 )   =   (M11    |   M12 )
    // (   L21     [   L22  )   (   0      |    U22 )       (M21    |   M22 )
    // HLU(M11)=L11 U11
    // FM(U11, M21) = L21
    // FMT(L11, M12) = U21
    // HLU( M22 - L21 U12) = L22 U22

    friend void HLU_fast(HMatrix &H, const Cluster<CoordinatePrecision> &t, HMatrix *L, HMatrix *U) {
        auto Htt = H.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        // Si on est sur un feuille on fait le vrai LU
        if (Htt->is_dense()) {
            Matrix<CoefficientPrecision> Hdense = *Htt->get_dense_data();
            auto hdense                         = Hdense;
            std::vector<int> pivot(t.get_size(), 0.0);
            Matrix<CoefficientPrecision> l(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> u(t.get_size(), t.get_size());
            get_lu_factorisation(hdense, l, u, pivot);
            if (Htt->get_target_cluster().get_children().size() == 0) {
                L->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(l));
                U->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(u));
            } else {
                L->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->assign(&l, t.get_offset(), t.get_offset());
                U->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->assign(&u, t.get_offset(), t.get_offset());
            }
        } else {
            // On est sur un bloc diagonal qui a des fils, on peut descendre
            auto &t_children = t.get_children();
            for (int i = 0; i < t_children.size(); ++i) {
                auto child_L = L->add_child(t_children[i].get(), t_children[i].get());
                auto child_U = U->add_child(t_children[i].get(), t_children[i].get());
                //  appel recursif
                HLU_fast(H, *t_children[i], child_L, child_U);

                for (int j = i + 1; j < t_children.size(); ++j) {
                    //// On apopelle forward sur la matrice H permutée par la perm donné par l'appele de LU(ti)
                    // Trouver U12
                    auto Uij = U->add_child(t_children[i].get(), t_children[j].get());
                    auto Hij = H.get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset());
                    child_L->FM_fast_build(*t_children[i], *t_children[j], Uij, Hij);
                    // trouver L21
                    auto Hji = H.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset());
                    auto Lji = L->add_child(t_children[j].get(), t_children[i].get());
                    child_U->FMT_fast_build(*t_children[j], *t_children[i], Lji, Hji);

                    for (int r = i + 1; r < t_children.size(); ++r) {
                        auto Htemp = H.get_block(t_children[j]->get_size(), t_children[r]->get_size(), t_children[j]->get_offset(), t_children[r]->get_offset());
                        auto Ltemp = L->get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset());
                        auto Utemp = U->get_block(t_children[i]->get_size(), t_children[r]->get_size(), t_children[i]->get_offset(), t_children[r]->get_offset());
                        // M22 = M22 - L21 U12
                        auto prod = Ltemp->hmatrix_product_fast(*Utemp);
                        Htemp->plus_egal(-1.0, &prod);
                    }
                }
            }
        }
    }

    // Trouver x tel que LUx= y
    std::vector<CoefficientPrecision> solve_LU_triangular(HMatrix &L, HMatrix &U, std::vector<CoefficientPrecision> &z) {
        auto &t = L.get_target_cluster();
        std::vector<CoefficientPrecision> temp(t.get_size());
        L.hmatrix_vector_triangular_L(temp, z, t, 0);
        std::vector<CoefficientPrecision> res(t.get_size());
        U.hmatrix_vector_triangular_U('N', res, temp, t, 0);
        return res;
    }

    // get bloc fils
    HMatrix<CoefficientPrecision, CoordinatePrecision> *get_block(const int &szt, const int &szs, const int &oft, const int &ofs) const {
        if ((this->get_target_cluster().get_size() == szt) && (this->get_target_cluster().get_offset() == oft) && (this->get_source_cluster().get_size() == szs) && (this->get_source_cluster().get_offset() == ofs)) {
            return const_cast<HMatrix<CoefficientPrecision, CoordinatePrecision> *>(this);
        } else {
            if (this->get_children().size() == 0) {
                // "je voit pas d'autre solution , on renvoie le plus gros blocs et il faudra extraire"
                return const_cast<HMatrix<CoefficientPrecision, CoordinatePrecision> *>(this);
            }
            for (auto &child : this->get_children()) {
                auto &target_cluster = child->get_target_cluster();
                auto &source_cluster = child->get_source_cluster();
                if (((szt + (oft - child->get_target_cluster().get_offset()) <= child->get_target_cluster().get_size()) and (szt <= child->get_target_cluster().get_size()) and (oft >= child->get_target_cluster().get_offset()))
                    and (((szs + (ofs - child->get_source_cluster().get_offset()) <= child->get_source_cluster().get_size()) and (szs <= child->get_source_cluster().get_size()) and (ofs >= child->get_source_cluster().get_offset())))) {
                    return child->get_block(szt, szs, oft, ofs);
                }
            }
        }
        return nullptr;
    }

    // get compression : ratio low rank / dense
    double get_compression() {
        double compr = 0.0;
        for (auto &l : this->get_leaves()) {
            if (l->is_dense()) {
                compr += l->get_target_cluster().get_size() * l->get_source_cluster().get_size();
            } else {
                compr += l->get_low_rank_data()->get_U().nb_cols() * (l->get_low_rank_data()->get_U().nb_rows() + l->get_low_rank_data()->get_V().nb_cols());
            }
        }
        compr = compr / (1.0 * this->get_target_cluster().get_size() * this->get_source_cluster().get_size());
        return (1 - compr);
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const {
    switch (m_storage_type) {
    case StorageType::Dense:
        if (m_symmetry == 'N') {
            m_dense_data->add_vector_product(trans, alpha, in, beta, out);
        } else {
            m_dense_data->add_vector_product_symmetric(trans, alpha, in, beta, out, m_UPLO, m_symmetry);
        }
        break;
    case StorageType::LowRank:
        m_low_rank_data->add_vector_product(trans, alpha, in, beta, out);
        break;
    default:
        threaded_hierarchical_add_vector_product(trans, alpha, in, beta, out);
        break;
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const {
    switch (m_storage_type) {
    case StorageType::Dense:
        if (m_symmetry == 'N') {
            m_dense_data->add_matrix_product_row_major(trans, alpha, in, beta, out, mu);
        } else {
            m_dense_data->add_matrix_product_symmetric_row_major(trans, alpha, in, beta, out, mu, m_UPLO, m_symmetry);
        }
        break;
    case StorageType::LowRank:
        m_low_rank_data->add_matrix_product_row_major(trans, alpha, in, beta, out, mu);
        break;
    default:
        threaded_hierarchical_add_matrix_product_row_major(trans, alpha, in, beta, out, mu);
        break;
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::threaded_hierarchical_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const {

    set_leaves_in_cache();

    // int rankWorld;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    int out_size(m_target_cluster->get_size());
    auto get_output_cluster{&HMatrix::get_target_cluster};
    auto get_input_cluster{&HMatrix::get_source_cluster};
    int local_input_offset  = m_source_cluster->get_offset();
    int local_output_offset = m_target_cluster->get_offset();
    char trans_sym          = (m_symmetry_type_for_leaves == 'S') ? 'T' : 'C';

    if (trans != 'N') {
        out_size            = m_source_cluster->get_size();
        get_input_cluster   = &HMatrix::get_target_cluster;
        get_output_cluster  = &HMatrix::get_source_cluster;
        local_input_offset  = m_target_cluster->get_offset();
        local_output_offset = m_source_cluster->get_offset();
        trans_sym           = 'N';
    }

    int incx(1), incy(1);
    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        // TODO: use blas
        std::transform(out, out + out_size, out, [&beta](CoefficientPrecision &c) { return c * beta; });
    }

// Contribution champ lointain
#if defined(_OPENMP)
#    pragma omp parallel
#endif
    {
        std::vector<CoefficientPrecision> temp(out_size, 0);
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
        for (int b = 0; b < m_leaves.size(); b++) {
            int input_offset  = (m_leaves[b]->*get_input_cluster)().get_offset();
            int output_offset = (m_leaves[b]->*get_output_cluster)().get_offset();
            m_leaves[b]->add_vector_product(trans, 1, in + input_offset - local_input_offset, 1, temp.data() + (output_offset - local_output_offset));
        }

        // Symmetry part of the diagonal part
        if (m_symmetry_type_for_leaves != 'N') {
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
            for (int b = 0; b < m_leaves_for_symmetry.size(); b++) {
                int input_offset  = (m_leaves_for_symmetry[b]->*get_input_cluster)().get_offset();
                int output_offset = (m_leaves_for_symmetry[b]->*get_output_cluster)().get_offset();
                m_leaves_for_symmetry[b]->add_vector_product(trans_sym, 1, in + output_offset - local_input_offset, 1, temp.data() + (input_offset - local_output_offset));
            }
        }

#if defined(_OPENMP)
#    pragma omp critical
#endif
        Blas<CoefficientPrecision>::axpy(&out_size, &alpha, temp.data(), &incx, out, &incy);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::threaded_hierarchical_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const {

    set_leaves_in_cache();

    if ((trans == 'T' && m_symmetry_type_for_leaves == 'H')
        || (trans == 'C' && m_symmetry_type_for_leaves == 'S')) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not supported (" + std::string(1, trans) + " with " + m_symmetry_type_for_leaves + ")"); // LCOV_EXCL_LINE
        // throw std::invalid_argument("[Htool error] Operation is not supported (" + std::string(1, trans) + " with " + m_symmetry_type_for_leaves + ")");                  // LCOV_EXCL_LINE
    }

    int out_size(m_target_cluster->get_size() * mu);
    auto get_output_cluster{&HMatrix::get_target_cluster};
    auto get_input_cluster{&HMatrix::get_source_cluster};
    int local_output_offset = m_target_cluster->get_offset();
    int local_input_offset  = m_source_cluster->get_offset();
    char trans_sym          = (m_symmetry_type_for_leaves == 'S') ? 'T' : 'C';

    if (trans != 'N') {
        out_size            = m_source_cluster->get_size() * mu;
        get_input_cluster   = &HMatrix::get_target_cluster;
        get_output_cluster  = &HMatrix::get_source_cluster;
        local_input_offset  = m_target_cluster->get_offset();
        local_output_offset = m_source_cluster->get_offset();
        trans_sym           = 'N';
    }

    int incx(1), incy(1);
    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        // TODO: use blas
        std::transform(out, out + out_size, out, [&beta](CoefficientPrecision &c) { return c * beta; });
    }

// Contribution champ lointain
#if defined(_OPENMP)
#    pragma omp parallel
#endif
    {
        std::vector<CoefficientPrecision> temp(out_size, 0);
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
        for (int b = 0; b < m_leaves.size(); b++) {
            int input_offset  = (m_leaves[b]->*get_input_cluster)().get_offset();
            int output_offset = (m_leaves[b]->*get_output_cluster)().get_offset();
            m_leaves[b]->add_matrix_product_row_major(trans, 1, in + (input_offset - local_input_offset) * mu, 1, temp.data() + (output_offset - local_output_offset) * mu, mu);
        }

        // Symmetry part of the diagonal part
        if (m_symmetry_type_for_leaves != 'N') {
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
            for (int b = 0; b < m_leaves_for_symmetry.size(); b++) {
                int input_offset  = (m_leaves_for_symmetry[b]->*get_input_cluster)().get_offset();
                int output_offset = (m_leaves_for_symmetry[b]->*get_output_cluster)().get_offset();
                m_leaves_for_symmetry[b]->add_matrix_product_row_major(trans_sym, 1, in + (output_offset - local_input_offset) * mu, 1, temp.data() + (input_offset - local_output_offset) * mu, mu);
            }
        }

#if defined(_OPENMP)
#    pragma omp critical
#endif
        Blas<CoefficientPrecision>::axpy(&out_size, &alpha, temp.data(), &incx, out, &incy);
    }
}

////////////////////
/// Hmat Hmat sumexpression_fast (07/24)
/// OK : test passed 10/07/2024
//////////////////

template <typename CoefficientPrecision, typename CoordinatePrecision>
HMatrix<CoefficientPrecision, CoordinatePrecision> HMatrix<CoefficientPrecision, CoordinatePrecision>::hmatrix_product_fast(const HMatrix<CoefficientPrecision, CoordinatePrecision> &B) const {
    if (this->get_source_cluster().get_size() != B.get_target_cluster().get_size()) {
        std::cerr << " Hmatrix unaligned" << std::endl;
    }
    HMatrix root_hmatrix(this->get_target_cluster(), B.get_source_cluster());
    root_hmatrix.set_admissibility_condition(this->m_tree_data->m_admissibility_condition);
    auto admissibility = this->m_tree_data->m_admissibility_condition;
    root_hmatrix.set_low_rank_generator(this->m_tree_data->m_low_rank_generator);
    root_hmatrix.set_eta(this->m_tree_data->m_eta);
    root_hmatrix.set_epsilon(this->m_tree_data->m_epsilon);
    root_hmatrix.set_maximal_block_size(this->m_tree_data->m_maxblocksize);
    root_hmatrix.set_minimal_target_depth(this->m_tree_data->m_minimal_target_depth);
    root_hmatrix.set_minimal_source_depth(this->m_tree_data->m_minimal_source_depth);
    SumExpression_fast<CoefficientPrecision, CoordinatePrecision> root_sum_expression(this, &B);
    root_hmatrix.recursive_build_hmatrix_product_fast(root_sum_expression, &admissibility);
    return root_hmatrix;
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product_fast(const SumExpression_fast<CoefficientPrecision, CoordinatePrecision> &sum_expr, std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> *adm) {
    auto &target_cluster  = this->get_target_cluster();
    auto &source_cluster  = this->get_source_cluster();
    auto &target_children = target_cluster.get_children();
    auto &source_children = source_cluster.get_children();
    auto admissible       = this->compute_admissibility(target_cluster, source_cluster);
    if (sum_expr.get_restr() == true) {
        if (admissible) {
            // ACA sur sumexpr
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(sum_expr, *this->m_tree_data->m_low_rank_generator, this->get_target_cluster(), this->get_source_cluster(), -1, this->get_epsilon());
            if ((lr.get_U().nb_rows() == 0) or (lr.get_U().nb_cols() == 0) or (lr.get_V().nb_rows() == 0) or (lr.get_V().nb_cols() == 0)) {
                // ACA plante , si onpeut on continue a decendre
                if ((target_children.size() > 0) and (source_children.size() > 0)) {
                    for (const auto &target_child : target_children) {
                        for (const auto &source_child : source_children) {
                            HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child       = this->add_child(target_child.get(), source_child.get());
                            SumExpression_fast<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.restrict_ACA(*target_child, *source_child);
                            hmatrix_child->recursive_build_hmatrix_product_fast(sum_restr, adm);
                        }
                    }
                } else {
                    // on peut pas descendre
                    this->compute_dense_data(sum_expr);
                }
            } else {
                // ACA a marché on affecte
                this->set_low_rank_data(lr);
            }

        } else {
            // on est pas admissible on descend si on peut
            if ((target_children.size() > 0) and (source_children.size() > 0)) {
                for (const auto &target_child : target_children) {
                    for (const auto &source_child : source_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child       = this->add_child(target_child.get(), source_child.get());
                        SumExpression_fast<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.restrict_ACA(*target_child, *source_child);
                        hmatrix_child->recursive_build_hmatrix_product_fast(sum_restr, adm);
                    }
                }
            } else {
                // on peut pas descendre
                this->compute_dense_data(sum_expr);
            }
        }
    } else {
        if (this->get_target_cluster().get_children().size() == 0 && this->get_source_cluster().get_children().size() == 0) {
            this->compute_dense_data(sum_expr);
        } else {
            Matrix<CoefficientPrecision> mat(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
            sum_expr.copy_submatrix(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset(), mat.data());
            this->assign(&mat, this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
        }
    }
}

///////
/// prod hmat triangulaires

template <typename CoefficientPrecision, typename CoordinatePrecision>
HMatrix<CoefficientPrecision, CoordinatePrecision> HMatrix<CoefficientPrecision, CoordinatePrecision>::hmatrix_product_fast_triangulaire(char transa, char transb, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B) const {
    if (this->get_source_cluster().get_size() != B.get_target_cluster().get_size()) {
        std::cerr << " Hmatrix unaligned" << std::endl;
    }
    HMatrix root_hmatrix(this->get_target_cluster(), B.get_source_cluster());
    root_hmatrix.set_admissibility_condition(this->m_tree_data->m_admissibility_condition);
    auto admissibility = this->m_tree_data->m_admissibility_condition;
    root_hmatrix.set_low_rank_generator(this->m_tree_data->m_low_rank_generator);
    root_hmatrix.set_eta(this->m_tree_data->m_eta);
    root_hmatrix.set_epsilon(this->m_tree_data->m_epsilon);
    root_hmatrix.set_maximal_block_size(this->m_tree_data->m_maxblocksize);
    root_hmatrix.set_minimal_target_depth(this->m_tree_data->m_minimal_target_depth);
    root_hmatrix.set_minimal_source_depth(this->m_tree_data->m_minimal_source_depth);
    SumExpression_fast<CoefficientPrecision, CoordinatePrecision> root_sum_expression(this, &B);
    root_hmatrix.recursive_build_hmatrix_product_fast_triangulaire(transa, transb, root_sum_expression, &admissibility);
    return root_hmatrix;
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product_fast_triangulaire(char transa, char transb, const SumExpression_fast<CoefficientPrecision, CoordinatePrecision> &sum_expr, std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> *adm) {
    auto &target_cluster  = this->get_target_cluster();
    auto &source_cluster  = this->get_source_cluster();
    auto &target_children = target_cluster.get_children();
    auto &source_children = source_cluster.get_children();
    auto admissible       = this->compute_admissibility(target_cluster, source_cluster);
    if (sum_expr.get_restr() == true) {
        if (admissible) {
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(sum_expr, *this->m_tree_data->m_low_rank_generator, this->get_target_cluster(), this->get_source_cluster(), -1, this->get_epsilon());
            if ((lr.get_U().nb_rows() == 0) or (lr.get_U().nb_cols() == 0) or (lr.get_V().nb_rows() == 0) or (lr.get_V().nb_cols() == 0)) {
                if ((target_children.size() > 0) and (source_children.size() > 0)) {
                    for (const auto &target_child : target_children) {
                        for (const auto &source_child : source_children) {
                            HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child       = this->add_child(target_child.get(), source_child.get());
                            SumExpression_fast<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.restrict_ACA_triangulaire(transa, transb, *target_child, *source_child);
                            hmatrix_child->recursive_build_hmatrix_product_fast_triangulaire(transa, transb, sum_restr, adm);
                        }
                    }
                } else {
                    this->compute_dense_data(sum_expr);
                }
            } else {
                this->set_low_rank_data(lr);
            }

        } else {
            if ((target_children.size() > 0) and (source_children.size() > 0)) {
                for (const auto &target_child : target_children) {
                    for (const auto &source_child : source_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child       = this->add_child(target_child.get(), source_child.get());
                        SumExpression_fast<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.restrict_ACA_triangulaire(transa, transb, *target_child, *source_child);
                        hmatrix_child->recursive_build_hmatrix_product_fast_triangulaire(transa, transb, sum_restr, adm);
                    }
                }
            } else {
                this->compute_dense_data(sum_expr);
            }
        }
    } else {
        if (this->get_target_cluster().get_children().size() == 0 && this->get_source_cluster().get_children().size() == 0) {
            this->compute_dense_data(sum_expr);
        } else {
            Matrix<CoefficientPrecision> mat(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
            sum_expr.copy_submatrix(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset(), mat.data());
            this->assign(&mat, this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
        }
    }
}

/////////////////////////////////////////////////////////////////////////////

/////
// plus egal : Addition de matrice bloc
/////
template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::plus_egal(const CoefficientPrecision &alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> *B) {

    if ((this->get_target_cluster().get_size() != B->get_target_cluster().get_size()) || (this->get_source_cluster().get_size() != B->get_source_cluster().get_size())) {
        std::cerr << " Hmatrices unaligned in 'plus_egal' " << std::endl;
    } else {
        if (this->get_children().size() == 0) {
            if (B->get_target_cluster().get_size() == this->get_target_cluster().get_size() && B->get_source_cluster().get_size() == this->get_source_cluster().get_size()) {
                // feuille alligné, on travail normal bloc par bloc-----------------> ca devrait toujours être le cas
                if (B->is_low_rank() && this->is_low_rank()) {
                    Matrix<CoefficientPrecision> Ualpha(B->get_low_rank_data()->get_U().nb_rows(), B->get_low_rank_data()->get_U().nb_cols());

                    std::transform(B->get_low_rank_data()->get_U().data(), B->get_low_rank_data()->get_U().data() + B->get_low_rank_data()->get_U().nb_rows() * B->get_low_rank_data()->get_U().nb_cols(), Ualpha.data(),
                                   [alpha](double val) { // capture alpha by value
                                       return alpha * val;
                                   });
                    SumExpression_fast<CoefficientPrecision, CoordinatePrecision> Sumexpr;
                    std::vector<Matrix<CoefficientPrecision>> sr_temp(4);
                    sr_temp[0] = this->get_low_rank_data()->get_U();
                    sr_temp[1] = this->get_low_rank_data()->get_V();
                    sr_temp[2] = Ualpha;
                    sr_temp[3] = B->get_low_rank_data()->get_V();
                    Sumexpr.set_sr(sr_temp, this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_new(Sumexpr, *this->get_low_rank_generator(), this->get_target_cluster(), this->get_source_cluster(), -1, this->get_epsilon());
                    if (lr_new.get_U().nb_cols() > 0) {
                        this->set_low_rank_data(lr_new);

                    } else {
                        int m                       = this->get_target_cluster().get_size();
                        int n                       = this->get_source_cluster().get_size();
                        int k                       = this->get_low_rank_data()->get_U().nb_cols();
                        CoefficientPrecision alpha0 = 1.0;
                        CoefficientPrecision beta0  = 0.0; // Initialement 0 pour la première opération
                        int lda                     = m;
                        int ldb                     = k;
                        int ldc                     = m;
                        char trans                  = 'N';

                        Matrix<CoefficientPrecision> bloc(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
                        add_matrix_matrix_product('N', 'N', 1.0, this->get_low_rank_data()->get_U(), this->get_low_rank_data()->get_V(), 1.0, bloc);
                        add_matrix_matrix_product('N', 'N', alpha, B->get_low_rank_data()->get_U(), B->get_low_rank_data()->get_V(), 1.0, bloc);

                        if (this->get_target_cluster().get_children().size() == 0) {
                            // std::unique_ptr<Matrix<CoefficientPrecision>> mat = std::make_unique<Matrix<CoefficientPrecision>>(bloc);
                            this->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(bloc));
                        } else {
                            this->assign(&bloc, this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                        }
                    }
                } else if (B->is_dense() && this->is_dense()) {
                    int size = this->get_target_cluster().get_size() * this->get_source_cluster().get_size();
                    int inc  = 1;
                    Matrix<CoefficientPrecision> hdense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
                    copy_to_dense(*this, hdense.data());
                    Blas<CoefficientPrecision>::axpy(&size, &alpha, B->get_dense_data()->data(), &inc, hdense.data(), &inc);

                    if (this->get_target_cluster().get_children().size() == 0) {
                        // std::unique_ptr<Matrix<CoefficientPrecision>> mat = std::make_unique<Matrix<CoefficientPrecision>>(hdense);
                        this->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(hdense));
                    } else {
                        this->assign(&hdense, this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                    }

                } else {
                    Matrix<CoefficientPrecision> hdense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
                    Matrix<CoefficientPrecision> bdense(B->get_target_cluster().get_size(), B->get_source_cluster().get_size());
                    copy_to_dense(*this, hdense.data());
                    copy_to_dense(*B, bdense.data());
                    int inc  = 1.0;
                    int size = hdense.nb_rows() * hdense.nb_cols();
                    Blas<CoefficientPrecision>::axpy(&size, &alpha, bdense.data(), &inc, hdense.data(), &inc);

                    if (this->get_target_cluster().get_children().size() == 0) {
                        // std::unique_ptr<Matrix<CoefficientPrecision>> mat = std::make_unique<Matrix<CoefficientPrecision>>(hdense);
                        this->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(hdense));
                    } else {
                        this->assign(&hdense, this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                    }
                }
            } else {
                std::cerr << " Hmatrices unaligned in 'plus_egal' " << std::endl;
            }
        } else {
            bool flag = true;
            for (auto &child : this->get_children()) {
                auto B_child = B->get_block(child->get_target_cluster().get_size(), child->get_source_cluster().get_size(), child->get_target_cluster().get_offset(), child->get_source_cluster().get_offset());

                if (B_child->get_target_cluster().get_size() != child->get_target_cluster().get_size() || B_child->get_source_cluster().get_size() != child->get_source_cluster().get_size()) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                for (auto &child : this->get_children()) {
                    auto B_child = B->get_block(child->get_target_cluster().get_size(), child->get_source_cluster().get_size(), child->get_target_cluster().get_offset(), child->get_source_cluster().get_offset());
                    child->plus_egal(alpha, B_child);
                }

            } else {
                // corner case , on fait quoi si B il est bizzard ?
                Matrix<CoefficientPrecision> bdense(B->get_target_cluster().get_size(), B->get_source_cluster().get_size());
                copy_to_dense(*B, bdense.data());
                Matrix<CoefficientPrecision> dense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
                copy_to_dense(*this, dense.data());
                int size = this->get_target_cluster().get_size() * this->get_source_cluster().get_size();
                int inc  = 1;
                if (B->get_target_cluster().get_size() == this->get_target_cluster().get_size() && B->get_source_cluster().get_size() == this->get_source_cluster().get_size()) {
                    // std::cout << "cas 1 " << std::endl;
                    Blas<CoefficientPrecision>::axpy(&size, &alpha, bdense.data(), &inc, dense.data(), &inc);
                    if (this->get_target_cluster().get_children().size() == 0) {
                        // std::unique_ptr<Matrix<CoefficientPrecision>> mat = std::make_unique<Matrix<CoefficientPrecision>>(dense);
                        this->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(dense));
                    } else {
                        this->assign(&dense, this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                    }
                } else {

                    auto brestr = copy_sub_matrix(bdense, this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset() - B->get_target_cluster().get_offset(), this->get_source_cluster().get_offset() - B->get_source_cluster().get_offset());
                    Blas<CoefficientPrecision>::axpy(&size, &alpha, brestr.data(), &inc, dense.data(), &inc);
                    if (this->get_target_cluster().get_children().size() == 0) {
                        // std::unique_ptr<Matrix<CoefficientPrecision>> mat = std::make_unique<Matrix<CoefficientPrecision>>(dense);
                        this->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(dense));
                    } else {
                        this->assign(&dense, this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                    }
                }
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::save_plot(const std::string &outputname) const {

    std::ofstream outputfile((outputname + ".csv").c_str());

    if (outputfile) {
        outputfile << this->get_target_cluster().get_size() << "," << this->get_source_cluster().get_size() << std::endl;
        auto leaves = this->get_leaves();
        for (auto l : leaves) {
            outputfile << l->get_target_cluster().get_offset() << "," << l->get_target_cluster().get_size() << "," << l->get_source_cluster().get_offset() << "," << l->get_source_cluster().get_size() << ",";
            if (l->is_low_rank()) {
                auto ll = l->get_low_rank_data();
                int k   = ll->rank_of();
                outputfile << k << "\n";
            } else {
                outputfile << -1 << "\n";
            }
        }
        outputfile.close();
    } else {
        std::cout << "Unable to create " << outputname << std::endl; // LCOV_EXCL_LINE
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void copy_to_dense(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, CoefficientPrecision *ptr) {

    int target_offset = hmatrix.get_target_cluster().get_offset();
    int source_offset = hmatrix.get_source_cluster().get_offset();
    int target_size   = hmatrix.get_target_cluster().get_size();

    for (auto leaf : hmatrix.get_leaves()) {
        int local_nr   = leaf->get_target_cluster().get_size();
        int local_nc   = leaf->get_source_cluster().get_size();
        int row_offset = leaf->get_target_cluster().get_offset() - target_offset;
        int col_offset = leaf->get_source_cluster().get_offset() - source_offset;
        if (leaf->is_dense()) {
            for (int k = 0; k < local_nc; k++) {
                for (int j = 0; j < local_nr; j++) {
                    ptr[j + row_offset + (k + col_offset) * target_size] = (*leaf->get_dense_data())(j, k);
                }
            }
        } else {

            Matrix<CoefficientPrecision> low_rank_to_dense(local_nr, local_nc);
            leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
            for (int k = 0; k < local_nc; k++) {
                for (int j = 0; j < local_nr; j++) {
                    ptr[j + row_offset + (k + col_offset) * target_size] = low_rank_to_dense(j, k);
                }
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void copy_to_dense_in_user_numbering(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, CoefficientPrecision *ptr) {

    const auto &target_cluster = hmatrix.get_target_cluster();
    const auto &source_cluster = hmatrix.get_source_cluster();
    if (!target_cluster.is_root() && !is_cluster_on_partition(target_cluster)) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Target cluster is neither root nor local, permutation is not stable and copy_to_dense_in_user_numbering cannot be used."); // LCOV_EXCL_LINE
    }

    if (!source_cluster.is_root() && !is_cluster_on_partition(source_cluster)) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Source cluster is neither root nor local, permutation is not stable and copy_to_dense_in_user_numbering cannot be used."); // LCOV_EXCL_LINE
    }

    if (is_cluster_on_partition(target_cluster) && !(target_cluster.is_permutation_local())) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Target cluster is local, but permutation is not local, copy_to_dense_in_user_numbering cannot be used"); // LCOV_EXCL_LINE}
    }

    if (is_cluster_on_partition(source_cluster) && !(source_cluster.is_permutation_local())) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Source cluster is local, but permutation is not local, copy_to_dense_in_user_numbering cannot be used"); // LCOV_EXCL_LINE}
    }
    int target_offset              = target_cluster.get_offset();
    int source_offset              = source_cluster.get_offset();
    int target_size                = target_cluster.get_size();
    const auto &target_permutation = target_cluster.get_permutation();
    const auto &source_permutation = source_cluster.get_permutation();

    for (auto leaf : hmatrix.get_leaves()) {
        int local_nr   = leaf->get_target_cluster().get_size();
        int local_nc   = leaf->get_source_cluster().get_size();
        int row_offset = leaf->get_target_cluster().get_offset();
        int col_offset = leaf->get_source_cluster().get_offset();

        if (leaf->is_dense()) {
            for (int k = 0; k < local_nc; k++) {
                for (int j = 0; j < local_nr; j++) {
                    ptr[target_permutation[j + row_offset] - target_offset + (source_permutation[k + col_offset] - source_offset) * target_size] = (*leaf->get_dense_data())(j, k);
                }
            }
        } else {

            Matrix<CoefficientPrecision> low_rank_to_dense(local_nr, local_nc);
            leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
            for (int k = 0; k < local_nc; k++) {
                for (int j = 0; j < local_nr; j++) {
                    ptr[target_permutation[j + row_offset] - target_offset + (source_permutation[k + col_offset] - source_offset) * target_size] = low_rank_to_dense(j, k);
                }
            }
        }
    }
    if (hmatrix.get_symmetry_for_leaves() != 'N') {
        for (auto leaf : hmatrix.get_leaves_for_symmetry()) {
            int local_nr   = leaf->get_target_cluster().get_size();
            int local_nc   = leaf->get_source_cluster().get_size();
            int row_offset = leaf->get_target_cluster().get_offset();
            int col_offset = leaf->get_source_cluster().get_offset();

            if (leaf->get_target_cluster().get_offset() == leaf->get_source_cluster().get_offset() and hmatrix.get_UPLO_for_leaves() == 'L') {
                for (int k = 0; k < local_nc; k++) {
                    for (int j = k + 1; j < local_nr; j++) {
                        ptr[source_permutation[k + col_offset] - target_offset + (target_permutation[j + row_offset] - source_offset) * target_size] = hmatrix.get_symmetry_for_leaves() == 'S' ? ptr[target_permutation[j + row_offset] - target_offset + (source_permutation[k + col_offset] - source_offset) * target_size] : conj_if_complex(ptr[target_permutation[j + row_offset] - target_offset + (source_permutation[k + col_offset] - source_offset) * target_size]);
                    }
                }
            } else if (leaf->get_target_cluster().get_offset() == leaf->get_source_cluster().get_offset() and hmatrix.get_UPLO_for_leaves() == 'U') {
                for (int k = 0; k < local_nc; k++) {
                    for (int j = 0; j < k; j++) {
                        ptr[source_permutation[k + col_offset] - target_offset + (target_permutation[j + row_offset] - source_offset) * target_size] = hmatrix.get_symmetry_for_leaves() == 'S' ? ptr[target_permutation[j + row_offset] - target_offset + (source_permutation[k + col_offset] - source_offset) * target_size] : conj_if_complex(ptr[target_permutation[j + row_offset] - target_offset + (source_permutation[k + col_offset] - source_offset) * target_size]);
                    }
                }
            } else {
                for (int k = 0; k < local_nc; k++) {
                    for (int j = 0; j < local_nr; j++) {
                        ptr[source_permutation[k + col_offset] - target_offset + (target_permutation[j + row_offset] - source_offset) * target_size] = hmatrix.get_symmetry_for_leaves() == 'S' ? ptr[target_permutation[j + row_offset] - target_offset + (source_permutation[k + col_offset] - source_offset) * target_size] : conj_if_complex(ptr[target_permutation[j + row_offset] - target_offset + (source_permutation[k + col_offset] - source_offset) * target_size]);
                    }
                }
            }
        }
    }
}

////////////// Arthur je rajoute ca pour la compression parce que get_low_rank_data est en private...
template <class CoefficientPrecision, class CoordinatePrecision>
CoefficientPrecision get_compression(const HMatrix<CoefficientPrecision, CoordinatePrecision> &A) {
    CoefficientPrecision sum = 0;
    for (auto &l : A.get_leaves()) {
        if (l->is_dense()) {
            sum += l->get_target_cluster().get_size() * l->get_source_cluster().get_size();
        } else {
            sum += l->get_low_rank_data()->rank_of() * (l->get_target_cluster().get_size() + l->get_source_cluster().get_size());
        }
    }
    return (1.0 - sum / (A.get_target_cluster().get_size() * A.get_source_cluster().get_size()));
}
template <typename CoefficientPrecision, typename CoordinatePrecision>
void copy_diagonal(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, CoefficientPrecision *ptr) {
    if (hmatrix.get_target_cluster().get_offset() != hmatrix.get_source_cluster().get_offset() || hmatrix.get_target_cluster().get_size() != hmatrix.get_source_cluster().get_size()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Matrix is not square a priori, get_local_diagonal cannot be used"); // LCOV_EXCL_LINE
        // throw std::logic_error("[Htool error] Matrix is not square a priori, get_local_diagonal cannot be used");                       // LCOV_EXCL_LINE
    }

    int target_offset = hmatrix.get_target_cluster().get_offset();
    int source_offset = hmatrix.get_source_cluster().get_offset();

    for (auto leaf : hmatrix.get_leaves()) {
        int local_nr = leaf->get_target_cluster().get_size();
        int local_nc = leaf->get_source_cluster().get_size();
        int offset_i = leaf->get_target_cluster().get_offset() - target_offset;
        int offset_j = leaf->get_source_cluster().get_offset() - source_offset;
        if (leaf->is_dense() && offset_i == offset_j) {
            for (int k = 0; k < std::min(local_nr, local_nc); k++) {
                ptr[k + offset_i] = (*leaf->get_dense_data())(k, k);
            }
        } else if (leaf->is_low_rank() && offset_i == offset_j) { // pretty rare...
            Matrix<CoefficientPrecision> low_rank_to_dense(local_nc, local_nr);
            leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
            for (int k = 0; k < std::min(local_nr, local_nc); k++) {
                ptr[k + offset_i] = (low_rank_to_dense)(k, k);
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void copy_diagonal_in_user_numbering(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, CoefficientPrecision *ptr) {
    if (hmatrix.get_target_cluster().get_offset() != hmatrix.get_source_cluster().get_offset() || hmatrix.get_target_cluster().get_size() != hmatrix.get_source_cluster().get_size()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Matrix is not square a priori, get_local_diagonal cannot be used"); // LCOV_EXCL_LINE
    }

    if (!(hmatrix.get_target_cluster().is_root() && hmatrix.get_source_cluster().is_root()) && !(is_cluster_on_partition(hmatrix.get_target_cluster()) && is_cluster_on_partition(hmatrix.get_source_cluster()))) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Clusters are neither root nor local, permutations are not stable and copy_diagonal_in_user_numbering cannot be used."); // LCOV_EXCL_LINE
    }

    if ((is_cluster_on_partition(hmatrix.get_target_cluster()) && is_cluster_on_partition(hmatrix.get_source_cluster())) && !(hmatrix.get_target_cluster().is_permutation_local() && hmatrix.get_source_cluster().is_permutation_local())) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Clusters are local, but permutations are not local, copy_diagonal_in_user_numbering cannot be used"); // LCOV_EXCL_LINE}
    }

    const auto &permutation = hmatrix.get_target_cluster().get_permutation();
    int target_offset       = hmatrix.get_target_cluster().get_offset();
    // int source_offset       = hmatrix.get_source_cluster().get_offset();

    for (auto leaf : hmatrix.get_leaves()) {
        int local_nr = leaf->get_target_cluster().get_size();
        int local_nc = leaf->get_source_cluster().get_size();
        int offset_i = leaf->get_target_cluster().get_offset();
        int offset_j = leaf->get_source_cluster().get_offset();
        if (leaf->is_dense() && offset_i == offset_j) {
            for (int k = 0; k < std::min(local_nr, local_nc); k++) {
                ptr[permutation[k + offset_i] - target_offset] = (*leaf->get_dense_data())(k, k);
            }
        } else if (leaf->is_low_rank() && offset_i == offset_j) { // pretty rare...
            Matrix<CoefficientPrecision> low_rank_to_dense(local_nc, local_nr);
            leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
            for (int k = 0; k < std::min(local_nr, local_nc); k++) {
                ptr[permutation[k + offset_i] - target_offset] = (low_rank_to_dense)(k, k);
            }
        }
    }
}

} // namespace htool
#endif
