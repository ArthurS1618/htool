#ifndef HTOOL_HMATRIX_HPP
#define HTOOL_HMATRIX_HPP

#if defined(_OPENMP)
#    include <omp.h>
#endif
#include "../basic_types/tree.hpp"
#include "../clustering/cluster_node.hpp"
#include "../misc/logger.hpp"
#include "hmatrix_tree_data.hpp"
#include "interfaces/virtual_admissibility_condition.hpp"
#include "interfaces/virtual_dense_blocks_generator.hpp"
#include "interfaces/virtual_generator.hpp"
#include "lrmat/lrmat.hpp"
#include "sum_expressions.hpp"
#include <cblas.h>
#include <mpi.h>

namespace htool {

// Class
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class HMatrix : public TreeNode<HMatrix<CoefficientPrecision, CoordinatePrecision>, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>> {

  private:
    // Data members
    const Cluster<CoordinatePrecision> *m_target_cluster, *m_source_cluster; // child's clusters are non owning
    char m_symmetry{'N'};
    char m_UPLO{'N'};

    std::unique_ptr<Matrix<CoefficientPrecision>> m_dense_data{nullptr};
    std::unique_ptr<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> m_low_rank_data{nullptr};

    // Cached leaves
    // std::vector<HMatrix *> m_dense_leaves{};
    // std::vector<HMatrix *> m_dense_leaves_in_diagonal_block{};
    // std::vector<HMatrix *> m_diagonal_dense_leaves{};
    // std::vector<HMatrix *> m_low_rank_leaves{};
    // std::vector<HMatrix *> m_low_rank_leaves_in_diagonal_block{};
    // std::vector<HMatrix *> m_diagonal_low_rank_leaves{};
    mutable std::vector<const HMatrix *> m_leaves{};
    mutable std::vector<const HMatrix *> m_leaves_for_symmetry{};
    mutable char m_symmetry_type_for_leaves{'N'};
    // std::vector<HMatrix *> m_leaves_in_diagonal_block{};

    enum class StorageType {
        Dense,
        LowRank,
        Hierarchical
    };
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

    void recursive_build_hmatrix_product(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr);
    void recursive_build_hmatrix_product_new(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr);

  public:
    // Root constructor
    HMatrix(std::shared_ptr<const Cluster<CoordinatePrecision>> target_cluster, std::shared_ptr<const Cluster<CoordinatePrecision>> source_cluster) : TreeNode<HMatrix, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(), m_target_cluster(target_cluster.get()), m_source_cluster(source_cluster.get()) {
        this->m_tree_data->m_target_cluster_tree = target_cluster;
        this->m_tree_data->m_source_cluster_tree = source_cluster;
    }

    // Root constructor from a sub hmatrix
    HMatrix(std::shared_ptr<const Cluster<CoordinatePrecision>> root_target_cluster, std::shared_ptr<const Cluster<CoordinatePrecision>> root_source_cluster, const Cluster<CoordinatePrecision> *target_cluster, const Cluster<CoordinatePrecision> *source_cluster) : TreeNode<HMatrix, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(), m_target_cluster(target_cluster), m_source_cluster(source_cluster) {
        this->m_tree_data->m_target_cluster_tree = root_target_cluster;
        this->m_tree_data->m_source_cluster_tree = root_source_cluster;
    };

    // Child constructor
    HMatrix(const HMatrix &parent, const Cluster<CoordinatePrecision> *target_cluster, const Cluster<CoordinatePrecision> *source_cluster) : TreeNode<HMatrix, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(parent), m_target_cluster(target_cluster), m_source_cluster(source_cluster) {}

    // no copy
    HMatrix(const HMatrix &)                = delete;
    HMatrix &operator=(const HMatrix &)     = delete;
    HMatrix(HMatrix &&) noexcept            = default;
    HMatrix &operator=(HMatrix &&) noexcept = default;
    virtual ~HMatrix()                      = default;

    // HMatrix getters
    const Cluster<CoordinatePrecision> &get_target_cluster() const { return *m_target_cluster; }
    const Cluster<CoordinatePrecision> &get_source_cluster() const { return *m_source_cluster; }
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
    const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> *get_low_rank_data() const { return m_low_rank_data.get(); }
    char get_symmetry() const { return m_symmetry; }
    char get_UPLO() const { return m_UPLO; }
    const HMatrixTreeData<CoefficientPrecision, CoordinatePrecision> *get_hmatrix_tree_data() const { return this->m_tree_data.get(); }

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
    void set_diagonal_hmatrix(const HMatrix<CoefficientPrecision, CoordinatePrecision> *diagonal_hmatrix) { this->m_tree_data->m_block_diagonal_hmatrix = diagonal_hmatrix; }
    void set_minimal_target_depth(unsigned int minimal_target_depth) { this->m_tree_data->m_minimal_target_depth = minimal_target_depth; }
    void set_minimal_source_depth(unsigned int minimal_source_depth) { this->m_tree_data->m_minimal_source_depth = minimal_source_depth; }

    // HMatrix Tree setters
    const HMatrix<CoefficientPrecision, CoordinatePrecision> *get_diagonal_hmatrix() const { return this->m_tree_data->m_block_diagonal_hmatrix; }
    char get_symmetry_for_leaves() const { return m_symmetry_type_for_leaves; }

    double get_compression() const {
        double compr = 0.0;
        auto leaves  = this->m_leaves;
        for (int k = 0; k < leaves.size(); ++k) {
            auto Hk = leaves[k];
            if (Hk->is_dense()) {
                compr += Hk->get_target_cluster().get_size() * Hk->get_source_cluster().get_size();
            } else if (Hk->is_low_rank()) {
                auto temp = (Hk->m_low_rank_data->rank_of() * (Hk->get_target_cluster().get_size() + Hk->get_source_cluster().get_size()));
                compr     = compr + temp;
            }
        }
        compr = compr / (this->get_target_cluster().get_size() * this->get_source_cluster().get_size());
        return 1 - compr;
    }
    std::vector<double> get_rank_info() {
        auto leaves = this->m_leaves;
        std::vector<double> rk;
        int min_rk  = 1000;
        int max_rk  = 0;
        double mean = 0.0;
        double cpt  = 0.0;
        for (int k = 0; k < leaves.size(); ++k) {
            auto lr = leaves[k];
            if (lr->is_low_rank()) {
                auto llr = lr->get_low_rank_data();
                min_rk   = std::min(min_rk, llr->rank_of());
                max_rk   = std::max(max_rk, llr->rank_of());
                mean += llr->rank_of();
                cpt += 1;
            }
        }
        mean = mean / (cpt * 1.0);
        rk.push_back(min_rk);
        rk.push_back(max_rk);
        rk.push_back(mean);
        return rk;
    }

    // Infos
    // const std::map<std::string, std::string> &get_infos() const { return infos; }
    // std::string get_infos(const std::string &key) const { return infos[key]; }
    // void add_info(const std::string &keyname, const std::string &value) const { infos[keyname] = value; }
    // void print_infos() const;
    // void save_infos(const std::string &outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string &sep = " = ") const;
    // underlying_type<CoefficientPrecision> compressed_size() const;
    // double compression_ratio() const;
    // double space_saving() const;
    // friend underlying_type<CoefficientPrecision> Frobenius_absolute_error<CoefficientPrecision>(const HMatrix<CoefficientPrecision,CoordinatePrecision> &B, const VirtualGenerator<CoefficientPrecision> &A);

    // // // Output structure
    // void save_plot(const std::string &outputname) const;
    // std::vector<DisplayBlock> get_output() const;

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

    // je rajoute ca
    void set_low_rank_data(const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lrmat) {
        m_low_rank_data = std::unique_ptr<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(new LowRankMatrix<CoefficientPrecision, CoordinatePrecision>(lrmat));
        m_storage_type  = StorageType::LowRank;
    }

    void set_dense_data(const Matrix<CoefficientPrecision> &M) {
        m_dense_data   = std::unique_ptr<Matrix<CoefficientPrecision>>(new Matrix<CoefficientPrecision>(M));
        m_storage_type = StorageType::Dense;
    }

    // Linear algebra
    void add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const;
    void add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const;

    HMatrix hmatrix_product(const HMatrix &B) const;
    // trouver x tq Lx= y
    // trouver x tqt Lx = y

    // je voulais vraiment pas faire cette fonction mais on est obligé de pouvoir atteindre des blocs qui sont pas forcemment des blocs fils dans ForwardM du coup je vois pas d'autres solutions
    HMatrix<CoefficientPrecision, CoordinatePrecision> *get_block(const int &szt, const int &szs, const int &oft, const int &ofs) const {
        if ((this->get_target_cluster().get_size() == szt) && (this->get_target_cluster().get_offset() == oft) && (this->get_source_cluster().get_size() == szs) && (this->get_source_cluster().get_offset() == ofs)) {
            return const_cast<HMatrix<CoefficientPrecision, CoordinatePrecision> *>(this);
        } else {
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

    HMatrix<CoefficientPrecision, CoordinatePrecision> *Get_block(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
        if (this->is_dense() or this->is_low_rank()) {
            Matrix<CoefficientPrecision> mat(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
            if (this->is_dense()) {
                mat = *this->get_dense_data();
            } else {
                mat = this->get_low_rank_data()->Get_U() * this->get_low_rank_data()->Get_V();
            }
            Matrix<CoefficientPrecision> sol(t.get_size(), s.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                for (int l = 0; l < s.get_size(); ++l) {
                    sol(k, l) = mat(k + t.get_offset(), l + s.get_offset());
                }
            }
            HMatrix res(this->m_tree_data->m_target_cluster_tree, this->m_tree_data->m_source_cluster_tree, &t, &s);
            // HMatrix res(t, s, this->m_tree_data->m_target_cluster, this->m_target_source);
            MatrixGenerator<CoefficientPrecision> ss(sol);
            res.compute_dense_data(ss);
            return &res;
        }
        if ((this->get_target_cluster().get_size() == t.get_size()) && (this->get_target_cluster().get_offset() == t.get_offset()) && (this->get_source_cluster().get_size() == s.get_size()) && (this->get_source_cluster().get_offset() == s.get_offset())) {
            return const_cast<HMatrix<CoefficientPrecision, CoordinatePrecision> *>(this);
        } else {
            for (auto &child : this->get_children()) {
                auto &target_cluster = child->get_target_cluster();
                auto &source_cluster = child->get_source_cluster();
                if (((t.get_size() + (t.get_offset() - child->get_target_cluster().get_offset()) <= child->get_target_cluster().get_size()) and (t.get_size() <= child->get_target_cluster().get_size()) and (t.get_offset() >= child->get_target_cluster().get_offset()))
                    and (((s.get_size() + (s.get_offset() - child->get_source_cluster().get_offset()) <= child->get_source_cluster().get_size()) and (s.get_size() <= child->get_source_cluster().get_size()) and (s.get_offset() >= child->get_source_cluster().get_offset())))) {
                    return child->Get_block(t, s);
                }
            }
        }
        return nullptr;
    }

    ///// A= A+B
    void Plus(const HMatrix<CoefficientPrecision, CoordinatePrecision> &B) const {
        // HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision> hmatrix_tree_builder(std::make_shared<Cluster<CoordinatePrecision>>(A.get_target_cluster()), std::make_shared<Cluster<CoordinatePrecision>>(A.get_target_cluster()), A.get_epsilon(), A.get_eta(), 'N', 'N');
        // Matrix<T> zeros(root_cluster_1->get_size(), root_cluster_1->get_size());
        // DenseGenerator<CoefficientPrecision> zz(test, *root_cluster_1, *root_cluster_1);
        auto leaves = this->get_leaves();
        for (auto &l : leaves) {
            auto a = this->Get_block(l->get_target_cluster(), l->get_source_cluster().get_size());
            auto b = B.Get_block(l->get_target_cluster(), l->get_source_cluster().get_size());
            Matrix<CoefficientPrecision>
                aa(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
            Matrix<CoefficientPrecision> bb(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
            copy_to_dense(*a, aa.data());
            copy_to_dense(*b, bb.data());
            auto c = aa + bb;
            this->Get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
        }
    }
    // void Moins(const HMatrix<CoefficientPrecision, CoordinatePrecision> &B) const {
    //     // HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision> hmatrix_tree_builder(std::make_shared<Cluster<CoordinatePrecision>>(A.get_target_cluster()), std::make_shared<Cluster<CoordinatePrecision>>(A.get_target_cluster()), A.get_epsilon(), A.get_eta(), 'N', 'N');
    //     // Matrix<T> zeros(root_cluster_1->get_size(), root_cluster_1->get_size());
    //     // DenseGenerator<CoefficientPrecision> zz(test, *root_cluster_1, *root_cluster_1);
    //     std::cout << "!" << std::endl;

    //     this->set_leaves_in_cache();
    //     std::cout << "!" << std::endl;

    //     auto leav = this->get_leaves();
    //     std::cout << "!" << std::endl;

    //     for (auto &l : leav) {
    //         std::cout << "on est sur une feuile" << std::endl;
    //         auto a = this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
    //         auto b = B.get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
    //         Matrix<CoefficientPrecision> aa(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //         Matrix<CoefficientPrecision> bb(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //         copy_to_dense(*a, aa.data());
    //         std::cout << " a ok " << std::endl;
    //         std::cout << b->get_target_cluster().get_size() << ',' << b->get_source_cluster().get_size() << b->get_target_cluster().get_offset() << ',' << b->get_source_cluster().get_offset() << std::endl;
    //         copy_to_dense(*b, bb.data());
    //         auto c = aa - bb;
    //         this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
    //         std::cout << "feuille calculée " << std::endl;
    //     }
    // }

    // void Moins(const HMatrix<CoefficientPrecision, CoordinatePrecision> &B) const {
    //     // HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision> hmatrix_tree_builder(std::make_shared<Cluster<CoordinatePrecision>>(A.get_target_cluster()), std::make_shared<Cluster<CoordinatePrecision>>(A.get_target_cluster()), A.get_epsilon(), A.get_eta(), 'N', 'N');
    //     // Matrix<T> zeros(root_cluster_1->get_size(), root_cluster_1->get_size());
    //     // DenseGenerator<CoefficientPrecision> zz(test, *root_cluster_1, *root_cluster_1);
    //     auto leaves = this->get_leaves();
    //     std::cout << "yo" << std::endl;
    //     for (auto &l : leaves) {
    //         auto a = this->Get_block(l->get_target_cluster(), l->get_source_cluster());
    //         std::cout << "a : " << a->get_target_cluster().get_size() << ',' << a->get_source_cluster().get_size() << std::endl;
    //         auto b = B.Get_block(l->get_target_cluster(), l->get_source_cluster());
    //         std::cout << "b :" << b->get_target_cluster().get_size() << ',' << b->get_source_cluster().get_size() << std::endl;

    //         Matrix<CoefficientPrecision> aa(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //         Matrix<CoefficientPrecision> bb(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //         copy_to_dense(*a, aa.data());
    //         copy_to_dense(*b, bb.data());
    //         auto c = aa - bb;
    //         this->Get_block(l->get_target_cluster(), l->get_source_cluster())->set_dense_data(c);
    //     }
    // }

    void Moins(const HMatrix<CoefficientPrecision, CoordinatePrecision> &B) const {
        // HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision> hmatrix_tree_builder(std::make_shared<Cluster<CoordinatePrecision>>(A.get_target_cluster()), std::make_shared<Cluster<CoordinatePrecision>>(A.get_target_cluster()), A.get_epsilon(), A.get_eta(), 'N', 'N');
        // Matrix<T> zeros(root_cluster_1->get_size(), root_cluster_1->get_size());
        // DenseGenerator<CoefficientPrecision> zz(test, *root_cluster_1, *root_cluster_1);
        auto leaves = this->get_leaves();
        for (auto &l : leaves) {
            auto a = this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
            auto b = B.get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
            if (b == nullptr) {
                Matrix<CoefficientPrecision> temp(B.get_target_cluster().get_size(), B.get_source_cluster().get_size());
                copy_to_dense(B, temp.data());

                Matrix<CoefficientPrecision> Bb(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                for (int k = 0; k < l->get_target_cluster().get_size(); ++k) {
                    for (int ll = 0; ll < l->get_source_cluster().get_size(); ++ll) {
                        Bb(k, ll) = temp(k + l->get_target_cluster().get_offset() - B.get_target_cluster().get_offset(), ll + l->get_source_cluster().get_offset() - B.get_source_cluster().get_offset());
                    }
                }
                Matrix<CoefficientPrecision> aa(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                copy_to_dense(*a, aa.data());
                auto c = aa - Bb;
                this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
            } else {

                Matrix<CoefficientPrecision> aa(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                Matrix<CoefficientPrecision> bb(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                copy_to_dense(*a, aa.data());
                copy_to_dense(*b, bb.data());
                auto c = aa - bb;
                this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
            }
        }
    }

    // void Moins(const HMatrix &M) {
    //     if (M.is_dense() or M.is_low_rank()) {
    //         auto block = M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
    //         Matrix<CoefficientPrecision> temp(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
    //         if (this->is_dense()) {
    //             temp = *this->get_dense_data();
    //         } else if (this->is_low_rank()) {
    //             temp = this->get_low_rank_data()->Get_U() * this->get_low_rank_data()->Get_V();
    //         }
    //         Matrix<CoefficientPrecision> b(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
    //         copy_to_dense(*block, b.data());
    //         auto res = temp + -1 * b;
    //         this->set_dense_data(res);
    //     } else {
    //         for (auto &child : this->get_children()) {
    //             child->Moins(M);
    //         }
    //     }
    // }

    ///////////////////////////////////////////////////
    //////// Fonctions pour avoir des H matrices triangulaire
    // template < typename CoordinatePrecision> ,
    // void Get_U(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &M) {
    //     auto ts = auto block = this->get_block(t, s);
    //     if (ts->get_children().size() == 0) {
    //         if (t == s) {
    //             auto block = this->get_block(t, s);
    //             auto m     = block->get_dense_data();
    //             Matrix<CoefficientPrecision> mm(t.get_size(), s.get_size());
    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 for (int l = 0; l < k; ++l) {
    //                     mm(k, l) = m(k, l);
    //                 }
    //                 m(k, k) = 1;
    //             }

    //         } else if (t.get_offset() < s.get_offset()) {
    //             if ()
    //         }
    //     }

    void Get_U(HMatrix &M) const {
        if (this->get_children().size() > 0) {
            for (auto &child : this->get_children()) {
                if (child->get_target_cluster().get_offset() <= child->get_source_cluster().get_offset()) {
                    child->Get_U(M);
                }
            }
        } else {
            if (this->get_target_cluster() == this->get_source_cluster()) {
                auto m = *(this->get_dense_data());
                Matrix<CoefficientPrecision> mm(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
                for (int k = 0; k < this->get_target_cluster().get_size(); ++k) {
                    for (int l = k; l < this->get_source_cluster().get_size(); ++l) {
                        mm(k, l) = m(k, l);
                    }
                }
                M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset())->set_dense_data(mm);
            } else {

                auto &sub = *this->get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                auto temp = M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                temp      = &sub;
                // if (this->is_low_rank()) {
                //     M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset())->set_low_rank_data(this->get_low_rank_data());
                // } else {
                //     M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset())->set_dense_data(this->get_dense_data());
                // }
            }
        }
    }

    void Get_L(HMatrix &M) const {
        if (this->get_children().size() > 0) {
            for (auto &child : this->get_children()) {
                if (child->get_target_cluster().get_offset() <= child->get_source_cluster().get_offset()) {
                    child->Get_L(M);
                }
            }
        } else {
            if (this->get_target_cluster() == this->get_source_cluster()) {
                auto m = *(this->get_dense_data());
                Matrix<CoefficientPrecision> mm(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
                for (int k = 0; k < this->get_target_cluster().get_size(); ++k) {
                    mm(k, k) = 1;
                    for (int l = 0; l < k; ++l) {
                        mm(k, l) = m(k, l);
                    }
                }
                M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset())->set_dense_data(mm);
            } else {

                auto &sub = *this->get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                auto temp = M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                temp      = &sub;
            }
        }
    }

    // void build_up(const HMatrix &M) {
    //     int szt = this->gezt_target_cluster().get_size();
    //     int szs = this->get_source_cluster().get_size();
    //     int oft = this->get_target_cluster().get_offset();
    //     int ofs = this->get_source_cluster().get_offset();
    //     if (M.get_children().size() > 0) {
    //         for (auto & t : this->get_target_cluster().get_children()){
    //             for (auto & s  : this->get_source_cluster().get_children()){

    //                 auto child = this->add_child(t,s) ;
    //                 auto m_child = M.get_block( t->get_size() , s.get_size() , t.get_offset() , s.get_offset()) ;
    //                 child->build_up

    //             }
    //         }
    // }

    ///////////////////////////////////////////////////////////
    /// HMULT comme dans hackbush
    ////////////////////////////////////////////////

    ////////////////////////////////////
    // COmme dans Hackbush
    /////////////////////////

    // pour faire une h matrice de  zeros , on l'appelle :    on l'appelle avec M =     HMatrix<CoefficientPrecision, CoordinatePrecision> M (std::make_shared<Cluster<CoordinatePrecision>>(H->get_target_cluster()), std::make_shared<Cluster<CoordinatePrecision>>(H->get_source_cluster()));

    // void mmr(const HMatrix<CoefficientPrecision, CoordinatePrecision> *H, const HMatrix<CoefficientPrecision, CoordinatePrecision> *K, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, const Cluster<CoordinatePrecision> &r) {
    //     auto Hts         = H->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //     auto Ksr         = K->get_block(s.get_size(), r.get_offset(), s.get_offset(), r.get_size());
    //     auto &H_children = Hts->get_children();
    //     auto &K_children = Ksr->get_children();
    //     // std::shared_ptr<Cluster<CoordinatePrecision>> ptr_t = std::make_shared<Cluster<CoordinatePrecision>>(&*t);
    //     // std::shared_ptr<Cluster<CoordinatePrecision>> ptr_r = std::make_shared<Cluster<CoordinatePrecision>>(&*r);
    //     HMatrix<CoefficientPrecision, CoordinatePrecision> Z(std::make_shared<Cluster<CoordinatePrecision>>(t), std::make_shared<Cluster<CoordinatePrecision>>(r));
    //     if (H_children.size() == 0 or K_children.size() == 0) {
    //         Matrix<CoefficientPrecision> z(t.get_size(), s.get_size());
    //         if ((H_children.size() > 0) and (K_children.size() == 0)) {
    //             if (Ksr->is_dense()) {
    //                 z = Hts->hmat_lr(*Ksr->get_dense_data());
    //             } else {
    //                 z = Hts->hmat_lr(Ksr->get_low_rank_data()->Get_U() * Ksr->get_low_rank_data()->Get_V());
    //             }
    //         } else if ((H_children.size() == 0) and (K_children.size() > 0)) {
    //             if (Hts->is_dense()) {
    //                 z = Ksr->lr_hmat(*Hts->get_dense_data());
    //             } else {
    //                 z = Ksr->lr_hmat(Hts->get_low_rank_data()->Get_U() * Hts->get_low_rank_data()->Get_V());
    //             }
    //         } else if ((H_children.size() == 0) and (K_children.size() == 0)) {
    //             if (Hts->is_dense() and Ksr->is_dense()) {
    //                 z = *Hts->get_dense_data() * *Ksr->get_dense_data();
    //             } else if ((Hts->is_dense()) and (Ksr->is_low_rank())) {
    //                 z = *Hts->get_dense_data() * Ksr->get_low_rank_data()->Get_U() * Ksr->get_low_rank_data()->Get_V();
    //             } else if ((Hts->is_low_rank()) and (Ksr->is_dense())) {
    //                 z = Hts->get_low_rank_data()->Get_U() * Hts->get_low_rank_data()->Get_V() * *Ksr->get_dense_data();
    //             } else if ((Hts->is_low_rank()) and (Ksr->is_low_rank())) {
    //                 z = Hts->get_low_rank_data()->Get_U() * Hts->get_low_rank_data()->Get_V() * Ksr->get_low_rank_data()->Get_U() * Ksr->get_low_rank_data()->Get_V();
    //             }
    //         }
    //         // Z.set_dense_data(z);
    //     } else {
    //         auto &t_children = t.get_children();
    //         auto &s_children = s.get_children();
    //         auto &r_children = r.get_children();
    //         for (auto &tt : t_children) {
    //             for (auto &ss : s_children) {
    //                 for (auto &rr : r_children) {
    //                     Z.mmr(H, K, *tt, *ss, *rr);
    //                     //  Z.MMR(H, K, std::make_shared<Cluster<CoordinatePrecision>>(*tt), std::make_shared<Cluster<CoordinatePrecision>>(*ss), std::make_shared<Cluster<CoordinatePrecision>>(*rr));
    //                 }
    //             }
    //         }
    //     }
    //     Matrix<CoefficientPrecision> gen(t.get_size(), r.get_size());
    //     DenseGenerator<CoefficientPrecision> zer(gen);
    //     auto temp = this->get_block(t.get_size(), r.get_size(), t.get_offset(), r.get_offset());
    //     Z.Plus(*temp);
    // }

    // void MMR(const HMatrix<CoefficientPrecision, CoordinatePrecision> *H, const HMatrix<CoefficientPrecision, CoordinatePrecision> *K, const std::shared_ptr<Cluster<CoordinatePrecision>> t, const std::shared_ptr<Cluster<CoordinatePrecision>> s, const std::shared_ptr<Cluster<CoordinatePrecision>> r) {
    //     auto Hts         = H->get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset());
    //     auto Ksr         = K->get_block(s->get_size(), r->get_offset(), s->get_offset(), r->get_size());
    //     auto &H_children = Hts->get_children();
    //     auto &K_children = Ksr->get_children();
    //     // std::shared_ptr<Cluster<CoordinatePrecision>> ptr_t = std::make_shared<Cluster<CoordinatePrecision>>(&*t);
    //     // std::shared_ptr<Cluster<CoordinatePrecision>> ptr_r = std::make_shared<Cluster<CoordinatePrecision>>(&*r);
    //     HMatrix<CoefficientPrecision, CoordinatePrecision> Z(t, r);
    //     if (H_children.size() == 0 or K_children.size() == 0) {
    //         Matrix<CoefficientPrecision> z(t->get_size(), s->get_size());
    //         if ((H_children.size() > 0) and (K_children.size() == 0)) {
    //             if (Ksr->is_dense()) {
    //                 z = Hts->hmat_lr(*Ksr->get_dense_data());
    //             } else {
    //                 z = Hts->hmat_lr(Ksr->get_low_rank_data()->Get_U() * Ksr->get_low_rank_data()->Get_V());
    //             }
    //         } else if ((H_children.size() == 0) and (K_children.size() > 0)) {
    //             if (Hts->is_dense()) {
    //                 z = Ksr->lr_hmat(*Hts->get_dense_data());
    //             } else {
    //                 z = Ksr->lr_hmat(Hts->get_low_rank_data()->Get_U() * Hts->get_low_rank_data()->Get_V());
    //             }
    //         } else if ((H_children.size() == 0) and (K_children.size() == 0)) {
    //             if (Hts->is_dense() and Ksr->is_dense()) {
    //                 z = *Hts->get_dense_data() * *Ksr->get_dense_data();
    //             } else if ((Hts->is_dense()) and (Ksr->is_low_rank())) {
    //                 z = *Hts->get_dense_data() * Ksr->get_low_rank_data()->Get_U() * Ksr->get_low_rank_data()->Get_V();
    //             } else if ((Hts->is_low_rank()) and (Ksr->is_dense())) {
    //                 z = Hts->get_low_rank_data()->Get_U() * Hts->get_low_rank_data()->Get_V() * *Ksr->get_dense_data();
    //             } else if ((Hts->is_low_rank()) and (Ksr->is_low_rank())) {
    //                 z = Hts->get_low_rank_data()->Get_U() * Hts->get_low_rank_data()->Get_V() * Ksr->get_low_rank_data()->Get_U() * Ksr->get_low_rank_data()->Get_V();
    //             }
    //         }
    //         // Z.set_dense_data(z);
    //     } else {
    //         auto &t_children = t->get_children();
    //         auto &s_children = s->get_children();
    //         auto &r_children = r->get_children();
    //         for (auto &tt : t_children) {
    //             for (auto &ss : s_children) {
    //                 for (auto &rr : r_children) {
    //                     Z.mmr(H, K, *tt, *ss, *rr);
    //                     //  Z.MMR(H, K, std::make_shared<Cluster<CoordinatePrecision>>(*tt), std::make_shared<Cluster<CoordinatePrecision>>(*ss), std::make_shared<Cluster<CoordinatePrecision>>(*rr));
    //                 }
    //             }
    //         }
    //     }
    //     Matrix<CoefficientPrecision> gen(t->get_size(), r->get_size());
    //     DenseGenerator<CoefficientPrecision> zer(gen);
    //     auto temp = this->get_block(t->get_size(), r->get_size(), t->get_offset(), r->get_offset());
    //     Z.Plus(*temp);
    // }

    // void
    // MM(const HMatrix<CoefficientPrecision, CoordinatePrecision> *H, const HMatrix<CoefficientPrecision, CoordinatePrecision> *K, const Cluster<CoefficientPrecision> *t, const Cluster<CoefficientPrecision> *s, const Cluster<CoefficientPrecision> *r) {
    //     auto Hts = H->get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset());
    //     auto Ksr = K->get_block(s->get_size(), r->get_size(), s->get_offset(), r->get_offset());
    //     auto Ltr = this->get_block(t->get_size(), r->get_size(), t->get_offset(), r->get_offset());
    //     if ((Hts->get_children().size() > 0) and (Ksr->get_children().size() > 0) and (Ltr->get_children().size() > 0)) {
    //         for (auto &tt : t->get_children()) {
    //             for (auto &ss : s->get_children()) {
    //                 for (auto &rr : r->get_children()) {
    //                     this->MM(H, K, tt.get(), ss.get(), rr.get());
    //                 }
    //             }
    //         }
    //     } else if (Ltr->get_children().size() > 0) {
    //         // du coup Hts et Ksr dont des feuilles
    //         Matrix<CoefficientPrecision> hh(t->get_size(), s->get_size());
    //         Matrix<CoefficientPrecision> kk(s->get_size(), r->get_size());
    //         if (Hts->is_low_rank()) {
    //             hh = Hts->get_low_rank_data()->Get_U() * Hts->get_low_rank_data()->Get_V();
    //         } else if (Hts->is_dense()) {
    //             hh = *Hts->get_dense_data();
    //         }
    //         if (Ksr->is_low_rank()) {
    //             kk = Ksr->get_low_rank_data()->Get_U() * Ksr->get_low_rank_data()->Get_V();
    //         } else if (Ksr->is_dense()) {
    //             kk = *Ksr->get_dense_data();
    //         }
    //         Matrix<CoefficientPrecision> l(t->get_size(), r->get_size());
    //         copy_to_dense(*Ltr, l.data());
    //         l = l + hh * kk;
    //         this->get_block(t->get_size(), r->get_size(), t->get_offset(), r->get_offset())->set_dense_data(l);
    //     } else {
    //         this->MMR(H, K, t, s, r);
    //     }
    // }

    // void forward_substitution_extract(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) const {
    //     auto Lt = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());

    //     if (Lt->get_children().size() == 0) {
    //         auto M = *(Lt->get_dense_data());
    //         for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
    //             x[j] = y[j];
    //             for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
    //                 y[i] = y[i] - M(i - t.get_offset(), j - t.get_offset()) * x[j];
    //             }
    //         }
    //     } else {
    //         for (auto &ti : t.get_children()) {
    //             this->forward_substitution_extract(*ti, y, x);
    //             for (auto &tj : t.get_children()) {
    //                 if (tj->get_offset() > ti->get_offset()) {
    //                     auto Lij = this->get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
    //                     std::vector<CoefficientPrecision> y_temp(tj->get_size(), 0.0);
    //                     for (int k = 0; k < tj->get_size(); ++k) {
    //                         y_temp[k] = y[k + tj->get_offset()];
    //                     }
    //                     std::vector<CoefficientPrecision> yy(tj->get_size(), 0.0);
    //                     std::vector<CoefficientPrecision> x_temp(ti->get_size(), 0.0);
    //                     for (int l = 0; l < ti->get_size(); ++l) {
    //                         x_temp[l] = x[l + ti->get_offset()];
    //                     }
    //                     Lij->add_vector_product('N', 1.0, x_temp.data(), 0.0, yy.data());
    //                     for (int k = 0; k < tj->get_size(); ++k) {
    //                         y[k + tj->get_offset()] = y[k + tj->get_offset()] - yy[k];
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    //// Ly= b

    void forward_substitution_extract(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &b) const {
        auto Lt = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Lt->get_children().size() == 0) {
            auto M = (*Lt->get_dense_data());
            for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
                y[j] = b[j];
                for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
                    b[i] = b[i] - M(i - t.get_offset(), j - t.get_offset()) * y[j];
                }
            }
        } else {
            for (auto &tj : t.get_children()) {
                this->forward_substitution_extract(*tj, y, b);
                for (auto &ti : t.get_children()) {
                    if (ti->get_offset() > tj->get_offset()) {
                        auto Lij = this->get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset());
                        std::vector<CoefficientPrecision> bti(ti->get_size(), 0.0);
                        for (int k = 0; k < ti->get_size(); ++k) {
                            bti[k] = b[k + ti->get_offset()];
                        }
                        std::vector<CoefficientPrecision> ytj(tj->get_size(), 0.0);
                        for (int l = 0; l < tj->get_size(); ++l) {
                            ytj[l] = y[l + tj->get_offset()];
                        }
                        Lij->add_vector_product('N', -1.0, ytj.data(), 1.0, bti.data());

                        for (int k = 0; k < ti->get_size(); ++k) {
                            b[k + ti->get_offset()] = bti[k];
                        }
                    }
                }
            }
        }
    }

    void forward_substitution(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &Y, std::vector<CoefficientPrecision> &B) const {
        auto Lt = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        std::vector<CoefficientPrecision> y(t.get_size(), 0.0);
        std::vector<CoefficientPrecision> b(t.get_size(), 0.0);
        // std::cout << "norme B" << norm2(B) << std::endl;
        // for (int k = 0; k < t.get_size(); ++k) {
        //     std::cout << "!!!" << B[k + t.get_offset()] << std::endl;
        //     y[k] = Y[k + t.get_offset()];
        //     b[k] = B[k + t.get_offset()];
        //     std::cout << "!!" << B[k + t.get_offset()] << std::endl;
        // }
        if (Lt->get_children().size() == 0) {
            auto M = (*Lt->get_dense_data());
            for (int j = 0; j < t.get_size(); ++j) {
                y[j] = b[j];
                for (int i = j + 1; i < t.get_size(); ++i) {
                    b[i] = b[i] - M(i, j) * y[j];
                }
            }
            for (int k = 0; k < t.get_size(); ++k) {
                Y[k + t.get_offset()] = y[k];
                B[k + t.get_offset()] = b[k];
            }
        } else {
            for (auto &tj : t.get_children()) {
                this->forward_substitution_extract(*tj, Y, B);
                for (auto &ti : t.get_children()) {
                    if (ti->get_offset() > tj->get_offset()) {
                        auto Lij = this->get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset());
                        std::vector<CoefficientPrecision> bti(ti->get_size(), 0.0);
                        for (int k = 0; k < ti->get_size(); ++k) {
                            bti[k] = B[k + ti->get_offset()];
                        }
                        std::vector<CoefficientPrecision> ytj(tj->get_size(), 0.0);
                        for (int l = 0; l < tj->get_size(); ++l) {
                            ytj[l] = Y[l + tj->get_offset()];
                        }
                        Lij->add_vector_product('N', -1.0, ytj.data(), 1.0, bti.data());

                        for (int k = 0; k < ti->get_size(); ++k) {
                            B[k + ti->get_offset()] = bti[k];
                        }
                    }
                }
            }
        }
    }

    // void backward_substitution_extract(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) {
    //     auto Ut = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
    //     if (Ut->get_children().size() == 0) {
    //         auto M = *(Ut->get_dense_data());
    //         for (int jj = t.get_size() - 1; jj >= 0; --jj) {
    //             x[t.get_offset() + jj] = y[t.get_offset() + jj] / M(jj, jj);
    //             for (int i = t.get_offset(); i < t.get_offset() + jj; ++i) {
    //                 y[i] = y[i] - M(i - t.get_offset(), jj) * x[t.get_offset() + jj];
    //             }
    //         }
    //     } else {
    //         auto child = Ut->get_target_cluster().get_children();
    //         std::vector<Cluster<CoordinatePrecision> *> Child;
    //         Child.reserve(child.size());
    //         for (const auto &ptr : child) {
    //             Child.push_back(ptr.get());
    //         }
    //         for (auto &ti : child) {
    //             for (auto ti = child.rbegin(); ti != child.rend(); ++ti) {
    //                 for (int i = Child.size() - 1; i >= 0; --i) {
    //                     auto ti = Child[i];

    //                     for (auto &ti = Ut->get_target_cluster().get_children().rbegin(); ti != Ut->get_target_cluster().get_children().rend(); ++ti) {
    //                         this->backward_substitution_extract(*ti, y, x);
    //                         for (auto &tj : Ut->get_target_cluster().get_children()) {
    //                             if (tj->get_offset() < ti->get_offset()) {
    //                                 std::vector<CoefficientPrecision> ytemp(tj->get_size(), 0.0);
    //                                 for (int k = 0; k < tj->get_size(); ++k) {
    //                                     ytemp[k] = y[k + tj->get_offset()];
    //                                 }
    //                                 std::vector<CoefficientPrecision> xtemp(ti->get_size(), 0.0);
    //                                 for (int l = 0; l < ti->get_size(); ++l) {
    //                                     std::cout << "ici" << x[l + ti->get_offset()] << std::endl;
    //                                     xtemp[l] = x[l + ti->get_offset()];
    //                                 }
    //                                 std::vector<CoefficientPrecision> yy(tj->get_size(), 0.0);
    //                                 auto Uts = this->get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
    //                                 Uts->add_vector_product('N', 1.0, xtemp.data(), 0.0, yy.data());
    //                                 for (int k = 0; k < tj->get_size(); ++k) {
    //                                     y[k + tj->get_offset()] = y[k + tj->get_offset()] - yy[k];
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }

    // void
    // backward_substitution_extract(const int &szt, const int &oft, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) const {
    //     auto Ut = this->get_block(szt, szt, oft, oft);
    //     if (Ut->get_children().size() == 0) {
    //         auto M = *(Ut->get_dense_data());
    //         for (int jj = szt - 1; jj >= 0; --jj) {
    //             x[oft + jj] = y[oft + jj] / M(jj, jj);
    //             for (int i = oft; i < oft + jj; ++i) {
    //                 y[i] = y[i] - M(i - oft, jj) * x[oft + jj];
    //             }
    //         }
    //     } else {
    //         std::vector<int> sizes;
    //         std::vector<int> offsets;
    //         for (auto &c : Ut->get_target_cluster().get_children()) {
    //             int s = c->get_size();
    //             int o = c->get_offset();
    //             sizes.push_back(s);
    //             offsets.push_back(o);
    //         }
    //         for (int i = sizes.size() - 1; i >= 0; --i) {
    //             int size_i = sizes[i];
    //             int of_i   = offsets[i];
    //             // std::cout << "i =" << i << ',' << "szt = " << size_i << ',' << "ofi" << std::endl;
    //             //  for (auto &ti = Ut->get_target_cluster().get_children().rbegin(); ti != Ut->get_target_cluster().get_children().rend(); ++ti) {
    //             this->backward_substitution_extract(size_i, of_i, y, x);
    //             for (int j = 0; j < i; ++j) {
    //                 int size_j = sizes[j];
    //                 int of_j   = offsets[j];
    //                 std::vector<CoefficientPrecision> ytemp(size_j, 0.0);
    //                 for (int k = 0; k < size_j; ++k) {
    //                     ytemp[k] = y[k + of_j];
    //                 }
    //                 std::vector<CoefficientPrecision> xtemp(size_i, 0.0);
    //                 for (int l = 0; l < size_i; ++l) {
    //                     // std::cout << "ici" << x[l + ti->get_offset()] << std::endl;
    //                     xtemp[l] = x[l + of_i];
    //                 }
    //                 std::vector<CoefficientPrecision> yy(size_j, 0.0);
    //                 auto Uts = this->get_block(size_j, size_i, of_j, of_i);
    //                 Uts->add_vector_product('N', 1.0, xtemp.data(), 0.0, yy.data());
    //                 for (int k = 0; k < size_j; ++k) {
    //                     y[k + of_j] = y[k + of_j] - yy[k];
    //                 }
    //             }
    //         }
    //     }
    // }

    /// U x = y

    void backward_substitution_extract(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &x, std::vector<CoefficientPrecision> &y) const {
        auto Uk = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());

        if (Uk->get_children().size() == 0) {

            auto dense = *Uk->get_dense_data();
            for (int j = t.get_offset() + t.get_size() - 1; j > t.get_offset() - 1; --j) {
                x[j] = y[j] / dense(j - t.get_offset(), j - t.get_offset());
                for (int i = t.get_offset(); i < j; ++i) {
                    y[i] = y[i] - dense(i - t.get_offset(), j - t.get_offset()) * x[j];
                }
            }
        } else {
            for (auto tj = t.get_children().rbegin(); tj != t.get_children().rend(); ++tj) {
                auto &ttj = *tj;
                this->backward_substitution_extract(*ttj, x, y);
                for (auto &ti : t.get_children()) {
                    auto &tti = *ti;
                    if (tti.get_offset() < ttj->get_offset()) {

                        auto Uij = this->get_block(ti->get_size(), ttj->get_size(), ti->get_offset(), ttj->get_offset());
                        std::vector<CoefficientPrecision> yti(ti->get_size(), 0.0);
                        for (int k = 0; k < ti->get_size(); ++k) {
                            yti[k] = y[k + ti->get_offset()];
                        }
                        std::vector<CoefficientPrecision> xtj(ttj->get_size(), 0.0);
                        for (int k = 0; k < ttj->get_size(); ++k) {
                            xtj[k] = x[k + ttj->get_offset()];
                        }
                        Uij->add_vector_product('N', -1.0, xtj.data(), 1.0, yti.data());
                        for (int k = 0; k < ti->get_size(); ++k) {
                            y[k + ti->get_offset()] = yti[k];
                        }
                    }
                }
            }
        }
    }

    // void backward_substitution_extract(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) {
    //     auto Ut = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
    //     if (Ut->get_children().size() == 0) {
    //         auto M = *(Ut->get_dense_data());
    //         for (int jj = 0; jj < t.get_size(); ++jj) {
    //             int j = t.get_size() + t.get_offset() - jj;
    //             x[j]  = y[j] / M(j - t.get_offset(), j - t.get_offset());
    //             for (int i = t.get_offset(); i < j; ++i) {
    //                 y[i] = y[i] - M(i - t.get_offset(), j - t.get_offset()) * x[j];
    //             }
    //         }
    //     } else {
    //         for (auto &ti : Ut->get_target_cluster().get_children()) {
    //             // for (auto &ti = Ut->get_target_cluster().get_children().rbegin(); ti != Ut->get_target_cluster().get_children().rend(); ++ti) {
    //             this->backward_substitution_extract(*ti, y, x);
    //             for (auto &tj : Ut->get_target_cluster().get_children()) {
    //                 if (tj->get_offset() < ti->get_offset()) {
    //                     std::vector<CoefficientPrecision> ytemp(tj->get_size());
    //                     for (int k = 0; k < tj->get_size(); ++k) {
    //                         ytemp[k] = y[k + tj->get_offset()];
    //                     }
    //                     std::vector<CoefficientPrecision> xtemp(ti->get_size());
    //                     for (int l = 0; l < ti->get_size(); ++l) {
    //                         xtemp[l] = x[l + ti->get_size()];
    //                     }
    //                     std::vector<CoefficientPrecision> yy(tj->get_size(), 0.0);
    //                     auto Uts = this->get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
    //                     Uts->add_vector_product('N', 1.0, xtemp.data(), 0.0, yy.data());
    //                     for (int k = 0; k < tj->get_size(); ++k) {
    //                         y[k + tj->get_offset()] = ytemp[k] - yy[k];
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    // void forward_substitution_extract(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) {
    //     auto Lt = this->get_block(t, t);
    //     if (Lt->get_children().size() == 0) {
    //         auto M = *(Lt->get_dense_data());
    //         for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
    //             x[j] = y[j];
    //             for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
    //                 y[i] = y[i] - M(i, j) * y[j];
    //             }
    //         }
    //     } else {
    //         std::vector<Cluster<CoordinatePrecision>> target_son;
    //         std::vector<Cluster<CoordinatePrecision>> source_son;
    //         for (auto &child : Lt->get_target_cluster().get_children()) {
    //             target_son.push_back(*child.get());
    //         }
    // for (auto &child : Lt->get_source_cluster().get_children()) {
    //     source_son.push_back(*child);
    // }
    // auto L21 = this->get_block(target_son[1], source_son[0]);
    //  // for (auto &child : Lt->get_children()) {
    //  //     if (child->get_target_cluster() == child->get_source_cluster()) {
    //  //         this->forward_substitution_extract(child->get_target_cluster(), y, x);
    //  //     }
    //  //     if (child->get_target_cluster().get_offset() > child->get_source_cluster().get_offset()) {
    //  //         std::vector<CoefficientPrecision> y_temp(child->get_target_cluster().get_size());
    //  //         for (int k = 0; k < child->get_target_cluster().get_size(); ++k) {
    //  //             y_temp[k] = y[k + child->get_target_cluster().get_offset()];
    //  //         }
    //  //         std::vector<CoefficientPrecision> yy(child->get_target_cluster().get_size());
    //  //         std::vector<CoefficientPrecision> x_temp(child->get_source_cluster().get_size());
    //  //         for (int l = 0; l < child->get_source_cluster().get_size(); ++l) {
    //  //             x_temp[l] = x[l + child->get_source_cluster().get_size()];
    //  //         }

    // //         child->add_vector_product('N', 1.0, x_temp.data(), 0.0, yy.data());
    // //         for (int k = 0; k < child->get_target_cluster().get_size(); ++k) {
    // //             y[k + child->get_target_cluster().get_offset()] = y[k + child->get_target_cluster().get_offset()] - yy[k];
    // //         }
    // //     }
    // // }
    // this->forward_substitution_extract(target_son[0], y, x);
    // std::vector<CoefficientPrecision> y_temp(target_son[1].get_size());
    // for (int k = 0; k < target_son[1].get_size(); ++k) {
    //     y_temp[k] = y[k + target_son[1].get_offset()];
    // }
    // std::vector<CoefficientPrecision> yy(target_son[1].get_size());
    // std::vector<CoefficientPrecision> x_temp(source_son[0].get_size());
    // for (int l = 0; l < source_son[0].get_size(); ++l) {
    //     x_temp[l] = x[l + target_son[1].get_size()];
    // }

    // L21->add_vector_product('N', 1.0, x_temp.data(), 0.0, yy.data());
    // for (int k = 0; k < target_son[1].get_size(); ++k) {
    //     y[k + target_son[1].get_offset()] = y[k + target_son[1].get_offset()] - yy[k];
    // }
    // this->forward_substitution_extract(target_son[1], y, x);
    //     }
    // }

    // void forward_substitution_extract(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) {
    //     auto Lt = this->get_block(t, t);
    //     std::cout << "Lt info" << Lt->get_target_cluster().get_offset() << ',' << Lt->get_source_cluster().get_offset() << '!' << Lt->get_target_cluster().get_size() << ',' << Lt->get_source_cluster().get_size() << std::endl;
    //     if (Lt == nullptr) {
    //         std::cout << "wtf" << std::endl;
    //     }
    //     if (this->get_block(t, t)->get_children().size() == 0) {
    //         auto M = *(Lt->get_dense_data());
    //         for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
    //             x[j] = y[j];
    //             for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
    //                 y[i] = y[i] - M(i, j) * y[j];
    //             }
    //         }
    //     } else {
    //         for (auto &child_s : t.get_children()) {
    //             this->forward_substitution_extract(*child_s, y, x);
    //             for (auto &child_t : t.get_children()) {
    //                 std::cout << "cluster info " << child_t->get_offset() << ',' << child_s->get_offset() << '!' << child_t->get_size() << ',' << child_s->get_size() << std::endl;
    //                 if (child_t->get_offset() > child_s->get_offset()) {
    //                     auto Lts = this->get_block(*child_t, *child_s);
    //                     std::vector<CoefficientPrecision> y_temp(child_t->get_size());
    //                     for (int k = 0; k < child_t->get_size(); ++k) {
    //                         y_temp[k] = y[k + child_t->get_offset()];
    //                     }
    //                     std::vector<CoefficientPrecision> yy(child_t->get_size());
    //                     std::vector<CoefficientPrecision> x_temp(child_s->get_size());
    //                     for (int l = 0; l < child_s->get_size(); ++l) {
    //                         x_temp[l] = x[l + child_s->get_offset()];
    //                     }
    //                     std::cout << "info son LT" << Lt->get_children().size() << std::endl;
    //                     std::cout << "LTS info " << Lts->get_target_cluster().get_size() << ',' << Lts->get_source_cluster().get_size() << std::endl;

    //                     Lts->add_vector_product('N', 1.0, x_temp.data(), 0.0, yy.data());
    //                     for (int k = 0; k < child_t->get_size(); ++k) {
    //                         y[k + child_t->get_offset()] = y[k + child_t->get_offset()] - yy[k];
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    void
    forward_substitution_extract_T(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> y, std::vector<CoefficientPrecision> x) {
        auto Ut = this->get_block(t, t);
        if (Ut->is_leaf()) {
            auto M = Ut->get_dense_data();
            for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
                x[j] = y[j] / M(j, j);
                for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
                    y[i] = y[i] - M(j, i) * y[j];
                }
            }
        } else {
            for (auto &child_s : t.get_children()) {
                this->forward_substitution(child_s, y, x);
                for (auto &child_t : t.get_children()) {
                    if (child_t.get_offset() > child_s.get_offset()) {
                        auto Uts = this->get_block(child_t, child_s);
                        std::vector<CoefficientPrecision> y_temp(child_t->get_size());
                        for (int k = 0; k < child_t->get_size(); ++k) {
                            y_temp[k] = y[k + child_t->get_offset()];
                        }
                        std::vector<CoefficientPrecision> yy(child_t->get_size());
                        std::vector<CoefficientPrecision> x_temp(child_s->get_size());
                        for (int l = 0; l < child_s->get_size(); ++l) {
                            x_temp[l] = x[l + child_s->get_offset()];
                        }
                        Uts->add_vector_product('T', 1.0, x_temp.data(), 0.0, yy.data());
                        for (int k = 0; k < child_t->get_size(); ++k) {
                            y[k + child_t->get_offset()] = y[k + child_t->get_offset()] - yy[k];
                        }
                    }
                }
            }
        }
    }

    friend void forward_backward(const HMatrix &L, const HMatrix &U, const Cluster<CoordinatePrecision> &root, std::vector<CoefficientPrecision> &x, std::vector<CoordinatePrecision> &y) {
        std::vector<CoefficientPrecision> ux(L.get_source_cluster().get_size(), 0.0);
        L.forward_substitution_extract(root, ux, y);
        U.backward_substitution_extract(root, x, ux);
    }

    // LX =Z

    // friend void Forward_M_extact(const HMatrix &L, HMatrix &X, const HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) {
    //     auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //     if (Zts->get_children().size() == 0) {
    //         if (Zts->is_dense()) {
    //             // column wise  foward
    //             auto Xts    = X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //             auto Zdense = Zts->get_dense_data();
    //             auto Xdense = Xts->get_dense_data();
    //             for (int j = 0; j < s.get_size(), ++j) {
    //                 auto Xtj = Zdense->get_col(j);
    //                 L.forward_substitution_extract(t, Xtj, Zdense->get_col(j));
    //                 Xdense->set_col(j, Xtj);
    //             }
    //             Xts->set_dense_data()
    //         }
    //     }
    // }

    // friend void Forward_M_extract(const HMatrix &L, Hmatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &X) {
    //     auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //     auto Xts = X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //     // if t,s \in P^-
    //     if (Zts->is_dense()) {
    //         // for j\in s :  forw&ard_substitutioon (L , t , Xtj , Ztj) -> LXtj = Ztj
    //         auto z = Zts->get_dense_data();
    //         for (int j = 0; j < Zts->get_source_cluster().get_size(); ++j) {
    //             std::vector<CoefficientPrecision> Xtj(L.get_source_cluster().get_size(), 0.0);
    //             std::vector<CoefficientPrecision> Ztj(L.get_target_cluster().get_size(), 0.0);
    //         }
    //     }
    // }

    // friend void Forward_M_extract(const HMatrix &L, HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &X) {
    //     auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //     auto Xts = X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //     if (Zts->is_dense()) {
    //         auto Zdense = *Zts->get_dense_data();
    //         Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
    //         Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
    //         auto ltemp = L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
    //         copy_to_dense(*ltemp, ll.data());
    //         for (int j = 0; j < s.get_size(); ++j) {
    //             std::vector<CoefficientPrecision> Xtj(X.get_target_cluster().get_size(), 0.0);
    //             std::vector<CoefficientPrecision> Ztj(Z.get_target_cluster().get_size(), 0.0);
    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 Ztj[k + t.get_offset()] = Zdense(k, j);
    //             }
    //             L.forward_substitution_extract(t, Xtj, Ztj);
    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 Xupdate(k, j) = Xtj[k + t.get_offset()];
    //             }
    //             std::vector<CoefficientPrecision> xrestr(t.get_size());
    //             auto zrestr = Zdense.get_col(j);

    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 xrestr[k] = Xtj[k + t.get_offset()];
    //             }
    //             // for (int k = 0; k < t.get_size(); ++k) {
    //             //     Zdense(k, j) = Ztj[k + t.get_offset()];
    //             // }
    //         }

    //         MatrixGenerator<CoefficientPrecision> mat(Xupdate, t.get_offset(), s.get_offset());
    //         Xts->compute_dense_data(mat);
    //         // MatrixGenerator<CoefficientPrecision> matz(Zdense, t.get_offset(), s.get_offset());

    //         // Zts->compute_dense_data(matz);

    //         std::cout << t.get_size() << ',' << t.get_offset() << ',' << s.get_size() << ',' << s.get_offset() << std::endl;
    //         std::cout << "0 = ? " << normFrob(ll * Xupdate - Zdense) / normFrob(Zdense) << std::endl;
    //         std::cout << "0 = ! " << normFrob(ll * *X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data() - Zdense) / normFrob(Zdense) << std::endl;

    //     } else if (Zts->is_low_rank()) {
    //         std::cout << "héhé " << std::endl;
    //         auto Zlr = Zts->get_low_rank_data();
    //         auto U   = Zlr->Get_U();
    //         auto V   = Zlr->Get_V();
    //         Matrix<CoefficientPrecision> Xu(t.get_size(), Zlr->rank_of());
    //         for (int j = 0; j < Zlr->rank_of(); ++j) {
    //             std::vector<CoefficientPrecision> Uj(Z.get_target_cluster().get_size());
    //             std::vector<CoefficientPrecision> Aj(L.get_target_cluster().get_size());
    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 Uj[k + t.get_offset()] = U(k, j);
    //             }
    //             L.forward_substitution_extract(t, Aj, Uj);
    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 Xu(k, j) = Aj[k + t.get_offset()];
    //             }
    //         }
    //         LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Xu, V);
    //         Xts->set_low_rank_data(Xlr);
    //     } else {
    //         for (auto &child_t : t.get_children()) {
    //             for (auto &child_s : s.get_children()) {
    //                 Forward_M_extract(L, Z, *child_t, *child_s, X);

    //                 for (auto &child_tt : t.get_children()) {
    //                     if (child_tt->get_offset() > child_t->get_offset()) {
    //                         auto Ztemp = Z.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
    //                         auto Ltemp = L.get_block(child_tt->get_size(), child_t->get_size(), child_tt->get_offset(), child_t->get_offset());

    //                         auto Xtemp = X.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
    //                         Matrix<CoefficientPrecision> ref(child_tt->get_size(), child_s->get_size());
    //                         auto temp = Ltemp->hmatrix_product(*Xtemp);

    //                         copy_to_dense(temp, ref.data());
    //                         Matrix<CoefficientPrecision> zz(Ztemp->get_target_cluster().get_size(), Ztemp->get_source_cluster().get_size());
    //                         Matrix<CoefficientPrecision> ll(Ltemp->get_target_cluster().get_size(), Ltemp->get_source_cluster().get_size());
    //                         copy_to_dense(*Ltemp, ll.data());
    //                         copy_to_dense(*Ztemp, zz.data());
    //                         std::cout << "____________________" << std::endl;
    //                         std::cout << normFrob(zz - ll * zz) << std::endl;
    //                         Ztemp->Moins(temp);
    //                         copy_to_dense(*Z.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset()), zz.data());
    //                         std::cout << normFrob(zz) << std::endl;
    //                         std::cout << "____________________" << std::endl;
    //                         // *Ztemp = *Ztemp - Ltemp->hmatrix_product(*Xtemp);
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // std::cout << "alors " << std::endl;
    //     // Matrix<CoefficientPrecision> xx(X.get_target_cluster().get_size(), X.get_source_cluster().get_size());
    //     // Matrix<CoefficientPrecision> ll(L.get_target_cluster().get_size(), L.get_source_cluster().get_size());
    //     // Matrix<CoefficientPrecision> zz(Z.get_target_cluster().get_size(), Z.get_source_cluster().get_size());
    //     // copy_to_dense(L, ll.data());
    //     // copy_to_dense(X, xx.data());
    //     // copy_to_dense(Z, zz.data());
    //     // std::cout << normFrob(ll * xx -)
    // }

    void to_dense(Matrix<CoefficientPrecision> M) {
        if (this->get_children().size() > 0) {
            for (auto &child : this->get_children()) {
                child->to_dense(M);
            }
        } else {
            auto ptr = this->get_dense_data();
            if (ptr != nullptr) {
                auto mat = *ptr;
                for (int k = 0; k < this->get_target_cluster().get_size(); ++k) {
                    for (int l = 0; l < this->get_source_cluster().get_size(); ++l) {
                        M(k + this->get_target_cluster().get_offset(), l + this->get_source_cluster().get_offset()) = mat(k, l);
                    }
                }
            } else {
                auto mat = this->get_low_rank_data()->Get_U() * this->get_low_rank_data()->Get_V();
                for (int k = 0; k < this->get_target_cluster().get_size(); ++k) {
                    for (int l = 0; l < this->get_source_cluster().get_size(); ++l) {
                        M(k + this->get_target_cluster().get_offset(), l + this->get_source_cluster().get_size()) = mat(k, l);
                    }
                }
            }
        }
    }
    friend void forward_Daquin(const HMatrix &L, HMatrix &X, HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) {
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        if (Zts->is_hierarchical()) {
            auto &t1 = *t.get_children()[0];
            auto &t2 = *t.get_children()[1];
            auto &s1 = *s.get_children()[0];
            auto &s2 = *s.get_children()[1];
            forward_Daquin(L, X, Z, t1, s1);
            forward_Daquin(L, X, Z, t1, s2);
            Z.get_block(t2.get_size(), s1.get_size(), t2.get_offset(), s1.get_offset())->Moins(L.get_block(t2.get_size(), s1.get_size(), t2.get_offset(), s1.get_offset())->hmatrix_product(*X.get_block(t2.get_size(), s1.get_size(), t2.get_offset(), s1.get_offset())));
            forward_Daquin(L, X, Z, t2, s1);
            Z.get_block(t2.get_size(), s2.get_size(), t2.get_offset(), s2.get_offset())->Moins(L.get_block(t2.get_size(), s2.get_size(), t2.get_offset(), s2.get_offset())->hmatrix_product(*X.get_block(t2.get_size(), s2.get_size(), t2.get_offset(), s2.get_offset())));
            forward_Daquin(L, X, Z, t2, s2);

        } else {
            auto Xts = X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
            if (Zts->is_dense()) {
                auto Zdense = *Zts->get_dense_data();
                Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
                Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
                auto ltemp = L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
                copy_to_dense(*ltemp, ll.data());
                for (int j = 0; j < s.get_size(); ++j) {
                    std::vector<CoefficientPrecision> Xtj(X.get_target_cluster().get_size(), 0.0);
                    std::vector<CoefficientPrecision> Ztj(Z.get_target_cluster().get_size(), 0.0);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Ztj[k + t.get_offset()] = Zdense(k, j);
                    }
                    L.forward_substitution_extract(t, Xtj, Ztj);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Xupdate(k, j) = Xtj[k + t.get_offset()];
                    }
                    std::vector<CoefficientPrecision> xrestr(t.get_size());
                    auto zrestr = Zdense.get_col(j);
                }
                MatrixGenerator<CoefficientPrecision> mat(Xupdate, t.get_offset(), s.get_offset());
                Xts->compute_dense_data(mat);
            } else if (Zts->is_low_rank()) {
                auto Zlr = Zts->get_low_rank_data();
                auto U   = Zlr->Get_U();
                auto V   = Zlr->Get_V();
                Matrix<CoefficientPrecision> Xu(t.get_size(), Zlr->rank_of());
                for (int j = 0; j < Zlr->rank_of(); ++j) {
                    std::vector<CoefficientPrecision> Uj(Z.get_target_cluster().get_size());
                    std::vector<CoefficientPrecision> Aj(L.get_target_cluster().get_size());
                    for (int k = 0; k < t.get_size(); ++k) {
                        Uj[k + t.get_offset()] = U(k, j);
                    }
                    L.forward_substitution_extract(t, Aj, Uj);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Xu(k, j) = Aj[k + t.get_offset()];
                    }
                }
                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Xu, V);
                Xts->set_low_rank_data(Xlr);
            }
        }
    }

    void forward_substitution_this(const HMatrix &L, HMatrix &Z, Matrix<CoefficientPrecision> z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, Matrix<CoefficientPrecision> &X) {
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());

        if (Zts->is_dense()) {
            Matrix<CoefficientPrecision> xdense(X.nb_rows(), X.nb_cols());
            // copy_to_dense(*this, xdense.data());
            auto Zdense = *Zts->get_dense_data();
            Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
            auto ltemp = L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
            copy_to_dense(*ltemp, ll.data());
            Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
            // std::cout << "avant la boucle : " << normFrob(X - xdense) / normFrob(X) << std::endl;

            for (int j = 0; j < s.get_size(); ++j) {
                std::vector<CoefficientPrecision> Xtj(L.get_source_cluster().get_size(), 0.0);
                std::vector<CoefficientPrecision> Ztj(Z.get_target_cluster().get_size(), 0.0);
                for (int k = 0; k < t.get_size(); ++k) {
                    Ztj[k + t.get_offset()] = z(k + t.get_offset(), j + s.get_offset());
                    // Ztj[k + t.get_offset()] = Zdense(k, j);
                }
                L.forward_substitution_extract(t, Xtj, Ztj);
                for (int k = 0; k < t.get_size(); ++k) {
                    X(k + t.get_offset(), j + s.get_offset()) = Xtj[k + t.get_offset()];
                    Xupdate(k, j)                             = Xtj[k + t.get_offset()];
                }
            }
            // std::cout << "aprés la boucle : " << normFrob(X - xdense) / normFrob(X) << std::endl;
            MatrixGenerator<CoefficientPrecision> gen(Xupdate, t.get_offset(), s.get_offset());
            // std::cout << "avant compute : " << normFrob(xdense) << std::endl;
            this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->compute_dense_data(gen);

            // // copy_to_dense(XX, xdense.data());
            // std::cout << "aprés compute : " << normFrob(xdense) << std::endl;
            // std::cout << "alors qu'on a rajourté une feuille de norme " << normFrob(*this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data()) << std::endl;

            /////test/////
            // std::cout << " erreur sur le blocs rajouté :" << normFrob((*this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data()) - Xupdate) / normFrob(Xupdate) << std::endl;
            // std::cout << "norme de la feuille " << normFrob((*this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data())) << ',' << normFrob(Xupdate) << std::endl;
            // auto Xtemp = this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
            // std::cout << "  get block ? " << Xtemp->get_target_cluster().get_size() << ',' << Xtemp->get_source_cluster().get_size() << ',' << Xtemp->get_target_cluster().get_offset() << ',' << Xtemp->get_source_cluster().get_offset() << std::endl;
            // std::cout << "   alors que " << t.get_size() << ',' << s.get_size() << ',' << t.get_offset() << ',' << s.get_offset() << std::endl;
            // Matrix<CoefficientPrecision> xxx(X.nb_rows(), X.nb_cols());
            // copy_to_dense(XX, xxx.data());
            // std::cout << "    erreur sur X : " << normFrob(X - xxx) / normFrob(X) << std::endl;
            // std::cout << "     wtf : " << normFrob(xxx) << ',' << normFrob(X) << std::endl;
            // // std::cout << "compute dense avant - aprés : " << normFrob(xxx - xdense) << "alors que apré =" << normFrob(xxx) << " , avant = " << normFrob(xdense) << " et on a rajouté " << normFrob((*this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data()) - Xupdate) << std::endl;
            // std::cout << "norme de la feuille " << normFrob(*this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data()) << std::endl;
        } else {
            for (int i = 0; i < t.get_children().size(); ++i) {
                auto &child_t = t.get_children()[i];
                for (auto &child_s : s.get_children()) {
                    // std::cout << "avant forward " << std::endl;
                    // Matrix<CoefficientPrecision> t1(X.nb_rows(), X.nb_cols());
                    // Matrix<CoefficientPrecision> t2(z.nb_rows(), z.nb_cols());
                    // copy_to_dense(XX, t1.data());
                    // copy_to_dense(Z, t2.data());
                    // std::cout << " X : " << normFrob(t1 - X) / normFrob(X) << " , Z : " << normFrob(z - t2) / normFrob(z) << std::endl;
                    this->forward_substitution_this(L, Z, z, *child_t, *child_s, X);
                    // std::cout << "aprés forward " << std::endl;
                    // Matrix<CoefficientPrecision> t11(X.nb_rows(), X.nb_cols());
                    // Matrix<CoefficientPrecision> t22(z.nb_rows(), z.nb_cols());
                    // copy_to_dense(XX, t11.data());
                    // copy_to_dense(Z, t22.data());
                    // std::cout << " X : " << normFrob(t11 - X) / normFrob(X) << " , Z : " << normFrob(z - t22) / normFrob(z) << std::endl;
                    for (int j = i + 1; j < t.get_children().size(); ++j) {
                        auto &child_tt = t.get_children()[j];
                        auto Ztemp     = Z.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
                        auto Ltemp     = L.get_block(child_t->get_size(), child_tt->get_size(), child_t->get_offset(), child_tt->get_offset());
                        auto Xtemp     = this->get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
                        Matrix<CoefficientPrecision> ll(Ltemp->get_target_cluster().get_size(), Ltemp->get_source_cluster().get_size());
                        copy_to_dense(*Ltemp, ll.data());
                        Matrix<CoefficientPrecision> xx(child_tt->get_size(), child_s->get_size());
                        for (int k = 0; k < child_tt->get_size(); ++k) {
                            for (int l = 0; l < child_s->get_size(); ++l) {
                                xx(k, l) = X(k + child_tt->get_offset(), l + child_s->get_offset());
                            }
                        }
                        auto M = ll * xx;
                        for (int k = 0; k < child_t->get_size(); ++k) {
                            for (int l = 0; l < child_s->get_size(); ++l) {
                                z(k + child_t->get_offset(), l + child_s->get_offset()) = z(k + child_t->get_offset(), l + child_s->get_offset()) - M(k, l);
                            }
                        }
                        // auto l = Ltemp->hmatrix_product(*Xtemp);
                        // Matrix<CoefficientPrecision> ltest(child_t->get_size(), child_s->get_size());
                        // copy_to_dense(l, ltest.data());
                        // std::cout << "erreur L* x :" << normFrob(ltest - M) << normFrob(M) << std::endl;
                        Ztemp->Moins(Ltemp->hmatrix_product(*Xtemp));
                        // // tests//
                        Matrix<CoefficientPrecision> ztest(Z.get_target_cluster().get_size(), Z.get_source_cluster().get_size());
                        copy_to_dense(Z, ztest.data());
                        std::cout << "erreur sur z : " << normFrob(z - ztest) / normFrob(z) << std::endl;
                    }
                }
            }
        }
    }

    friend void forward_substitution_dense(const HMatrix &L, HMatrix &Z, Matrix<CoefficientPrecision> z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, Matrix<CoefficientPrecision> &X, HMatrix &XX) {
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());

        if (Zts->is_dense()) {
            Matrix<CoefficientPrecision> xdense(X.nb_rows(), X.nb_cols());
            copy_to_dense(XX, xdense.data());
            auto Zdense = *Zts->get_dense_data();
            Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
            auto ltemp = L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
            copy_to_dense(*ltemp, ll.data());
            Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
            std::cout << "avant la boucle : " << normFrob(X - xdense) / normFrob(X) << std::endl;

            for (int j = 0; j < s.get_size(); ++j) {
                std::vector<CoefficientPrecision> Xtj(L.get_source_cluster().get_size(), 0.0);
                std::vector<CoefficientPrecision> Ztj(Z.get_target_cluster().get_size(), 0.0);
                for (int k = 0; k < t.get_size(); ++k) {
                    Ztj[k + t.get_offset()] = z(k + t.get_offset(), j + s.get_offset());
                    // Ztj[k + t.get_offset()] = Zdense(k, j);
                }
                L.forward_substitution_extract(t, Xtj, Ztj);
                for (int k = 0; k < t.get_size(); ++k) {
                    X(k + t.get_offset(), j + s.get_offset()) = Xtj[k + t.get_offset()];
                    Xupdate(k, j)                             = Xtj[k + t.get_offset()];
                }
            }
            std::cout << "aprés la boucle : " << normFrob(X - xdense) / normFrob(X) << std::endl;
            MatrixGenerator<CoefficientPrecision> gen(Xupdate, t.get_offset(), s.get_offset());
            std::cout << "avant compute : " << normFrob(xdense) << std::endl;
            // XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->compute_dense_data(gen);
            XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xupdate);
            copy_to_dense(XX, xdense.data());
            std::cout << "aprés compute : " << normFrob(xdense) << std::endl;
            std::cout << "alors qu'on a rajourté une feuille de norme " << normFrob(*XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data()) << std::endl;

            /////test/////
            std::cout << " erreur sur le blocs rajouté :" << normFrob((*XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data()) - Xupdate) / normFrob(Xupdate) << std::endl;
            std::cout << "norme de la feuille " << normFrob((*XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data())) << ',' << normFrob(Xupdate) << std::endl;
            auto Xtemp = XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
            std::cout << "  get block ? " << Xtemp->get_target_cluster().get_size() << ',' << Xtemp->get_source_cluster().get_size() << ',' << Xtemp->get_target_cluster().get_offset() << ',' << Xtemp->get_source_cluster().get_offset() << std::endl;
            std::cout << "   alors que " << t.get_size() << ',' << s.get_size() << ',' << t.get_offset() << ',' << s.get_offset() << std::endl;
            Matrix<CoefficientPrecision> xxx(X.nb_rows(), X.nb_cols());
            copy_to_dense(XX, xxx.data());
            std::cout << "    erreur sur X : " << normFrob(X - xxx) / normFrob(X) << std::endl;
            std::cout << "     wtf : " << normFrob(xxx) << ',' << normFrob(X) << std::endl;
            // std::cout << "compute dense avant - aprés : " << normFrob(xxx - xdense) << "alors que apré =" << normFrob(xxx) << " , avant = " << normFrob(xdense) << " et on a rajouté " << normFrob((*XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data()) - Xupdate) << std::endl;
            std::cout << "norme de la feuille " << normFrob(*XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data()) << std::endl;
        } else {
            for (int i = 0; i < t.get_children().size(); ++i) {
                auto &child_t = t.get_children()[i];
                for (auto &child_s : s.get_children()) {
                    // std::cout << "avant forward " << std::endl;
                    // Matrix<CoefficientPrecision> t1(X.nb_rows(), X.nb_cols());
                    // Matrix<CoefficientPrecision> t2(z.nb_rows(), z.nb_cols());
                    // copy_to_dense(XX, t1.data());
                    // copy_to_dense(Z, t2.data());
                    // std::cout << " X : " << normFrob(t1 - X) / normFrob(X) << " , Z : " << normFrob(z - t2) / normFrob(z) << std::endl;
                    forward_substitution_dense(L, Z, z, *child_t, *child_s, X, XX);
                    // std::cout << "aprés forward " << std::endl;
                    // Matrix<CoefficientPrecision> t11(X.nb_rows(), X.nb_cols());
                    // Matrix<CoefficientPrecision> t22(z.nb_rows(), z.nb_cols());
                    // copy_to_dense(XX, t11.data());
                    // copy_to_dense(Z, t22.data());
                    // std::cout << " X : " << normFrob(t11 - X) / normFrob(X) << " , Z : " << normFrob(z - t22) / normFrob(z) << std::endl;
                    for (int j = i + 1; j < t.get_children().size(); ++j) {
                        auto &child_tt = t.get_children()[j];
                        auto Ztemp     = Z.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
                        auto Ltemp     = L.get_block(child_t->get_size(), child_tt->get_size(), child_t->get_offset(), child_tt->get_offset());
                        auto Xtemp     = XX.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
                        Matrix<CoefficientPrecision> ll(Ltemp->get_target_cluster().get_size(), Ltemp->get_source_cluster().get_size());
                        copy_to_dense(*Ltemp, ll.data());
                        Matrix<CoefficientPrecision> xx(child_tt->get_size(), child_s->get_size());
                        for (int k = 0; k < child_tt->get_size(); ++k) {
                            for (int l = 0; l < child_s->get_size(); ++l) {
                                xx(k, l) = X(k + child_tt->get_offset(), l + child_s->get_offset());
                            }
                        }
                        auto M = ll * xx;
                        for (int k = 0; k < child_t->get_size(); ++k) {
                            for (int l = 0; l < child_s->get_size(); ++l) {
                                z(k + child_t->get_offset(), l + child_s->get_offset()) = z(k + child_t->get_offset(), l + child_s->get_offset()) - M(k, l);
                            }
                        }
                        // auto l = Ltemp->hmatrix_product(*Xtemp);
                        // Matrix<CoefficientPrecision> ltest(child_t->get_size(), child_s->get_size());
                        // copy_to_dense(l, ltest.data());
                        // std::cout << "erreur L* x :" << normFrob(ltest - M) << normFrob(M) << std::endl;
                        Ztemp->Moins(Ltemp->hmatrix_product(*Xtemp));
                        // // tests//
                        Matrix<CoefficientPrecision> ztest(Z.get_target_cluster().get_size(), Z.get_source_cluster().get_size());
                        copy_to_dense(Z, ztest.data());
                        std::cout << "erreur sur z : " << normFrob(z - ztest) / normFrob(z) << std::endl;
                    }
                }
            }
        }
    }

    friend void Forward_M_extract(const HMatrix &L, HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &X) {
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        auto Xts = X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        if (Zts->is_dense()) {
            auto Zdense = *Zts->get_dense_data();
            Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
            Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
            auto ltemp = L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
            copy_to_dense(*ltemp, ll.data());
            for (int j = 0; j < s.get_size(); ++j) {
                std::vector<CoefficientPrecision> Xtj(X.get_target_cluster().get_size(), 0.0);
                std::vector<CoefficientPrecision> Ztj(Z.get_target_cluster().get_size(), 0.0);
                for (int k = 0; k < t.get_size(); ++k) {
                    Ztj[k + t.get_offset()] = Zdense(k, j);
                }
                L.forward_substitution_extract(t, Xtj, Ztj);
                for (int k = 0; k < t.get_size(); ++k) {
                    Xupdate(k, j) = Xtj[k + t.get_offset()];
                }
                std::vector<CoefficientPrecision> xrestr(t.get_size());
                auto zrestr = Zdense.get_col(j);

                // for (int k = 0; k < t.get_size(); ++k) {
                //     Zdense(k, j) = Ztj[k + t.get_offset()];
                // }
            }

            MatrixGenerator<CoefficientPrecision> mat(Xupdate, t.get_offset(), s.get_offset());
            Xts->compute_dense_data(mat);
            // MatrixGenerator<CoefficientPrecision> matz(Zdense, t.get_offset(), s.get_offset());

            // Zts->compute_dense_data(matz);

            std::cout << t.get_size() << ',' << t.get_offset() << ',' << s.get_size() << ',' << s.get_offset() << std::endl;
            std::cout << "0 = ? " << normFrob(ll * Xupdate - Zdense) / normFrob(Zdense) << std::endl;
            std::cout << "0 = ! " << normFrob(ll * *X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data() - Zdense) / normFrob(Zdense) << std::endl;

        } else if (Zts->is_low_rank()) {
            std::cout << "héhé " << std::endl;
            auto Zlr = Zts->get_low_rank_data();
            auto U   = Zlr->Get_U();
            auto V   = Zlr->Get_V();
            Matrix<CoefficientPrecision> Xu(t.get_size(), Zlr->rank_of());
            for (int j = 0; j < Zlr->rank_of(); ++j) {
                std::vector<CoefficientPrecision> Uj(Z.get_target_cluster().get_size());
                std::vector<CoefficientPrecision> Aj(L.get_target_cluster().get_size());
                for (int k = 0; k < t.get_size(); ++k) {
                    Uj[k + t.get_offset()] = U(k, j);
                }
                L.forward_substitution_extract(t, Aj, Uj);
                for (int k = 0; k < t.get_size(); ++k) {
                    Xu(k, j) = Aj[k + t.get_offset()];
                }
            }
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Xu, V);
            Xts->set_low_rank_data(Xlr);
        } else {
            for (int i = 0; i < t.get_children().size(); ++i) {
                auto &child_t = t.get_children()[i];
                for (auto &child_s : s.get_children()) {
                    Forward_M_extract(L, Z, *child_t, *child_s, X);

                    for (int j = i + 1; j < t.get_children().size(); ++j) {
                        auto &child_tt = t.get_children()[j];
                        auto Ztemp     = Z.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
                        auto Ltemp     = L.get_block(child_t->get_size(), child_tt->get_size(), child_t->get_offset(), child_tt->get_offset());

                        auto Xtemp = X.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
                        Matrix<CoefficientPrecision> ref(child_t->get_size(), child_s->get_size());
                        auto temp = Ltemp->hmatrix_product(*Xtemp);

                        copy_to_dense(temp, ref.data());
                        Matrix<CoefficientPrecision> zz(Ztemp->get_target_cluster().get_size(), Ztemp->get_source_cluster().get_size());
                        Matrix<CoefficientPrecision> ll(Ltemp->get_target_cluster().get_size(), Ltemp->get_source_cluster().get_size());
                        copy_to_dense(*Ltemp, ll.data());
                        copy_to_dense(*Ztemp, zz.data());
                        std::cout << "____________________" << std::endl;
                        std::cout << normFrob(zz - ll * zz) << std::endl;
                        Ztemp->Moins(temp);
                        copy_to_dense(*Z.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset()), zz.data());
                        std::cout << normFrob(zz) << std::endl;
                        std::cout << "____________________" << std::endl;
                    }
                }
            }
        }
    }

    friend void Forward_M(const HMatrix &L, HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &X) {
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        auto Xts = X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        if (Zts->is_dense()) {
            auto Zdense = *Zts->get_dense_data();
            Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
            Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
            auto ltemp = L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
            copy_to_dense(*ltemp, ll.data());
            for (int j = 0; j < s.get_size(); ++j) {
                std::vector<CoefficientPrecision> Xtj(X.get_target_cluster().get_size(), 0.0);
                std::vector<CoefficientPrecision> Ztj(Z.get_target_cluster().get_size(), 0.0);
                for (int k = 0; k < t.get_size(); ++k) {
                    Ztj[k + t.get_offset()] = Zdense(k, j);
                }
                L.forward_substitution(t, Xtj, Ztj);
                for (int k = 0; k < t.get_size(); ++k) {
                    Xupdate(k, j) = Xtj[k + t.get_offset()];
                }
                std::vector<CoefficientPrecision> xrestr(s.get_size());
                auto zrestr = Zdense.get_col(j);

                for (int k = 0; k < s.get_size(); ++k) {
                    xrestr[k] = Xtj[k + s.get_offset()];
                }
            }

            MatrixGenerator<CoefficientPrecision> mat(Xupdate);
            Xts->compute_dense_data(mat);

            std::cout << t.get_size() << ',' << t.get_offset() << ',' << s.get_size() << ',' << s.get_offset() << std::endl;
            std::cout << "0 = ? " << normFrob(ll * Xupdate - Zdense) / normFrob(Zdense) << std::endl;

        } else if (Zts->is_low_rank()) {
            std::cout << "héhé " << std::endl;
            auto Zlr = Zts->get_low_rank_data();
            auto U   = Zlr->Get_U();
            auto V   = Zlr->Get_V();
            Matrix<CoefficientPrecision> Xu(t.get_size(), Zlr->rank_of());
            for (int j = 0; j < Zlr->rank_of(); ++j) {
                std::vector<CoefficientPrecision> Uj(Z.get_target_cluster().get_size());
                std::vector<CoefficientPrecision> Aj(L.get_target_cluster().get_size());
                for (int k = 0; k < t.get_size(); ++k) {
                    Uj[k + t.get_offset()] = U(k, j);
                }
                L.forward_substitution(t, Aj, Uj);
                for (int k = 0; k < t.get_size(); ++k) {
                    Xu(k, j) = Aj[k + t.get_offset()];
                }
            }
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Xu, V);
            Xts->set_low_rank_data(Xlr);
        } else {
            for (auto &child_t : t.get_children()) {
                for (auto &child_s : s.get_children()) {
                    Forward_M(L, Z, *child_t, *child_s, X);

                    for (auto &child_tt : t.get_children()) {
                        if (child_tt->get_offset() > child_t->get_offset()) {
                            auto Ztemp = Z.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
                            auto Ltemp = L.get_block(child_tt->get_size(), child_t->get_size(), child_tt->get_offset(), child_t->get_offset());

                            auto Xtemp = X.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
                            Matrix<CoefficientPrecision> ref(child_tt->get_size(), child_s->get_size());
                            auto temp = Ltemp->hmatrix_product(*Xtemp);

                            copy_to_dense(temp, ref.data());
                            Matrix<CoefficientPrecision> zz(Ztemp->get_target_cluster().get_size(), Ztemp->get_source_cluster().get_size());
                            copy_to_dense(*Ztemp, zz.data());
                            std::cout << "____________________" << std::endl;
                            std::cout << normFrob(zz - ref) << std::endl;
                            Ztemp->Moins(temp);
                            copy_to_dense(*Ztemp, zz.data());
                            std::cout << normFrob(zz) << std::endl;
                            std::cout << "____________________" << std::endl;
                            // *Ztemp = *Ztemp - Ltemp->hmatrix_product(*Xtemp);
                        }
                    }
                }
            }
        }
    }

    // X U = Z
    friend void ForwardT_M_extract(const HMatrix &U, const HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &X) {
        auto Zts = Z.get_block(t, s);
        auto Xts = X.get_block(t, s);
        if (Zts->is_dense()) {
            auto Zdense = Zts->get_dense_data();
            auto Xdense = Xts->get_dense_data();
            Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
            for (int j = 0; j < t.get_size(); ++j) {
                std::vector<CoefficientPrecision> Xtj(s.get_size());
                std::vector<CoefficientPrecision> Ztj(s.get_size());
                for (int k = 0; k < s.get_size(); ++k) {
                    Xtj[k] = Xdense(j, k);
                    Ztj[k] = Zdense(j, k);
                }
                U.forward_substitution_extractT(t, Ztj, Xtj);
                for (int k = 0; k < s.get_size(); ++k) {
                    Xupdate(j, k) = Xtj[k];
                }
            }
            MatrixGenerator<CoefficientPrecision> mat(Xupdate);
            Xts->compute_dense_data(mat);
        } else if (Zts->is_low_rank()) {
            auto Zlr = Zts.get_low_rank_data();
            auto Uz  = Zlr->Get_U();
            auto Vz  = Zlr->Get_V();
            Matrix<CoefficientPrecision> Xv(Zlr->get_rank_of(), s.get_size());
            for (int j = 0; j < Zlr->get_rank_of(); ++j) {
                std::vector<CoefficientPrecision> Vj(s.get_size());
                std::vector<CoefficientPrecision> Aj(s.get_size());
                for (int k = 0; k < s.get_size(); ++k) {
                    Vj[k] = Vz(j, k);
                }
                U.forward_substitution_extractT(t, Vj, Aj);
                for (int k = 0; k < t.get_size(); ++k) {
                    Xv(j, k) = Aj[k];
                }
            }
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(U, Xv);
            Xts->set_low_rank_data(Xlr);
        } else {
            for (auto &child_t : t.get_children()) {
                for (auto &child_s : s.get_children()) {
                    Forward_M_extractT(U, Z, child_t, child_s, X);

                    for (auto &child_tt : t.get_children()) {
                        if (child_tt->get_offset() > child_t->get_offset()) {
                            auto Ztemp = Z.get_block(child_tt, child_s);
                            auto Utemp = U.get_block(child_t, child_tt);
                            auto Xtemp = X.get_block(child_t, child_s);
                            *Ztemp     = *Ztemp - Utemp->hmatrix_product(Xtemp);
                        }
                    }
                }
            }
        }
    }

    void friend H_LU(const HMatrix &M, HMatrix &L, HMatrix &U, const Cluster<CoordinatePrecision> &t) {
        auto Mt = M.get_block(t, t);
        if (Mt->is_leaf()) {
            auto m = Mt->get_dense_data();
            // l, u = lu(m) , L.get_block(t,t).set_danse_data(l)
        } else {
            for (auto &child_t : t.get_children()) {
                H_LU(M, L, U, child_t);
                for (auto &child_tt : t.get_children()) {
                    if (child_tt->get_offset() > child_t.get_offset()) {
                        ForwardT_M_extract(U, L, M, child_tt, child_t);
                        Forward_M_extract(L, U, M, child_t, child_tt);
                        for (auto &child : t.get_children()) {
                            if (child->get_offset() > child_t.get_offset()) {
                                auto Mts = M.get_block(child_tt, child);
                                auto Lts = L.get_block(child_tt, child_t);
                                auto Uts = U.get_block(child_t, child);
                                *Mts     = *Mts - Lts.hmatrix_product(Uts);
                            }
                        }
                    }
                }
            }
        }
    }

    // void forward_substitution(std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) {

    //     if (this->is_leaf()) {
    //         auto M = this->get_dense_data();
    //         for (int j = this->get_target_cluster().get_offset(); j < this->get_target_cluster().get_size() + this->get_target_cluster().get_offset(); ++j) {
    //             x[j] = y[j];
    //             for (int i = j + 1; j < this->get_target_cluster().get_size() + this->get_source_cluster().get_offset(); ++i) {
    //                 y[i] = y[i] - M(i, j) * x[j];
    //             }
    //         }
    //     } else {
    //         auto child = this->get_children();
    //         for (int k = 0; k < child.size(); ++k) {
    //             auto &hk = child[k];
    //             if (hk->get_target_cluster() == hk->get_source_cluster()) {
    //                 hk->forward_substitution(y, x);
    //             } else if (hk->get_target_cluster().get_offset() > hk->get_source_cluster().get_offset()) {
    //                 std::vector<CoefficientPrecision> y_restr(hk->get_target_cluster().get_size());
    //                 std::vector<CoefficientPrecision> x_restr(hk->get_source_cluster().get_size());
    //                 for (int i = 0; i < hk->get_source_cluster().get_size(); ++i) {
    //                     x_restr[i] = x[i + hk->get_source_cluster().get_offset()];
    //                 }
    //                 for (int j = 0; j < hk->get_target_cluster().get_size(); ++j) {
    //                     y_restr[j] = y[j + hk->get_target_cluster().get_offset()];
    //                 }
    //                 y_restr = y_restr - hk->add_vector_product('N', 1.0, x_restr.data(), 0.0, y_restr.data());

    //                 for (int j = 0; j < hk->get_target_cluster().get_size(); ++j) {
    //                     y[j + hk->get_target_cluster().get_offset()] = y_restr[j];
    //                 }
    //             }
    //         }
    //     }
    // }

    // // void backward_substitution(const Cluster t, std::vectort<CoefficientPecision>)
    // void forward_substitution(const Cluster &tau, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) {
    //     auto M = this->get_dense_data();
    //     if (this->is_leaf()) {
    //         for (int j = this->get_target_cluster().get_offset(); j < this->get_target_cluster().get_size() + this->get_target_cluster().get_offset(); ++j) {
    //             x[j] = y[j];
    //             for (int i = j + 1; j < this->get_target_cluster().get_size() + this->get_source_cluster().get_offset(); ++i) {
    //                 y[i] = y[i] - M(i, j) * x[j];
    //             }
    //         }
    //     } else {
    //         auto child = this->get_children();
    //         for (int k = 0; k < child.size(); ++k) {
    //             auto &hk = child[k];
    //             if (hk->get_target_cluster() == hk->get_source_cluster()) {
    //                 hk->forward_substitution(y, x);
    //             } else if (hk->get_target_cluster().get_offset() > hk->get_source_cluster().get_offset()) {
    //                 std::vector<CoefficientPrecision> y_restr(hk->get_target_cluster().get_size());
    //                 std::vector<CoefficientPrecision> x_restr(hk->get_source_cluster().get_size());
    //                 for (int i = 0; i < hk->get_source_cluster().get_size(); ++i) {
    //                     x_restr[i] = x[i + hk->get_source_cluster().get_offset()];
    //                 }
    //                 for (int j = 0; j < hk->get_target_cluster().get_size(); ++j) {
    //                     y_restr[j] = y[j + hk->get_target_cluster().get_offset()];
    //                 }
    //                 y_restr = y_restr - hk->add_vector_product('N', 1.0, x_restr.data(), 0.0, y_restr.data());

    //                 for (int j = 0; j < hk->get_target_cluster().get_size(); ++j) {
    //                     y[j + hk->get_target_cluster().get_offset()] = y_restr[j];
    //                 }
    //             }
    //         }
    //     }
    // }

    // void backward_substitution(std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) {
    //     if (this->is_leaf()) {
    //         for (int j = 0; j < this->get_target_cluster().get_size(); ++j) {
    //             auto U                                          = this->get_dense_data();
    //             int jj                                          = this->get_target_cluster().get_size() - j - 1;
    //             x[jj + this->get_target_cluster().get_offset()] = y[jj + this->get_target_cluster().get_offset()] / U(jj, jj);
    //             for (int i = 0; i < jj; ++i) {
    //                 y[i + this->get_target_cluster().get_offset()] = y[i + this->get_target_cluster().get_offset()] - U(i, jj) * x[jj + this->get_target_cluster().get_offset()];
    //             }
    //         }

    //     } else {
    //         auto child = this->get_children();
    //         for (int k = 0; k < child.size(); ++k) {
    //             auto hk = child[k];
    //             if (hk->get_target_cluster() == hk->get_source_cluster()) {
    //                 backward_substitution(y, x);
    //             } else if (hk->get_source_cluster().get_offset() > hk->get_target_cluster().get_offset()) {
    //                 std::vector<CoefficientPrecision> y_restr(hk->get_target_cluster().get_size());
    //                 std::vector<CoefficientPrecision> x_restr(hk->get_source_cluster().get_size());
    //                 for (int i = 0; i < hk->get_target_cluster().get_size(); ++i) {
    //                     y_restr[i] = y[i + hk->get_target_cluster().get_offset()];
    //                 }
    //                 for (int j = 0; j < hk->get_source_cluster().get_size(); ++j) {
    //                     x_restr[j] = x[j + hk->get_source_cluster().get_offset()];
    //                 }
    //                 y_restr = y_restr - hk->add_vector_product('N', 1.0, x_restr.data(), 0.0, y_restr.data());
    //                 for (int i = 0; i < hk->get_target_cluster().get_size(); ++i) {
    //                     y[i + hk->get_target_cluster().get_size()] = y_restr[i];
    //                 }
    //             }
    //         }
    //     }
    // }

    // void forward_backward(const HMatrix<CoefficientPrecision, CoordinatePrecision> &L, const HMatrix<CoefficientPrecision, CoordinatePrecision> &U, std::vector<CoefficientPrecision> y, std::vector<CoefficientPrecision> x) {
    //     std::vector<CoefficientPrecision> ux(U.get_atrget_cluster().get_size(), 0.0);
    //     L.forward_substitution(y, ux);
    //     U.backward_substitution(ux, x);
    // }

    // // résoudre x^t U = y^t
    // void forward_substitutionT(std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) {
    //     auto M = this->get_dense_data();
    //     if (this->is_leaf()) {
    //         for (int j = this->get_target_cluster().get_offset(); j < this->get_target_cluster().get_size() + this->get_target_cluster().get_offset(); ++j) {
    //             x[j] = y[j] / M(j, j);
    //             for (int i = j + 1; j < this->get_target_cluster().get_size() + this->get_source_cluster().get_offset(); ++i) {
    //                 y[i] = y[i] - M(j, i) * x[j];
    //             }
    //         }
    //     } else {
    //         auto child = this->get_children();
    //         for (int k = 0; k < child.size(); ++k) {
    //             auto &hk = child[k];
    //             if (hk->get_target_cluster() == hk->get_source_cluster()) {
    //                 hk->forward_substitution(y, x);
    //             } else if (hk->get_target_cluster().get_offset() < hk->get_source_cluster().get_offset()) {
    //                 std::vector<CoefficientPrecision> x_restr(hk->get_target_cluster().get_size());
    //                 std::vector<CoefficientPrecision> y_restr(hk->get_source_cluster().get_size());
    //                 for (int i = 0; i < hk->get_source_cluster().get_size(); ++i) {
    //                     y_restr[i] = y[i + hk->get_source_cluster().get_offset()];
    //                 }
    //                 for (int j = 0; j < hk->get_target_cluster().get_size(); ++j) {
    //                     x_restr[j] = x[j + hk->get_target_cluster().get_offset()];
    //                 }
    //                 y_restr = y_restr - hk->add_vector_product('T', 1.0, x_restr.data(), 0.0, y_restr.data());

    //                 for (int j = 0; j < hk->get_source_cluster().get_size(); ++j) {
    //                     y[j + hk->get_source_cluster().get_offset()] = y_restr[j];
    //                 }
    //             }
    //         }
    //     }
    // }

    // // résoudre LX =Z ,
    // friend void Forward_M(HMatrix<CoefficientPrecision, CoordinatePrecision> *L, HMatrix<CoefficientPrecision, CoordinatePrecision> *X, HMatrix<CoefficientPrecision, CoordinatePrecision> *Z) {
    //     if (Z.is_dense()) {
    //         auto Zd = Z->get_dense_data();
    //         auto Xd = X->get_dense_data();
    //         for (int k = 0; k < Z.get_source_cluster().get_size(); ++k) {
    //             std::vector<CoefficientPrecision> Zk(Z->get_target_cluster().get_size());
    //             for (int l = 0; l < Z->target_cluster().get_size(); ++l) {
    //                 Zk[l] = Zd(l, k);
    //             }
    //             // sous entendu X tau sigma dense aussi
    //             auto Xd = X->get_dense_data();
    //             std::vector<CoefficientPrecision> Zk(Z->get_target_cluster().get_size());
    //             for (int l = 0; l < Z->target_cluster().get_size(); ++l) {
    //                 Xk[l] = Xd(l, k);
    //             }

    //             L->forward_substitution(Zk, Xk);
    //             for (int l = 0; l < Z->target_cluster().get_size(); ++l) {
    //                 Zd(l, k) = Zk[l];
    //             }
    //             // sous entendu X tau sigma dense aussi
    //             for (int l = 0; l < Z->target_cluster().get_size(); ++l) {
    //                 Xd(l, k) = Xk[l];
    //             }
    //         }
    //         MatrixGenerator<CoefficientPrecision> mz(Zd);
    //         MatrixGenerator<CoefficientPrecision> mx(Xd);
    //         Z->compute_dense_data(Zd);
    //         X->compute_dense_data(Xd);
    //     }

    //     else if (Z.is_low_rank()) {
    //         auto lrz                        = Z->get_low_rank_data();
    //         Matrix<CoefficientPrecision> Uz = lrz->Get_U();
    //         Matrix<COefficientPrecision> Vz = lrz->Get_V();
    //         int r                           = lrz->rank_of();
    //         Matrix<CoefficientPrecision> Ux(X->get_target_cluster().get_size(), r);
    //         for (int k = 0; k < r; ++k) {
    //             std::vector<CoefficientPrecision> uxk(X->get_target_cluster().get_size());
    //             std::vector<CoefficientPrecision> uzk(Z->get_target_cluster().get_size());
    //             for (int l = 0; l < Z->get_target_cluster().get_size(); ++l) {
    //                 uzk[l] = Uz(l, k);
    //             }
    //             L->forward_substitution(uzk, uxk);
    //             for (int l = 0; l < X->get_target_clsuter().get_size(); ++l) {
    //                 Ux(l, k) = uxk[l];
    //             }
    //         }
    //         LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lrx(Ux, Vz);
    //         X->set_low_rank_data(lrx);
    //     } else {
    //         auto lchild = L->get_childen();
    //         for (int k = 0; k < lchild.size(); ++l) {
    //             auto &lk = lchild[k];
    //             if (lk->get_target_cluster() == lk->get_source_cluster()) {
    //                 for (auto &xk : X->get_children()) {
    //                     if (xk->get_target_cluster() == lk->get_target_cluster()) {
    //                         for (auto &zk : Z->get_children()) {
    //                             if (zk->get_target_cluser() == X->get_target_cluster() and zk->get_slurce_cluster() == X->get_source_cluster()) {
    //                                 lk->Forward_M(zk, xk);
    //                             }
    //                         }
    //                     }
    //                 }
    //                 for (auto &xxk : X->get_children()) {
    //                 }
    //             }
    //         }
    //     }
    // }
    // void add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const;
    // void add_matrix_product(CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const;

    // void mvprod_global_to_global(const T *const in, T *const out, const int &mu = 1) const;
    // void mvprod_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;

    // void mvprod_transp_global_to_global(const T *const in, T *const out, const int &mu = 1) const;
    // void mvprod_transp_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;

    // void mymvprod_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;
    // void mymvprod_global_to_local(const T *const in, T *const out, const int &mu = 1) const;
    // void mymvprod_transp_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;
    // void mymvprod_transp_local_to_global(const T *const in, T *const out, const int &mu = 1) const;

    // void mvprod_subrhs(const T *const in, T *const out, const int &mu, const int &offset, const int &size, const int &margin) const;
    // std::vector<CoefficientPrecision> operator*(const std::vector<CoefficientPrecision> &x) const;
    // Matrix<CoefficientPrecision> operator*(const Matrix<CoefficientPrecision> &x) const;

    // // Permutations
    // void source_to_cluster_permutation(const T *const in, T *const out) const;
    // void local_target_to_local_cluster(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;
    // void local_source_to_local_cluster(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;
    // void local_cluster_to_local_target(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;
    // void local_cluster_to_local_source(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;

    // // local to global
    // void local_to_global_source(const T *const in, T *const out, const int &mu) const;
    // void local_to_global_target(const T *const in, T *const out, const int &mu) const;

    // // Convert
    // Matrix<CoefficientPrecision> get_local_dense() const;
    // Matrix<CoefficientPrecision> get_local_dense_perm() const;
    // void copy_to_dense(CoefficientPrecision *) const;

    // // Apply Dirichlet condition
    // void apply_dirichlet(const std::vector<int> &boundary);
    void fois(const std::vector<CoefficientPrecision> &x, std::vector<CoefficientPrecision> &y) const {
        if (this->get_children().size() == 0) {
            std::vector<CoefficientPrecision> restr(x.begin() + this->get_source_cluster().get_offset(), x.begin() + this->get_source_cluster().get_offset() + this->get_source_cluster().get_size()); // copier le sous-vecteur restreint
            if (this->is_low_rank()) {
                auto ytemp = this->get_low_rank_data()->Get_U() * this->get_low_rank_data()->Get_V() * restr;
                for (int k = 0; k < ytemp.size(); ++k) {
                    y[k + this->get_target_cluster().get_size()] += ytemp[k];
                }
            } else if (this->is_dense()) {
                auto ytemp = (*this->get_dense_data()) * restr;
                for (int k = 0; k < ytemp.size(); ++k) {
                    y[k + this->get_target_cluster().get_size()] += ytemp[k];
                }
            }
        } else {
            for (auto &child : this->get_children()) {
                child->fois(x, y);
            }
        }
    }

    // void mat_vec(const std::vector<CoefficientPrecision> &x, std::vector<CoefficientPrecision> &y) {
    //     if (this->get_children().size() == 0) {
    //         // std::vector<CoefficientPrecision> restr(x.begin() + this->get_source_cluster().get_offset(), x.begin() + this->get_source_cluster().get_offset() + this->get_source_cluster().get_size()); // copier le sous-vecteur restreint
    //         std::vector<CoefficientPrecision> restr(this->get_source_cluster().get_size(), 0.0);
    //         for (int k = 0; k < restr.size(); ++k) {
    //             restr[k] = x[k + this->get_source_cluster().get_offset()];
    //         }
    //         std::vector<CoefficientPrecision> ytemp(this->get_target_cluster().get_size());
    //         this->add_vector_product('N', 1.0, restr.data(), 0.0, ytemp.data());
    //         for (int k = 0; k < ytemp.size(); ++k) {
    //             y[k + this->get_target_cluster().get_size()] += ytemp[k];
    //         }
    //     }

    //     else {
    //         for (auto &child : this->get_children()) {
    //             child->mat_vec(x, y);
    //         }
    //     }
    // }

    void mat_vec(const std::vector<CoefficientPrecision> &x, std::vector<CoefficientPrecision> &y) {
        if (this->get_children().size() == 0) {
            // std::vector<CoefficientPrecision> restr(this->get_target_cluster().get_size(), 0.0);
            // for (int k = 0; k < restr.size(); ++k) {
            //     restr[k] = x[k + this->get_target_cluster().get_offset()];
            // }
            std::vector<CoefficientPrecision> restr(x.begin() + this->get_source_cluster().get_offset(), x.begin() + this->get_source_cluster().get_offset() + this->get_source_cluster().get_size()); // copier le sous-vecteur restreint

            std::vector<CoefficientPrecision> ytemp(this->get_target_cluster().get_size(), 0.0);
            this->add_vector_product('N', 1.0, restr.data(), 0.0, ytemp.data());
            for (int k = 0; k < ytemp.size(); ++k) {
                y[k + this->get_target_cluster().get_offset()] += ytemp[k];
            }
        }

        else {
            for (auto &child : this->get_children()) {
                child->mat_vec(x, y);
            }
        }
    }
    void vec_mat(const std::vector<CoefficientPrecision> &x, std::vector<CoefficientPrecision> &y) const {
        if (this->get_children().size() == 0) {
            std::vector<CoefficientPrecision> restr(x.begin() + this->get_target_cluster().get_offset(), x.begin() + this->get_target_cluster().get_offset() + this->get_target_cluster().get_size()); // copier le sous-vecteur restreint
            std::vector<CoefficientPrecision> ytemp(this->get_source_cluster().get_size());
            this->add_vector_product('T', 1.0, restr.data, 0.0, ytemp.data());
            ytemp.insert(this->get_source_cluster().get_offset(), 0.0);
            ytemp.resize(y.size(), 0.0);
            y = y + ytemp;
        }

        else {
            for (auto &child : this->get_children()) {
                child->vec_mat(x, y);
            }
        }
    }

    // routine por mult Hmat/lrmat ou lrmat Hmat

    // const Matrix<CoefficientPrecision>
    // hmat_lr(const Matrix<CoefficientPrecision> &U) const {
    //     Matrix<CoefficientPrecision> res(this->get_target_cluster().get_size(), U.nb_cols());
    //     for (int k = 0; k < U.nb_cols(); ++k) {
    //         std::vector<CoefficientPrecision> x(U.nb_cols(), 0.0);
    //         x[k] = 1;
    //         // std::vector<CoefficientPrecision> col_k(U.nb_rows());
    //         // for (int i = 0; i < U.nb_rows(); ++i) {
    //         //     col_k[i] = U(i, k);
    //         // }
    //         std::vector<CoefficientPrecision> col_k = U * x;
    //         std::vector<CoefficientPrecision> temp(this->get_target_cluster().get_size(), 0.0);
    //         this->add_vector_product('N', 1.0, col_k.data(), 0.0, temp.data());
    //         for (int i = 0; i < U.nb_rows(); ++i) {
    //             res(i, k) = temp[i];
    //         }
    //     }
    //     return res;
    // }

    // ca peut se paraléliser super facilement mais en soit ca s'appelle sur les lr donc on peut pas avoir plus de threads que le rank de la matrice ( nb_col)

    const Matrix<CoefficientPrecision> hmat_lr(const Matrix<CoefficientPrecision> &U) const {
        std::vector<CoefficientPrecision> Hu(U.nb_rows(), 0.0);
        Matrix<CoefficientPrecision> res(this->get_target_cluster().get_size(), U.nb_cols());
        auto start_time = std::clock(); // temps de départ
        for (int k = 0; k < U.nb_cols(); ++k) {
            this->add_vector_product('N', 1.0, U.get_col(k).data(), 0.0, Hu.data());
            res.set_col(k, Hu);
        }
        return res;
    }

    void hmat_lr_plus(const std::vector<std::vector<CoefficientPrecision>> col_U, std::vector<std::vector<CoefficientPrecision>> res) const {
        std::cout << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << ',' << this->get_target_cluster().get_offset() << ',' << this->get_source_cluster().get_offset() << std::endl;
        if (this->get_children().size() == 0) {
            const double alpha = 1.0;
            const double beta  = 0.0;
            const int inc      = 1;
            if (this->is_low_rank()) {
                std::vector<CoefficientPrecision> temp(this->get_low_rank_data()->Get_V().nb_rows(), 0.0);
                for (int k = 0; k < col_U.size(); ++k) {
                    std::cout << "step " << k << std::endl;
                    int nr = this->get_target_cluster().get_size();
                    std::cout << "nr ok " << std::endl;
                    int nc = this->get_source_cluster().get_size();
                    std::cout << "nc ok" << std::endl;
                    int oft = this->get_target_cluster().get_offset();
                    std::cout << "oft ok " << std::endl;
                    int ofs = this->get_source_cluster().get_offset();
                    std::cout << "ofs ok" << std::endl;
                    char n   = 'N';
                    auto lda = nr;
                    std::cout << "begin" << std::endl;
                    Blas<CoefficientPrecision>::gemv(&n, &nr, &nc, &alpha, &(*this->get_low_rank_data()->Get_V().data()), &lda, col_U[k].data() + ofs, &inc, &beta, temp.data(), &inc);
                    std::cout << "begin1 " << std::endl;
                    Blas<CoefficientPrecision>::gemv(&n, &nr, &nc, &alpha, &(*this->get_low_rank_data()->Get_U().data()), &lda, temp.data(), &inc, &alpha, res[k].data() + oft, &inc);
                    std::cout << "eeeend" << std::endl;
                }
            } else if (this->is_dense()) {
                std::cout << "begin dense " << std::endl;
                for (int k = 0; k < col_U.size(); ++k) {

                    int nr   = this->get_target_cluster().get_size();
                    int nc   = this->get_source_cluster().get_size();
                    int oft  = this->get_target_cluster().get_offset();
                    int ofs  = this->get_source_cluster().get_offset();
                    char n   = 'N';
                    auto lda = nr;
                    Blas<CoefficientPrecision>::gemv(&n, &nr, &nc, &alpha, this->get_dense_data()->data(), &lda, col_U[k].data() + ofs, &inc, &alpha, res[k].data() + ofs, &inc);
                }
                std::cout << "end dense " << std::endl;
            }
        } else {
            for (auto &child : this->get_children()) {
                child->hmat_lr_plus(col_U, res);
            }
        }
    }

    const Matrix<CoefficientPrecision>
    lr_hmat(const Matrix<CoefficientPrecision> &V) const {
        std::vector<CoefficientPrecision> vH(V.nb_cols(), 0.0);
        Matrix<CoefficientPrecision> res(V.nb_rows(), this->get_source_cluster().get_size());
        for (int k = 0; k < V.nb_rows(); ++k) {
            this->add_vector_product('T', 1.0, V.get_row(k).data(), 0.0, vH.data());
            res.set_row(k, vH);
        }
        return res;
    }

    // const Matrix<CoefficientPrecision> lr_hmat(const Matrix<CoefficientPrecision> &V) const {
    //     Matrix<CoefficientPrecision> res(V.nb_rows(), this->get_source_cluster().get_size());
    //     for (int k = 0; k < V.nb_rows(); ++k) {
    //         // std::vector<CoefficientPrecision> row_k(V.nb_cols());
    //         // for (int j = 0; j < V.nb_cols(); ++j) {
    //         //     row_k[j] = V(k, j);
    //         // }
    //         std::vector<CoefficientPrecision> x(V.nb_rows(), 0.0);
    //         x[k] = 1;
    //         std::vector<CoefficientPrecision> row_k(V.nb_cols(), 0.0);
    //         V.add_vector_product('T', 1.0, x.data(), 0.0, row_k.data());
    //         std::vector<CoefficientPrecision>
    //             temp(this->get_source_cluster().get_size(), 0.0);
    //         this->add_vector_product('T', 1.0, row_k.data(), 0.0, temp.data());
    //         for (int j = 0; j < this->get_source_cluster().get_size(); ++j) {
    //             res(k, j) = temp[j];
    //         }
    //     }
    //     return res;
    // }

    const Matrix<CoefficientPrecision>
    mult(const Matrix<CoefficientPrecision> &B, char C) const {
        // bon c'est vraiment affreux mais je comprend pas comment utiliser mult  , ce cas ca sert a faire Hmat*lrmat en faisant Hmat*mat*mat
        Matrix<CoefficientPrecision> res;
        std::cout << "hoho" << std::endl;
        if (C == 'N') {
            Matrix<CoefficientPrecision> H_dense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
            auto &H = *this;
            // add_matrix_product_row_major('N', 1, B.data(), 1, Coefficient, 1);
            // void add_matrix_product_row_major(char trans, CoefficientPrecision alpha, CoefficientPrecision *in, CoefficientPrecision 0, CoefficientPrecision *res.data(), int 1);
            copy_to_dense(H, H_dense.data());
            res.assign(H_dense.nb_rows(), B.nb_cols(), (H_dense * B).data(), true);
            // res = H_dense * B;
        } else {
            std::cout << "héhé 1" << std::endl;
            Matrix<CoefficientPrecision> H_dense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
            std::cout << "héhé 2" << std::endl;
            auto &H = *this;
            std::cout << "héhé 3" << std::endl;
            copy_to_dense(H, H_dense.data());
            std::cout << "héhé 4" << std::endl;
            res.assign(B.nb_rows(), H_dense.nb_cols(), (B * H_dense).data(), true);
            std::cout << "héhé 5" << std::endl;
            // res = B * H_dense;
        }
        return res;
    }
    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_mult(LowRankMatrix<CoefficientPrecision, CoordinatePrecision> B, char C) {
        // N = H*lr , T = lr*H
        if (C == 'N') {
            auto u  = B.Get_U();
            auto v  = B.Get_V();
            auto Hu = this->mult(u, 'N');
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(Hu, v);
            return lr;
        } else {
            auto u  = B.Get_U();
            auto v  = B.Get_V();
            auto vH = this->mult(v, 'T');
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(u, vH);
            return lr;
        }
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

// // build
// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::build(VirtualGenerator<CoefficientPrecision> &mat) {

//     // std::vector<double> mytimes(3), maxtime(3), meantime(3);

//     // // Default compression: sympartialACA
//     // if (m_block_tree_properties->m_low_rank_generator == nullptr) {
//     //     m_block_tree_properties->m_low_rank_generator = std::make_shared<sympartialACA<CoefficientPrecision>>();
//     // }

//     // Build block tree
//     bool not_pushed = false;
//     // if (m_blockm_block_tree_properties->m_UPLO == 'U' || m_block_tree_properties->m_UPLO == 'L') {
//     //     not_pushed = build_symmetric_block_tree();
//     // } else {
//     not_pushed = build_block_tree();
//     // }
//     // for (const std::unique_ptr<HMatrix<CoefficientPrecision,CoordinatePrecision>> &child : m_children) {
//     //     std::cout << child->get_target_cluster_tree().get_offset() << " " << child->get_target_cluster_tree().get_size() << " " << child->get_source_cluster_tree().get_offset() << " " << child->get_source_cluster_tree().get_size() << "\n";
//     // }

//     reset_root_of_block_tree();
//     if (not_pushed) {
//         m_block_tree_properties->m_tasks.push_back(this);
//     }

//     // for (const std::unique_ptr<HMatrix<CoefficientPrecision,CoordinatePrecision>> &child : m_children) {
//     //     std::cout << child->get_target_cluster_tree().get_offset() << " " << child->get_target_cluster_tree().get_size() << " " << child->get_source_cluster_tree().get_offset() << " " << child->get_source_cluster_tree().get_size() << "\n";
//     // }

//     // // Sort local tasks
//     // std::sort(m_block_tree_properties->m_tasks.begin(), m_block_tree_properties->m_tasks.end(), [](HMatrix *hmatrix_A, HMatrix *hmatrix_B) { return (hmatrix_A->m_source_cluster_tree->get_offset() == hmatrix_B->m_source_cluster_tree->get_offset()) ? (hmatrix_A->m_target_cluster_tree->get_offset() < hmatrix_B->m_target_cluster_tree->get_offset()) : hmatrix_A->m_source_cluster_tree->get_offset() < hmatrix_B->m_source_cluster_tree->get_offset(); });

//     // Compute blocks
//     compute_blocks(mat);

//     // // Infos
//     // ComputeInfos(mytimes);
// }

// // Symmetry build
// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::build(VirtualGenerator<CoefficientPrecision> &mat, const double *const xt) {

//     MPI_Comm_size(comm, &sizeWorld);
//     MPI_Comm_rank(comm, &rankWorld);
//     std::vector<double> mytimes(3), maxtime(3), meantime(3);

//     this->nc        = mat.nb_cols();
//     this->nr        = mat.nb_rows();
//     this->dimension = mam_target_cluster_tree->get_dimension();

//     // Default compression: sympartialACA
//     if (this->LowRankGenerator == nullptr) {
//         this->LowRankGenerator = std::make_shared<sympartialACA<CoefficientPrecision>>();
//     }

//     // Default admissibility condition
//     if (this->AdmissibilityCondition == nullptr) {
//         this->AdmissibilityCondition = std::make_shared<RjasanowSteinbach>();
//     }

//     // Zero generator when we delay the dense computation
//     if (delay_dense_computation) {
//         zerogenerator = std::unique_ptr<ZeroGenerator<CoefficientPrecision>>(new ZeroGenerator<CoefficientPrecision>(mat.nb_rows(), mat.nb_cols(), mam_target_cluster_tree->get_dimension()));
//     }

//     // Construction arbre des paquets
//     local_size   = cluster_tree_t->get_local_size();
//     local_offset = cluster_tree_t->get_local_offset();

//     // Construction arbre des blocs
//     double time = MPI_Wtime();

//     if (this->OffDiagonalApproximation != nullptr) {
//         this->BlockTree.reset(new Block<CoefficientPrecision>(this->AdmissibilityCondition->get(), cluster_tree_t->get_local_cluster(), cluster_tree_s->get_local_cluster()));
//     } else {
//         this->BlockTree.reset(new Block<CoefficientPrecision>(this->AdmissibilityCondition->get(), *cluster_tree_t, *cluster_tree_s));
//     }
//     this->BlockTree->set_mintargetdepth(m_mintargetdepth);
//     this->BlockTree->set_minsourcedepth(m_minsourcedepth);
//     this->BlockTree->set_maxblocksize(this->maxblocksize);
//     this->BlockTree->set_eta(this->eta);
//     bool force_sym = true;
//     this->BlockTree->build(UPLO, force_sym, comm);

//     mytimes[0] = MPI_Wtime() - time;

//     // Assemblage des sous-matrices
//     time = MPI_Wtime();
//     ComputeBlocks(mat, xt, xt);
//     mytimes[1] = MPI_Wtime() - time;

//     // Infos
//     ComputeInfos(mytimes);
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::build_dense_blocks(VirtualDenseBlocksGenerator<CoefficientPrecision> &dense_block_generator) {

//     int number_of_dense_blocks = m_block_tree_properties->m_dense_leaves.size();
//     auto &dense_blocks         = m_block_tree_properties->m_dense_leaves;

//     std::vector<int> row_sizes(number_of_dense_blocks, 0), col_sizes(number_of_dense_blocks, 0);
//     std::vector<const int *> rows(number_of_dense_blocks, nullptr), cols(number_of_dense_blocks, nullptr);
//     std::vector<T *> ptr(number_of_dense_blocks, nullptr);

//     for (int i = 0; i < number_of_dense_blocks; i++) {
//         row_sizes[i] = dense_blocks[i]->get_target_cluster()->get_size();
//         col_sizes[i] = dense_blocks[i]->get_source_cluster()->get_size();
//         rows[i]      = dense_blocks[i]->get_target_cluster()->get_perm_data();
//         cols[i]      = dense_blocks[i]->get_source_cluster()->get_perm_data();
//         ptr[i]       = dense_blocks[i]->get_dense_block_data()->data();
//     }
//     dense_block_generator.copy_dense_blocks(row_sizes, col_sizes, rows, cols, ptr);
// }

// // // Compute blocks recursively
// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::compute_blocks(const VirtualGenerator<CoefficientPrecision> &mat) {
// #if defined(_OPENMP) && !defined(PYTHON_INTERFACE)
// #    pragma omp parallel
// #endif
//     {
//         std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> local_dense_leaves{};
//         std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> local_low_rank_leaves{};
//         std::vector<HMatrix *> &local_tasks = m_block_tree_properties->m_tasks;

//         int false_positive_local = 0;
// #if defined(_OPENMP) && !defined(PYTHON_INTERFACE)
// #    pragma omp for schedule(guided)
// #endif
//         for (int p = 0; p < m_block_tree_properties->m_tasks.size(); p++) {
//             if (!local_tasks[p]->is_admissible()) {
//                 local_tasks[p]->add_dense_leaf(mat, local_dense_leaves);
//             } else {
//                 // bool not_pushed;
//                 local_tasks[p]->add_low_rank_leaf(mat, local_low_rank_leaves);
//                 if (local_tasks[p]->get_low_rank_data()->rank_of() == -1) {
//                     local_tasks[p]->m_low_rank_data.reset();
//                     local_low_rank_leaves.pop_back();
//                     local_tasks[p]->add_dense_leaf(mat, local_dense_leaves);
//                 }
//                 // if (m_symmetry == 'H' || m_symmetry == 'S') {
//                 //     not_pushed = ComputeAdmissibleBlocksSym(mat, *(local_tasks[p]), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//                 // } else {
//                 //     not_pushed = ComputeAdmissibleBlock(mat, *(local_tasks[p]), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//                 // }

//                 // if (not_pushed) {
//                 //     local_tasks[p]->add_dense_leaf(mat, local_dense_leaves);
//                 // }
//             }
//         }
// #if defined(_OPENMP) && !defined(PYTHON_INTERFACE)
// #    pragma omp critical
// #endif
//         {
//             m_block_tree_properties->m_low_rank_leaves.insert(m_block_tree_properties->m_low_rank_leaves.end(), std::make_move_iterator(local_low_rank_leaves.begin()), std::make_move_iterator(local_low_rank_leaves.end()));

//             m_block_tree_properties->m_dense_leaves.insert(m_block_tree_properties->m_dense_leaves.end(), std::make_move_iterator(local_dense_leaves.begin()), std::make_move_iterator(local_dense_leaves.end()));

//             m_block_tree_properties->m_false_positive += false_positive_local;
//         }
//     }

//     m_block_tree_properties->m_leaves.insert(m_block_tree_properties->m_leaves.end(), m_block_tree_properties->m_dense_leaves.begin(), m_block_tree_properties->m_dense_leaves.end());
//     m_block_tree_properties->m_leaves.insert(m_block_tree_properties->m_leaves.end(), m_block_tree_properties->m_low_rank_leaves.begin(), m_block_tree_properties->m_low_rank_leaves.end());

//     if (m_block_tree_properties->m_block_diagonal_hmatrix != nullptr) {
//         int local_offset_s = m_block_tree_properties->m_block_diagonal_hmatrix->get_source_cluster_tree().get_offset();
//         int local_size_s   = m_block_tree_properties->m_block_diagonal_hmatrix->get_source_cluster_tree().get_size();

//         // Build vectors of pointers for diagonal blocks
//         for (auto leaf : m_block_tree_properties->m_leaves) {
//             if (local_offset_s <= leaf->get_source_cluster_tree().get_offset() && leaf->get_source_cluster_tree().get_offset() < local_offset_s + local_size_s) {
//                 m_block_tree_properties->m_leaves_in_diagonal_block.push_back(leaf);
//             }
//         }
//         for (auto low_rank_leaf : m_block_tree_properties->m_low_rank_leaves) {
//             if (local_offset_s <= low_rank_leaf->get_source_cluster_tree().get_offset() && low_rank_leaf->get_source_cluster_tree().get_offset() < local_offset_s + local_size_s) {
//                 m_block_tree_properties->m_low_rank_leaves_in_diagonal_block.push_back(low_rank_leaf);
//                 if (low_rank_leaf->get_source_cluster_tree().get_offset() == low_rank_leaf->get_target_cluster_tree().get_offset()) {
//                     m_block_tree_properties->m_diagonal_low_rank_leaves.push_back(low_rank_leaf);
//                 }
//             }
//         }
//         for (auto dense_leaf : m_block_tree_properties->m_dense_leaves) {
//             if (local_offset_s <= dense_leaf->get_source_cluster_tree().get_offset() && dense_leaf->get_source_cluster_tree().get_offset() < local_offset_s + local_size_s) {
//                 m_block_tree_properties->m_dense_leaves_in_diagonal_block.push_back(dense_leaf);
//                 if (dense_leaf->get_source_cluster_tree().get_offset() == dense_leaf->get_target_cluster_tree().get_offset()) {
//                     m_block_tree_properties->m_diagonal_dense_leaves.push_back(dense_leaf);
//                 }
//             }
//         }
//     }
// }

// template <typename CoefficientPrecision>
// bool HMatrix<CoefficientPrecision,CoordinatePrecision>::ComputeAdmissibleBlock(VirtualGenerator<CoefficientPrecision> &mat, Block<CoefficientPrecision> &task, const double *const xt, const double *const xs, std::vector<Block<CoefficientPrecision> *> &MyComputedBlocks_local, std::vector<Block<CoefficientPrecision> *> &MyNearFieldMats_local, std::vector<Block<CoefficientPrecision> *> &MyFarFieldMats_local, int &false_positive_local) {
//     if (task.IsAdmissible()) { // When called recursively, it may not be admissible
//         AddFarFieldMat(mat, task, xt, xs, MyComputedBlocks_local, MyFarFieldMats_local, reqrank);
//         if (MyFarFieldMats_local.back()->get_rank_of() != -1) {
//             return false;
//         } else {
//             MyComputedBlocks_local.back()->clear_data();
//             MyFarFieldMats_local.pop_back();
//             MyComputedBlocks_local.pop_back();
//             false_positive_local += 1;
//         }
//     }
//     // We could compute a dense block if its size is small enough, we focus on improving compression for now
//     // else if (task->get_size()<maxblocksize){
//     //     AddNearFieldMat(mat,task,MyComputedBlocks_local, MyNearFieldMats_local);
//     //     return false;
//     // }

//     std::size_t bsize       = task->get_size();
//     const VirtualCluster &t = m_target_cluster;
//     const VirtualCluster &s = m_source_cluster.

//     if (s.IsLeaf()) {
//         if (t.IsLeaf()) {
//             return true;
//         } else {
//             std::vector<bool> Blocks_not_pushed(m_target_cluster_tree->number_of_children());
//             for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                 task.build_son(m_target_cluster_tree->get_children()[p], s);

//                 Blocks_not_pushed[p] = ComputeAdmissibleBlock(mat, task->get_children()[p], xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//             }

//             if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                 task.clear_sons();
//                 return true;
//             } else {
//                 for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                     if (Blocks_not_pushed[p]) {
//                         AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                     }
//                 }
//                 return false;
//             }
//         }
//     } else {
//         if (t.IsLeaf()) {
//             std::vector<bool> Blocks_not_pushed(m_source_cluster_tree->number_of_children());
//             for (int p = 0; p < m_source_cluster_tree->number_of_children(); p++) {
//                 task.build_son(t, m_source_cluster_tree->get_children()[p]);
//                 Blocks_not_pushed[p] = ComputeAdmissibleBlock(mat, task->get_children()[p], xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//             }

//             if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                 task.clear_sons();
//                 return true;
//             } else {
//                 for (int p = 0; p < m_source_cluster_tree->number_of_children(); p++) {
//                     if (Blocks_not_pushed[p]) {
//                         AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                     }
//                 }
//                 return false;
//             }
//         } else {
//             if (m_target_cluster_tree->get_size() > m_source_cluster_tree->get_size()) {
//                 std::vector<bool> Blocks_not_pushed(m_target_cluster_tree->number_of_children());
//                 for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                     task.build_son(m_target_cluster_tree->get_children()[p], s);
//                     Blocks_not_pushed[p] = ComputeAdmissibleBlock(mat, task->get_children()[p], xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//                 }

//                 if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                     task.clear_sons();
//                     return true;
//                 } else {
//                     for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                         if (Blocks_not_pushed[p]) {
//                             AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                         }
//                     }
//                     return false;
//                 }
//             } else {
//                 std::vector<bool> Blocks_not_pushed(m_source_cluster_tree->number_of_children());
//                 for (int p = 0; p < m_source_cluster_tree->number_of_children(); p++) {
//                     task.build_son(t, m_source_cluster_tree->get_children()[p]);
//                     Blocks_not_pushed[p] = ComputeAdmissibleBlock(mat, task->get_children()[p], xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//                 }

//                 if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                     task.clear_sons();
//                     return true;
//                 } else {
//                     for (int p = 0; p < m_source_cluster_tree->number_of_children(); p++) {
//                         if (Blocks_not_pushed[p]) {
//                             AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                         }
//                     }
//                     return false;
//                 }
//             }
//         }
//     }
// }

// template <typename CoefficientPrecision>
// bool HMatrix<CoefficientPrecision,CoordinatePrecision>::ComputeAdmissibleBlocksSym(VirtualGenerator<CoefficientPrecision> &mat, Block<CoefficientPrecision> &task, const double *const xt, const double *const xs, std::vector<Block<CoefficientPrecision> *> &MyComputedBlocks_local, std::vector<Block<CoefficientPrecision> *> &MyNearFieldMats_local, std::vector<Block<CoefficientPrecision> *> &MyFarFieldMats_local, int &false_positive_local) {

//     if (task.IsAdmissible()) {

//         AddFarFieldMat(mat, task, xt, xs, MyComputedBlocks_local, MyFarFieldMats_local, reqrank);
//         if (MyFarFieldMats_local.back()->get_rank_of() != -1) {
//             return false;
//         } else {
//             MyComputedBlocks_local.back()->clear_data();
//             MyFarFieldMats_local.pop_back();
//             MyComputedBlocks_local.pop_back();
//             false_positive_local += 1;
//             // AddNearFieldMat(mat, task,MyComputedBlocks_local, MyNearFieldMats_local);
//             // return false;
//         }
//     }
//     // We could compute a dense block if its size is small enough, we focus on improving compression for now
//     // else if (task->get_size()<maxblocksize){
//     //     AddNearFieldMat(mat,task,MyComputedBlocks_local, MyNearFieldMats_local);
//     //     return false;
//     // }

//     std::size_t bsize       = task->get_size();
//     const VirtualCluster &t = m_target_cluster;
//     const VirtualCluster &s = m_source_cluster.

//     if (s.IsLeaf()) {
//         if (t.IsLeaf()) {
//             return true;
//         } else {
//             std::vector<bool> Blocks_not_pushed(m_target_cluster_tree->number_of_children());
//             for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                 task.build_son(m_target_cluster_tree->get_children()[p], s);
//                 Blocks_not_pushed[p] = ComputeAdmissibleBlocksSym(mat, task->get_children()[p], xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//             }

//             if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                 task.clear_sons();
//                 return true;
//             } else {
//                 for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                     if (Blocks_not_pushed[p]) {
//                         AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                     }
//                 }
//                 return false;
//             }
//         }
//     } else {
//         if (t.IsLeaf()) {
//             std::vector<bool> Blocks_not_pushed(m_source_cluster_tree->number_of_children());
//             for (int p = 0; p < m_source_cluster_tree->number_of_children(); p++) {
//                 task.build_son(t, m_source_cluster_tree->get_children()[p]);
//                 Blocks_not_pushed[p] = ComputeAdmissibleBlocksSym(mat, task->get_children()[p], xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//             }

//             if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                 task.clear_sons();
//                 return true;
//             } else {
//                 for (int p = 0; p < m_source_cluster_tree->number_of_children(); p++) {
//                     if (Blocks_not_pushed[p]) {
//                         AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                     }
//                 }
//                 return false;
//             }
//         } else {
//             std::vector<bool> Blocks_not_pushed(m_target_cluster_tree->number_of_children() * m_source_cluster_tree->number_of_children());
//             for (int l = 0; l < m_source_cluster_tree->number_of_children(); l++) {
//                 for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                     task.build_son(m_target_cluster_tree->get_children()[p], m_source_cluster_tree->get_son(l));
//                     Blocks_not_pushed[p + l * m_target_cluster_tree->number_of_children()] = ComputeAdmissibleBlocksSym(mat, task->get_son(p + l * m_target_cluster_tree->number_of_children()), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//                 }
//             }
//             if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                 task.clear_sons();
//                 return true;
//             } else {
//                 for (int p = 0; p < Blocks_not_pushed.size(); p++) {
//                     if (Blocks_not_pushed[p]) {
//                         AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                     }
//                 }
//                 return false;
//             }
//         }
//     }
// }

// // Build a dense block
// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::add_dense_leaf(const VirtualGenerator<CoefficientPrecision> &mat, std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> &local_dense_leaves) {

//     if (!m_block_tree_properties->m_delay_dense_computation) {
//         m_dense_data = std::unique_ptr<Matrix<CoefficientPrecision>>(new Matrix<CoefficientPrecision>(m_target_cluster_tree->get_size(), m_source_cluster_tree->get_size()));

//         mat.copy_submatrix(m_target_cluster_tree->get_size(), m_source_cluster_tree->get_size(), m_target_cluster_tree->get_offset(), m_source_cluster_tree->get_offset(), m_dense_data->data());
//     }
//     local_dense_leaves.push_back(this);
//     m_storage_type = StorageType::Dense;
// }

// // // Build a low rank block
// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::add_low_rank_leaf(const VirtualGenerator<CoefficientPrecision> &mat, std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> &local_low_rank_leaves) {

//     m_low_rank_data = std::unique_ptr<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(new LowRankMatrix<CoefficientPrecision, CoordinatePrecision>(mat, *m_block_tree_properties->m_low_rank_generator, *m_target_cluster_tree, *m_source_cluster_tree, m_block_tree_properties->m_reqrank, m_block_tree_properties->m_epsilon));

//     local_low_rank_leaves.push_back(this);
//     m_storage_type = StorageType::LowRank;
// }

// // Compute infos
// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::ComputeInfos(const std::vector<double> &mytime) {
//     // 0 : block tree ; 1 : compute blocks ;
//     std::vector<double> maxtime(2), meantime(2);
//     // 0 : dense mat ; 1 : lr mat ; 2 : rank ; 3 : local_size
//     std::vector<std::size_t> maxinfos(4, 0), mininfos(4, std::max(nc, nr));
//     std::vector<double> meaninfos(4, 0);
//     // Infos
//     for (int i = 0; i < MyNearFieldMats.size(); i++) {
//         std::size_t size = MyNearFieldMats[i]->get_target_cluster()->get_size() * MyNearFieldMats[i]->get_source_cluster()->get_size();
//         maxinfos[0]      = std::max(maxinfos[0], size);
//         mininfos[0]      = std::min(mininfos[0], size);
//         meaninfos[0] += size;
//     }
//     for (int i = 0; i < MyFarFieldMats.size(); i++) {
//         std::size_t size = MyFarFieldMats[i]->get_target_cluster()->get_size() * MyFarFieldMats[i]->get_source_cluster()->get_size();
//         std::size_t rank = MyFarFieldMats[i]->get_rank_of();
//         maxinfos[1]      = std::max(maxinfos[1], size);
//         mininfos[1]      = std::min(mininfos[1], size);
//         meaninfos[1] += size;
//         maxinfos[2] = std::max(maxinfos[2], rank);
//         mininfos[2] = std::min(mininfos[2], rank);
//         meaninfos[2] += rank;
//     }
//     maxinfos[3]  = local_size;
//     mininfos[3]  = local_size;
//     meaninfos[3] = local_size;

//     if (rankWorld == 0) {
//         MPI_Reduce(MPI_IN_PLACE, &(maxinfos[0]), 4, my_MPI_SIZE_T, MPI_MAX, 0, comm);
//         MPI_Reduce(MPI_IN_PLACE, &(mininfos[0]), 4, my_MPI_SIZE_T, MPI_MIN, 0, comm);
//         MPI_Reduce(MPI_IN_PLACE, &(meaninfos[0]), 4, MPI_DOUBLE, MPI_SUM, 0, comm);
//         MPI_Reduce(MPI_IN_PLACE, &(false_positive), 1, MPI_INT, MPI_SUM, 0, comm);
//     } else {
//         MPI_Reduce(&(maxinfos[0]), &(maxinfos[0]), 4, my_MPI_SIZE_T, MPI_MAX, 0, comm);
//         MPI_Reduce(&(mininfos[0]), &(mininfos[0]), 4, my_MPI_SIZE_T, MPI_MIN, 0, comm);
//         MPI_Reduce(&(meaninfos[0]), &(meaninfos[0]), 4, MPI_DOUBLE, MPI_SUM, 0, comm);
//         MPI_Reduce(&(false_positive), &(false_positive), 1, MPI_INT, MPI_SUM, 0, comm);
//     }

//     int nlrmat   = this->get_nlrmat();
//     int ndmat    = this->get_ndmat();
//     meaninfos[0] = (ndmat == 0 ? 0 : meaninfos[0] / ndmat);
//     meaninfos[1] = (nlrmat == 0 ? 0 : meaninfos[1] / nlrmat);
//     meaninfos[2] = (nlrmat == 0 ? 0 : meaninfos[2] / nlrmat);
//     meaninfos[3] = meaninfos[3] / sizeWorld;
//     mininfos[0]  = (ndmat == 0 ? 0 : mininfos[0]);
//     mininfos[1]  = (nlrmat == 0 ? 0 : mininfos[1]);
//     mininfos[2]  = (nlrmat == 0 ? 0 : mininfos[2]);

//     // timing
//     MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0, comm);
//     MPI_Reduce(&(mytime[0]), &(meantime[0]), 2, MPI_DOUBLE, MPI_SUM, 0, comm);

//     meantime /= sizeWorld;

//     infos["Block_tree_mean"] = NbrToStr(meantime[0]);
//     infos["Block_tree_max"]  = NbrToStr(maxtime[0]);
//     infos["Blocks_mean"]     = NbrToStr(meantime[1]);
//     infos["Blocks_max"]      = NbrToStr(maxtime[1]);

//     // Size
//     infos["Source_size"]              = NbrToStr(this->nc);
//     infos["Target_size"]              = NbrToStr(this->nr);
//     infos["Dimension"]                = NbrToStr(this->dimension);
//     infos["Dense_block_size_max"]     = NbrToStr(maxinfos[0]);
//     infos["Dense_block_size_mean"]    = NbrToStr(meaninfos[0]);
//     infos["Dense_block_size_min"]     = NbrToStr(mininfos[0]);
//     infos["Low_rank_block_size_max"]  = NbrToStr(maxinfos[1]);
//     infos["Low_rank_block_size_mean"] = NbrToStr(meaninfos[1]);
//     infos["Low_rank_block_size_min"]  = NbrToStr(mininfos[1]);

//     infos["Rank_max"]                 = NbrToStr(maxinfos[2]);
//     infos["Rank_mean"]                = NbrToStr(meaninfos[2]);
//     infos["Rank_min"]                 = NbrToStr(mininfos[2]);
//     infos["Number_of_lrmat"]          = NbrToStr(nlrmat);
//     infos["Number_of_dmat"]           = NbrToStr(ndmat);
//     infos["Number_of_false_positive"] = NbrToStr(false_positive);
//     infos["Local_compressed_size"]    = NbrToStr(this->local_compressed_size());
//     infos["Compression_ratio"]        = NbrToStr(this->compression_ratio());
//     infos["Space_saving"]             = NbrToStr(this->space_saving());
//     infos["Local_size_max"]           = NbrToStr(maxinfos[3]);
//     infos["Local_size_mean"]          = NbrToStr(meaninfos[3]);
//     infos["Local_size_min"]           = NbrToStr(mininfos[3]);

//     infos["Number_of_MPI_tasks"] = NbrToStr(sizeWorld);
// #if defined(_OPENMP)
//     infos["Number_of_threads_per_tasks"] = NbrToStr(omp_get_max_threads());
//     infos["Number_of_procs"]             = NbrToStr(sizeWorld * omp_get_max_threads());
// #else
//     infos["Number_of_procs"] = NbrToStr(sizeWorld);
// #endif

//     infos["Eta"]                   = NbrToStr(eta);
//     infos["Eps"]                   = NbrToStr(epsilon);
//     infos["MinTargetDepth"]        = NbrToStr(mintargetdepth);
//     infos["MinSourceDepth"]        = NbrToStr(minsourcedepth);
//     infos["MinClusterSizeTarget"]  = NbrToStr(cluster_tree_t->get_minclustersize());
//     infos["MinClusterSizeSource"]  = NbrToStr(cluster_tree_s->get_minclustersize());
//     infos["MinClusterDepthTarget"] = NbrToStr(cluster_tree_t->get_min_depth());
//     infos["MaxClusterDepthTarget"] = NbrToStr(cluster_tree_t->get_max_depth());
//     infos["MinClusterDepthSource"] = NbrToStr(cluster_tree_s->get_min_depth());
//     infos["MaxClusterDepthSource"] = NbrToStr(cluster_tree_s->get_max_depth());
//     infos["MaxBlockSize"]          = NbrToStr(maxblocksize);
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const {

//     int target_size   = m_target_cluster_tree->get_size();
//     int source_size   = m_source_cluster_tree->get_size();
//     int source_offset = m_source_cluster_tree->get_offset();
//     int target_offset = m_target_cluster_tree->get_offset();

//     int incx(1), incy(1);
//     CoefficientPrecision da(1);
//     const auto &blocks                   = m_block_tree_properties->m_tasks;
//     const auto &blocks_in_diagonal_block = m_block_tree_properties->m_leaves_in_diagonal_block;
//     const auto &diagonal_dense_blocks    = m_block_tree_properties->m_diagonal_dense_leaves;
//     CoefficientPrecision local_beta      = 0;

//     char symmetry_trans;
//     if (m_block_tree_properties->m_symmetry == 'S') {
//         if (trans == 'N') {
//             symmetry_trans = 'T';
//         } else {
//             symmetry_trans = 'N';
//         }
//     } else if (m_block_tree_properties->m_symmetry == 'H') {
//         if (trans == 'N') {
//             symmetry_trans = 'C';
//         } else {
//             symmetry_trans = 'N';
//         }
//     }

//      std::transform(out, out + target_size, out, [beta](CoefficientPrecision a) { return a * beta; });

// // Contribution champ lointain
// #if defined(_OPENMP)
// #    pragma omp parallel
// #endif
//     {
//         std::vector<CoefficientPrecision> temp(target_size, 0);
// #if defined(_OPENMP)
// #    pragma omp for schedule(guided) nowait
// #endif
//         for (int b = 0; b < blocks.size(); b++) {
//             int offset_i = blocks[b]->get_target_cluster_tree().get_offset() - target_offset;
//             int offset_j = blocks[b]->get_source_cluster_tree().get_offset() - source_offset;
//              std::cout << offset_i << " " << offset_j << "\n";
//              if (m_block_tree_properties->m_symmetry == 'N' || offset_i != offset_j) { // remove strictly diagonal blocks
//                  blocks[b]->is_dense() ? blocks[b]->get_dense_data()->add_vector_product(trans, alpha, in + offset_j, local_beta, temp.data() + offset_i) : blocks[b]->get_low_rank_data()->add_vector_product(trans, alpha, in + offset_j, local_beta, temp.data() + offset_i);
//                  ;
//              }
//          }

//         // Symmetry part of the diagonal part
//         if (m_block_tree_properties->m_symmetry != 'N') {
// #if defined(_OPENMP)
// #    pragma omp for schedule(guided) nowait
// #endif
//              for (int b = 0; b < blocks_in_diagonal_block.size(); b++) {
//                  int offset_i = blocks_in_diagonal_block[b]->get_source_cluster_tree().get_offset();
//                  int offset_j = blocks_in_diagonal_block[b]->get_target_cluster_tree().get_offset();

//                  if (offset_i != offset_j) { // remove strictly diagonal blocks
//                      blocks[b]->is_dense() ? blocks_in_diagonal_block[b]->get_dense_data()->add_vector_product(symmetry_trans, alpha, in + offset_j - target_offset, local_beta, temp.data() + (offset_i - source_offset)) : blocks_in_diagonal_block[b]->get_low_rank_data()->add_vector_product(symmetry_trans, alpha, in + offset_j - target_offset, local_beta, temp.data() + (offset_i - source_offset));
//                  }
//              }

// #if defined(_OPENMP)
// #    pragma omp for schedule(guided) nowait
// #endif
//             for (int b = 0; b < diagonal_dense_blocks.size(); b++) {
//                 int offset_i = diagonal_dense_blocks[b]->get_target_cluster_tree().get_offset();
//                 int offset_j = diagonal_dense_blocks[b]->get_source_cluster_tree().get_offset();

//                 diagonal_dense_blocks[b]->get_dense_data()->add_vector_product_symmetric(trans, alpha, in + offset_j - source_offset, local_beta, temp.data() + (offset_i - target_offset), m_block_tree_properties->m_UPLO, m_block_tree_properties->m_symmetry);
//             }
//         }

// #if defined(_OPENMP)
// #    pragma omp critical
// #endif

//         Blas<CoefficientPrecision>::axpy(&target_size, &da, temp.data(), &incx, out, &incy);
//     }
// }

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::threaded_hierarchical_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const {

    set_leaves_in_cache();

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
        htool::Logger::get_instance().log(Logger::LogLevel::ERROR, "Operation is not supported (" + std::string(1, trans) + " with " + m_symmetry_type_for_leaves + ")"); // LCOV_EXCL_LINE
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

// root_hmatrix = A*B (A.hmatrix_product(B))
// template <typename CoefficientPrecision, typename CoordinatePrecision>
// HMatrix<CoefficientPrecision, CoordinatePrecision> HMatrix<CoefficientPrecision, CoordinatePrecision>::hmatrix_product(const HMatrix &B) const {
//     HMatrix root_hmatrix(this->m_tree_data->m_target_cluster_tree, B.m_tree_data->m_source_cluster_tree);
//     root_hmatrix.set_admissibility_condition(this->m_tree_data->m_admissibility_condition);
//     root_hmatrix.set_low_rank_generator(this->m_tree_data->m_low_rank_generator);

//     SumExpression<CoefficientPrecision, CoordinatePrecision> root_sum_expression(this, &B);

//     root_hmatrix.recursive_build_hmatrix_product(root_sum_expression);

//     return root_hmatrix;
// }

////////////////////////
// Ca marche pas le coup du m_tree data a cause des sous _block
template <typename CoefficientPrecision, typename CoordinatePrecision>
HMatrix<CoefficientPrecision, CoordinatePrecision> HMatrix<CoefficientPrecision, CoordinatePrecision>::hmatrix_product(const HMatrix &B) const {
    // auto root_cluster_1 = std::make_shared<Cluster<CoordinatePrecision>>(this->get_target_cluster());
    // auto root_cluster_2 = std::make_shared<Cluster<CoordinatePrecision>>(B.get_source_cluster());
    // const Cluster<CoordinatePrecision> *const_cluster_ptr_t                        = &this->get_target_cluster();
    // std::shared_ptr<const Cluster<CoordinatePrecision>> shared_const_cluster_ptr_t = std::const_pointer_cast<const Cluster<CoordinatePrecision>>(std::shared_ptr<Cluster<CoordinatePrecision>>(const_cast<Cluster<CoordinatePrecision> *>(const_cluster_ptr_t)));
    // const Cluster<CoordinatePrecision> *const_cluster_ptr_s                        = &B.get_source_cluster();
    // std::shared_ptr<const Cluster<CoordinatePrecision>> shared_const_cluster_ptr_s = std::const_pointer_cast<const Cluster<CoordinatePrecision>>(std::shared_ptr<Cluster<CoordinatePrecision>>(const_cast<Cluster<CoordinatePrecision> *>(const_cluster_ptr_s)));
    // auto root_1         = const_cast<Cluster<CoordinatePrecision> *>(&this->get_target_cluster());
    // auto root_cluster_1 = std::make_shared<Cluster<CoordinatePrecision>>(*root_1);
    //  auto root_cluster_1 = std::make_shared<Cluster<CoordinatePrecision>>(std::move(this->get_target_cluster()));
    //  auto root_cluster_2 = std::make_shared<Cluster<CoordinatePrecision>>(std::move(B.get_source_cluster()));
    HMatrix root_hmatrix(this->m_tree_data->m_target_cluster_tree, B.m_tree_data->m_source_cluster_tree, this->m_target_cluster, &B.get_source_cluster());
    root_hmatrix.set_admissibility_condition(this->m_tree_data->m_admissibility_condition);
    root_hmatrix.set_low_rank_generator(this->m_tree_data->m_low_rank_generator);

    SumExpression<CoefficientPrecision, CoordinatePrecision> root_sum_expression(this, &B);

    root_hmatrix.recursive_build_hmatrix_product(root_sum_expression);

    return root_hmatrix;
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr) {
    // std::cout << "recursif build " << std::endl;
    // std::cout << "rec build" << std::endl;
    // std::cout << "appel de recursiv build sur " << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << std::endl;
    auto &target_cluster  = this->get_target_cluster();
    auto &source_cluster  = this->get_source_cluster();
    auto &target_children = target_cluster.get_children();
    auto &source_children = source_cluster.get_children();
    // if (sum_expr.get_sr().size() > 0) {
    //     std::cout << "ici   !!!!   " << sum_expr.get_sr()[0].nb_rows() << ',' << sum_expr.get_sr()[0].nb_cols() << ',' << sum_expr.get_sr()[1].nb_rows() << ',' << sum_expr.get_sr()[1].nb_cols() << std::endl;
    // }
    // critère pour descendre : on est sur une feuille ou pas:
    bool admissible    = this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, this->get_eta());
    auto test_restrict = sum_expr.is_restrictible();
    if (admissible) {
        // this->compute_dense_data(sum_expr);
        //  this->compute_low_rank_data(sum_expr, *this->get_lr(), this->get_rk(), this->get_epsilon());
        // auto &temp = *this;
        LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(sum_expr, *this->get_lr(), this->get_target_cluster(), this->get_source_cluster(), -1, this->get_epsilon());
        if ((lr.Get_U().nb_rows() == 0) or (lr.Get_U().nb_cols() == 0) or (lr.Get_V().nb_rows() == 0) or (lr.Get_V().nb_cols() == 0)) {
            this->compute_dense_data(sum_expr);
            // this->Plus(temp);

        } else {
            std::cout << "il y a du lr" << std::endl;
            this->set_low_rank_data(lr);
            // this->Plus(temp);
        }
    } else if ((target_children.size() == 0) and (source_children.size() == 0)) {
        this->compute_dense_data(sum_expr);
    } else {
        if ((target_children.size() > 0) and (source_children.size() > 0)) {
            if ((test_restrict[0] == 0) and (test_restrict[1] == 0)) {
                for (const auto &target_child : target_children) {
                    for (const auto &source_child : source_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), source_child.get());
                        SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                        hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                    }
                }
            } else {
                // std::cout << "????????????????????????" << std::endl;
                // std::cout << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << std::endl;
                this->compute_dense_data(sum_expr);
            }
        }

        else if ((target_children.size() == 0) and (source_children.size() > 0)) {
            if (test_restrict[1] == 0) {
                for (const auto &source_child : source_children) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(&target_cluster, source_child.get());
                    SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_cluster.get_size(), target_cluster.get_offset(), source_child->get_size(), source_child->get_offset());
                    hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                }
            } else {
                this->compute_dense_data(sum_expr);
            }
        } else if ((source_children.size() == 0) and (target_children.size() > 0)) {
            if (test_restrict[0] == 0) {
                for (const auto &target_child : target_children) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), &source_cluster);
                    SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_child->get_size(), target_child->get_offset(), source_cluster.get_size(), source_cluster.get_offset());
                    hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                }
            } else {
                this->compute_dense_data(sum_expr);
            }
        }
    }
}

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr) {
//     // std::cout << "recursif build " << std::endl;
//     // std::cout << "rec build" << std::endl;
//     // std::cout << "appel de recursiv build sur " << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << std::endl;
//     auto &target_cluster  = this->get_target_cluster();
//     auto &source_cluster  = this->get_source_cluster();
//     auto &target_children = target_cluster.get_children();
//     auto &source_children = source_cluster.get_children();
//     // if (sum_expr.get_sr().size() > 0) {
//     //     std::cout << "ici   !!!!   " << sum_expr.get_sr()[0].nb_rows() << ',' << sum_expr.get_sr()[0].nb_cols() << ',' << sum_expr.get_sr()[1].nb_rows() << ',' << sum_expr.get_sr()[1].nb_cols() << std::endl;
//     // }
//     // critère pour descendre : on est sur une feuille ou pas:
//     bool admissible    = this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, this->get_eta());
//     auto test_restrict = sum_expr.is_restrictible();
//     if (admissible) {
//         // this->compute_dense_data(sum_expr);
//         //  this->compute_low_rank_data(sum_expr, *this->get_lr(), this->get_rk(), this->get_epsilon());
//         LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(sum_expr, *this->get_lr(), this->get_target_cluster(), this->get_source_cluster(), -1, this->get_epsilon());
//         if ((lr.Get_U().nb_rows() == 0) or (lr.Get_U().nb_cols() == 0) or (lr.Get_V().nb_rows() == 0) or (lr.Get_V().nb_cols() == 0)) {
//             this->compute_dense_data(sum_expr);

//         } else {
//             std::cout << "il y a du lr" << std::endl;
//             this->set_low_rank_data(lr);
//         }
//     } else if ((target_children.size() == 0) and (source_children.size() == 0)) {
//         this->compute_dense_data(sum_expr);
//     } else {
//         if ((target_children.size() > 0) and (source_children.size() > 0)) {
//             if ((test_restrict[0] == 0) and (test_restrict[1] == 0)) {
//                 for (const auto &target_child : target_children) {
//                     for (const auto &source_child : source_children) {
//                         HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), source_child.get());
//                         SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
//                         hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//                     }
//                 }
//             } else {
//                 // std::cout << "????????????????????????" << std::endl;
//                 // std::cout << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << std::endl;
//                 this->compute_dense_data(sum_expr);
//             }
//         }

//         else if ((target_children.size() == 0) and (source_children.size() > 0)) {
//             if (test_restrict[1] == 0) {
//                 for (const auto &source_child : source_children) {
//                     HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(&target_cluster, source_child.get());
//                     SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_cluster.get_size(), target_cluster.get_offset(), source_child->get_size(), source_child->get_offset());
//                     hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//                 }
//             } else {
//                 this->compute_dense_data(sum_expr);
//             }
//         } else if ((source_children.size() == 0) and (target_children.size() > 0)) {
//             if (test_restrict[0] == 0) {
//                 for (const auto &target_child : target_children) {
//                     HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), &source_cluster);
//                     SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_child->get_size(), target_child->get_offset(), source_cluster.get_size(), source_cluster.get_offset());
//                     hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//                 }
//             } else {
//                 this->compute_dense_data(sum_expr);
//             }
//         }
//     }
// }

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product_new(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr) {
    auto &target_cluster  = this->get_target_cluster();
    auto &source_cluster  = this->get_source_cluster();
    auto &target_children = target_cluster.get_children();
    auto &source_children = source_cluster.get_children();
    bool admissible       = this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, 1);
    // auto test_restrict    = sum_expr.is_restrictible();
    if (admissible) {
        this->compute_dense_data(sum_expr);
        //  this->compute_low_rank_data(sum_expr, *this->get_lr(), this->get_rk(), this->get_epsilon());
    } else if ((target_children.size() == 0) and (source_children.size() == 0)) {
        this->compute_dense_data(sum_expr);
    } else {
        if ((target_children.size() > 0) and (source_children.size() > 0)) {
            // if ((test_restrict[0] == 0) and (test_restrict[1] == 0)) {
            for (const auto &target_child : target_children) {
                for (const auto &source_child : source_children) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), source_child.get());
                    SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_new(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                    hmatrix_child->recursive_build_hmatrix_product_new(sum_restr);
                }
            }
            // } else {
            //     // this->compute_dense_data(sum_expr);
            // }
        }

        else if ((target_children.size() == 0) and (source_children.size() > 0)) {
            // if (test_restrict[1] == 0) {
            for (const auto &source_child : source_children) {
                HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(&target_cluster, source_child.get());
                SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_new(target_cluster.get_size(), target_cluster.get_offset(), source_child->get_size(), source_child->get_offset());
                hmatrix_child->recursive_build_hmatrix_product_new(sum_restr);
            }
            // } else {
            //     // this->compute_dense_data(sum_expr);
            // }
        } else if ((source_children.size() == 0) and (target_children.size() > 0)) {
            // if (test_restrict[0] == 0) {
            for (const auto &target_child : target_children) {
                HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), &source_cluster);
                SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_new(target_child->get_size(), target_child->get_offset(), source_cluster.get_size(), source_cluster.get_offset());
                hmatrix_child->recursive_build_hmatrix_product_new(sum_restr);
            }
            // } else {
            //     // this->compute_dense_data(sum_expr);
            // }
        }
    }
}

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr) {
//     std::cout << "recursif build " << std::endl;
//     const auto &target_cluster  = this->get_target_cluster();
//     const auto &source_cluster  = this->get_source_cluster();
//     const auto &target_children = target_cluster.get_children();
//     const auto &source_children = source_cluster.get_children();
//     // Récursion sur les fl des clusters : 3 cas
//     // if (!target_cluster.is_leaf() && !source_cluster.is_leaf()) {
//     if ((target_children.size() > 0) and (source_children.size() > 0)) {
//         std::cout << "H mat Hmat " << std::endl;
//         for (const auto &target_child : target_children) {
//             for (const auto &source_child : source_children) {
//                 // std::cout << "taille de sourcechild" << source_child->get_size() << ',' << source_child->get_offset() << std::endl;
//                 HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(target_child.get(), source_child.get());
//                 // std::cout << "les parents on des enfants ?  target " << target_cluster.get_children().size() << "source" << source_cluster.get_children().size() << std::endl;
//                 // std::cout << "appel de restrict avec target =  " << target_child->get_size() << ',' << target_child->get_offset() << " et source = " << source_child->get_size() << ',' << source_child->get_offset() << std::endl;
//                 SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_child->get_size(), source_child->get_size(), target_child->get_offset(), source_child->get_offset());
//                 // std::cout << "restrict ok " << std::endl;
//                 //  sum_restr.Sort();
//                 //  std::cout << "sort ok " << std::endl;
//                 hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//                 // std::cout << "récursiv build ok " << std::endl;
//                 //  call recursive avec srestr et Lk
//             }
//         }
//     }
//     // else if (target_cluster.is_leaf()) {
//     else if (target_children.size() == 0) {
//         std::cout << "cas 2 " << std::endl;
//         for (const auto &source_child : source_children) {
//             HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(&target_cluster, source_child.get());
//             SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_cluster.get_size(), source_child->get_size(), target_cluster.get_offset(), source_child->get_offset());
//             // sum_restr.Sort();
//             hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//             // call recursive avec srestr et Lk
//         }

//     }
//     // else if (source_cluster.is_leaf()) {
//     else if (source_children.size() == 0) {
//         for (const auto &target_child : target_children) {
//             HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), &source_cluster);
//             SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_child->get_size(), source_cluster.get_size(), target_child->get_offset(), source_cluster.get_offset());
//             // sum_restr.Sort();
//             hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//             // call recursive avec srestr et Lk
//         }
//     } else {
//         std::cout << "feuille mais bon_______________________________________________________ " << std::endl;
//         // feuilles : 2 cas
//         // low rank
//         if (this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, this->m_tree_data->m_eta)) {
//             std::cout << "feuille admissible" << std::endl; // admissibility{
//             this->compute_low_rank_data(sum_expr, *(this->m_tree_data->m_low_rank_generator.get()), this->m_tree_data->m_reqrank, this->m_tree_data->m_epsilon);
//             auto lr = this->get_low_rank_data();
//             auto U  = lr->Get_U();
//             auto V  = lr->Get_V();
//             std::cout << "U " << U.nb_rows() << ',' << U.nb_cols() << " V " << V.nb_rows() << ',' << V.nb_cols() << std::endl;
//         }
//         // dense
//         else {
//             std::cout << "feuille dense" << std::endl;
//             this->compute_dense_data(sum_expr); // normalement comme sum_expr hérite de virul generator ca devrit le faire
//         }
//     }
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr) {
//     std::cout << "recursif build " << std::endl;
//     auto &target_cluster  = this->get_target_cluster();
//     auto &source_cluster  = this->get_source_cluster();
//     auto &target_children = target_cluster.get_children();
//     auto &source_children = source_cluster.get_children();
//     if ((target_children.size() > 0) and (source_children.size() > 0)) {
//         std::cout << "H mat Hmat " << std::endl;
//         for (const auto &target_child : target_children) {
//             for (const auto &source_child : source_children) {
//                 HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), source_child.get());
//                 SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_child->get_size(), source_child->get_size(), target_child->get_offset(), source_child->get_offset());
//                 hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//             }
//         }
//     }
//     // else if (target_cluster.is_leaf()) {
//     else if ((target_children.size() == 0) and (source_children.size() > 0)) {
//         std::cout << "cas 2 " << std::endl;
//         for (const auto &source_child : source_children) {
//             HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(&target_cluster, source_child.get());
//             SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_cluster.get_size(), source_child->get_size(), target_cluster.get_offset(), source_child->get_offset());
//             // sum_restr.Sort();
//             hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//             // call recursive avec srestr et Lk
//         }

//     }
//     // else if (source_cluster.is_leaf()) {
//     else if ((source_children.size() == 0) and (target_children.size() > 0)) {
//         for (const auto &target_child : target_children) {
//             HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), &source_cluster);
//             SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_child->get_size(), source_cluster.get_size(), target_child->get_offset(), source_cluster.get_offset());
//             // sum_restr.Sort();
//             hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//             // call recursive avec srestr et Lk
//         }
//     } else {
//         bool admissible = 2 * std::min(target_cluster.get_radius(), source_cluster.get_radius()) < this->m_tree_data->m_eta * std::max((norm2(target_cluster.get_center() - source_cluster.get_center()) - target_cluster.get_radius() - source_cluster.get_radius()), 0.);
//         std::cout << "il a calculé le critère" << std::endl;
//         if (admissible) {
//             std::cout << "feuille admissible" << std::endl;
//             // std::cout << sum_expr.get_sr().size() << ',' << sum_expr.get_sh().size() << std::endl;
//             LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_mat(sum_expr, *this->m_tree_data->m_low_rank_generator.get(), target_cluster, source_cluster);
//             std::cout << "'?? " << std::endl;
//             this->compute_low_rank_data(sum_expr, *this->m_tree_data->m_low_rank_generator.get(), this->m_tree_data->m_reqrank, this->m_tree_data->m_epsilon);
//             std::cout << "low rank ok " << std::endl;
//             // auto lr = this->get_low_rank_data();
//             // auto U  = lr->Get_U();
//             // auto V  = lr->Get_V();
//             // std::cout << "U " << U.nb_rows() << ',' << U.nb_cols() << " V " << V.nb_rows() << ',' << V.nb_cols() << std::endl;

//         }
//         // dense
//         else {
//             std::cout << "feuille dense" << std::endl;
//             this->compute_dense_data(sum_expr); // normalement comme sum_expr hérite de virul generator ca devrit le faire
//         }
//     }
// }
// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr) {
//     std::cout << "recursif build " << std::endl;
//     const auto &target_cluster  = this->get_target_cluster();
//     const auto &source_cluster  = this->get_source_cluster();
//     const auto &target_children = target_cluster.get_children();
//     const auto &source_children = source_cluster.get_children();
//     // Récursion sur les fl des clusters : 3 cas
//     // if (!target_cluster.is_leaf() && !source_cluster.is_leaf()) {
//     if ((target_children.size() > 0) and (source_children.size() > 0)) {
//         std::cout << "H mat Hmat " << std::endl;
//         for (const auto &target_child : target_children) {
//             for (const auto &source_child : source_children) {
//                 // std::cout << "taille de sourcechild" << source_child->get_size() << ',' << source_child->get_offset() << std::endl;
//                 HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(target_child.get(), source_child.get());
//                 // std::cout << "les parents on des enfants ?  target " << target_cluster.get_children().size() << "source" << source_cluster.get_children().size() << std::endl;
//                 // std::cout << "appel de restrict avec target =  " << target_child->get_size() << ',' << target_child->get_offset() << " et source = " << source_child->get_size() << ',' << source_child->get_offset() << std::endl;
//                 SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_child->get_size(), source_child->get_size(), target_child->get_offset(), source_child->get_offset());
//                 // std::cout << "restrict ok " << std::endl;
//                 //  sum_restr.Sort();
//                 //  std::cout << "sort ok " << std::endl;
//                 hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//                 // std::cout << "récursiv build ok " << std::endl;
//                 //  call recursive avec srestr et Lk
//             }
//         }
//     }
//     // else if (target_cluster.is_leaf()) {
//     else if ((target_children.size() == 0) and (source_children.size() > 0)) {
//         std::cout << "cas 2 " << std::endl;
//         for (const auto &source_child : source_children) {
//             HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(&target_cluster, source_child.get());
//             SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_cluster.get_size(), source_child->get_size(), target_cluster.get_offset(), source_child->get_offset());
//             // sum_restr.Sort();
//             hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//             // call recursive avec srestr et Lk
//         }

//     }
//     // else if (source_cluster.is_leaf()) {
//     else if ((source_children.size() == 0) and (target_children.size() > 0)) {
//         for (const auto &target_child : target_children) {
//             HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), &source_cluster);
//             SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(target_child->get_size(), source_cluster.get_size(), target_child->get_offset(), source_cluster.get_offset());
//             // sum_restr.Sort();
//             hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//             // call recursive avec srestr et Lk
//         }
//     } else {
//         std::cout << "feuille mais bon_______________________________________________________ " << std::endl;
//         // feuilles : 2 cas
//         // low rank
//         // auto &lr_crit = this->m_tree_data->m_admissibility_condition;
//         // std::cout << this->m_tree_data->m_eta << std::endl;
//         // auto admissible = (lr_crit.get())->ComputeAdmissibility(target_cluster, source_cluster, this->m_tree_data->m_eta);
//         // std::cout << "hey" << std::endl;
//         // bool adm = this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, this->m_tree_data->m_eta);
//         // bool adm = this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, this->m_tree_data->m_eta);
//         bool admissible = 2 * std::min(target_cluster.get_radius(), source_cluster.get_radius()) < this->m_tree_data->m_eta * std::max((norm2(target_cluster.get_center() - source_cluster.get_center()) - target_cluster.get_radius() - source_cluster.get_radius()), 0.);
//         std::cout << "il a calculé le critère" << std::endl;
//         if (admissible) {
//             std::cout << "feuille admissible" << std::endl;
//             std::cout << sum_expr.get_sr().size() << ',' << sum_expr.get_sh().size() << std::endl;
//             LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_mat(sum_expr, *this->m_tree_data->m_low_rank_generator.get(), target_cluster, source_cluster);
//             std::cout << "'?? " << std::endl;
//             this->compute_low_rank_data(sum_expr, *this->m_tree_data->m_low_rank_generator.get(), this->m_tree_data->m_reqrank, this->m_tree_data->m_epsilon);
//             std::cout << "low rank ok " << std::endl;
//             // auto lr = this->get_low_rank_data();
//             // auto U  = lr->Get_U();
//             // auto V  = lr->Get_V();
//             // std::cout << "U " << U.nb_rows() << ',' << U.nb_cols() << " V " << V.nb_rows() << ',' << V.nb_cols() << std::endl;

//         }
//         // dense
//         else {
//             std::cout << "feuille dense" << std::endl;
//             this->compute_dense_data(sum_expr); // normalement comme sum_expr hérite de virul generator ca devrit le faire
//         }
//     }
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mymvprod_transp_local_to_global(const T *const in, T *const out, const int &mu) const {
//     std::fill(out, out + this->nc * mu, 0);
//     int incx(1), incy(1);
//     int global_size_rhs = this->nc * mu;
//     T da(1);

//     // Contribution champ lointain
// #if defined(_OPENMP)
// #    pragma omp parallel
// #endif
//     {
//         std::vector<CoefficientPrecision> temp(this->nc * mu, 0);
// #if defined(_OPENMP)
// #    pragma omp for schedule(guided) nowait
// #endif
//         for (int b = 0; b < MyComputedBlocks.size(); b++) {
//             int offset_i = MyComputedBlocks[b]->get_target_cluster()->get_offset();
//             int offset_j = MyComputedBlocks[b]->get_source_cluster()->get_offset();
//             if (!(m_symmetry != 'N') || offset_i != offset_j) { // remove strictly diagonal blocks
//                 MyComputedBlocks[b]->get_block_data()->add_mvprod_row_major(in + (offset_i - local_offset) * mu, temp.data() + offset_j * mu, mu, 'T', 'T');
//             }
//         }

// #if defined(_OPENMP)
// #    pragma omp critical
// #endif
//         Blas<CoefficientPrecision>::axpy(&(global_size_rhs), &da, temp.data(), &incx, out, &incy);
//     }

//     MPI_Allreduce(MPI_IN_PLACE, out, this->nc * mu, wrapper_mpi<CoefficientPrecision>::mpi_type(), MPI_SUM, comm);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::local_to_global_target(const T *const in, T *const out, const int &mu) const {
//     // Allgather
//     std::vector<int> recvcounts(sizeWorld);
//     std::vector<int> displs(sizeWorld);

//     displs[0] = 0;

//     for (int i = 0; i < sizeWorld; i++) {
//         recvcounts[i] = (cluster_tree_t->get_masteroffset(i).second) * mu;
//         if (i > 0)
//             displs[i] = displs[i - 1] + recvcounts[i - 1];
//     }

//     MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::local_to_global_source(const T *const in, T *const out, const int &mu) const {
//     // Allgather
//     std::vector<int> recvcounts(sizeWorld);
//     std::vector<int> displs(sizeWorld);

//     displs[0] = 0;

//     for (int i = 0; i < sizeWorld; i++) {
//         recvcounts[i] = (cluster_tree_s->get_masteroffset(i).second) * mu;
//         if (i > 0)
//             displs[i] = displs[i - 1] + recvcounts[i - 1];
//     }
//     MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mymvprod_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
//     double time      = MPI_Wtime();
//     bool need_delete = false;
//     if (work == nullptr) {
//         work        = new T[this->nc * mu];
//         need_delete = true;
//     }
//     this->local_to_global_source(in, work, mu);
//     this->mymvprod_global_to_local(work, out, mu);

//     if (need_delete) {
//         delete[] work;
//         work = nullptr;
//     }
//     infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
//     infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mymvprod_transp_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
//     int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;

//     if (this->m_symmetry == 'S' || this->m_symmetry == 'H') {
//         this->mymvprod_local_to_local(in, out, mu, work);
//         return;
//     }

//     double time      = MPI_Wtime();
//     bool need_delete = false;
//     if (work == nullptr) {
//         work        = new T[(this->nc + local_size_source * sizeWorld) * mu];
//         need_delete = true;
//     }

//     std::fill(out, out + local_size_source * mu, 0);
//     int incx(1), incy(1);
//     int global_size_rhs = this->nc * mu;
//     T da(1);

//     std::fill(work, work + this->nc * mu, 0);
//     T *rbuf = work + this->nc * mu;

//     // Contribution champ lointain
// #if defined(_OPENMP)
// #    pragma omp parallel
// #endif
//     {
//         std::vector<CoefficientPrecision> temp(this->nc * mu, 0);
// #if defined(_OPENMP)
// #    pragma omp for schedule(guided) nowait
// #endif
//         for (int b = 0; b < MyComputedBlocks.size(); b++) {
//             int offset_i = MyComputedBlocks[b]->get_target_cluster()->get_offset();
//             int offset_j = MyComputedBlocks[b]->get_source_cluster()->get_offset();
//             if (!(m_symmetry != 'N') || offset_i != offset_j) { // remove strictly diagonal blocks
//                 MyComputedBlocks[b]->get_block_data()->add_mvprod_row_major(in + (offset_i - local_offset) * mu, temp.data() + offset_j * mu, mu, 'T', 'T');
//             }
//         }

// #if defined(_OPENMP)
// #    pragma omp critical
// #endif
//         Blas<CoefficientPrecision>::axpy(&(global_size_rhs), &da, temp.data(), &incx, work, &incy);
//     }

//     std::vector<int> scounts(sizeWorld), rcounts(sizeWorld);
//     std::vector<int> sdispls(sizeWorld), rdispls(sizeWorld);

//     sdispls[0] = 0;
//     rdispls[0] = 0;

//     for (int i = 0; i < sizeWorld; i++) {
//         scounts[i] = (cluster_tree_s->get_masteroffset(i).second) * mu;
//         rcounts[i] = (local_size_source)*mu;
//         if (i > 0) {
//             sdispls[i] = sdispls[i - 1] + scounts[i - 1];
//             rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
//         }
//     }

//     MPI_Alltoallv(work, &(scounts[0]), &(sdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), rbuf, &(rcounts[0]), &(rdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);

//     for (int i = 0; i < sizeWorld; i++)
//         std::transform(out, out + local_size_source * mu, rbuf + rdispls[i], out, std::plus<CoefficientPrecision>());

//     if (need_delete) {
//         delete[] work;
//         work = nullptr;
//     }
//     infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
//     infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mvprod_global_to_global(const T *const in, T *const out, const int &mu) const {
//     double time = MPI_Wtime();

//     if (mu == 1) {
//         std::vector<CoefficientPrecision> out_perm(local_size);
//         std::vector<CoefficientPrecision> buffer(std::max(nc, nr));

//         // Permutation
//         if (use_permutation) {
//             this->source_to_cluster_permutation(in, buffer.data());
//             mymvprod_global_to_local(buffer.data(), out_perm.data(), 1);

//         } else {
//             mymvprod_global_to_local(in, out_perm.data(), 1);
//         }

//         // Allgather
//         std::vector<int> recvcounts(sizeWorld);
//         std::vector<int> displs(sizeWorld);

//         displs[0] = 0;

//         for (int i = 0; i < sizeWorld; i++) {
//             recvcounts[i] = cluster_tree_t->get_masteroffset(i).second * mu;
//             if (i > 0)
//                 displs[i] = displs[i - 1] + recvcounts[i - 1];
//         }

//         if (use_permutation) {
//             MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), buffer.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);

//             // Permutation
//             this->cluster_to_target_permutation(buffer.data(), out);
//         } else {
//             MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
//         }

//     } else {

//         std::vector<CoefficientPrecision> in_perm(std::max(nr, nc) * mu * 2);
//         std::vector<CoefficientPrecision> out_perm(local_size * mu);
//         std::vector<CoefficientPrecision> buffer(nc);

//         for (int i = 0; i < mu; i++) {
//             // Permutation
//             if (use_permutation) {
//                 this->source_to_cluster_permutation(in + i * nc, buffer.data());
//                 // Transpose
//                 for (int j = 0; j < nc; j++) {
//                     in_perm[i + j * mu] = buffer[j];
//                 }
//             } else {
//                 // Transpose
//                 for (int j = 0; j < nc; j++) {
//                     in_perm[i + j * mu] = in[j + i * nc];
//                 }
//             }
//         }

//         if (m_symmetry == 'H') {
//             conj_if_complex(in_perm.data(), nc * mu);
//         }

//         mymvprod_global_to_local(in_perm.data(), in_perm.data() + nc * mu, mu);

//         // Tranpose
//         for (int i = 0; i < mu; i++) {
//             for (int j = 0; j < local_size; j++) {
//                 out_perm[i * local_size + j] = in_perm[i + j * mu + nc * mu];
//             }
//         }

//         if (m_symmetry == 'H') {
//             conj_if_complex(out_perm.data(), out_perm.size());
//         }

//         // Allgather
//         std::vector<int> recvcounts(sizeWorld);
//         std::vector<int> displs(sizeWorld);

//         displs[0] = 0;

//         for (int i = 0; i < sizeWorld; i++) {
//             recvcounts[i] = cluster_tree_t->get_masteroffset(i).second * mu;
//             if (i > 0)
//                 displs[i] = displs[i - 1] + recvcounts[i - 1];
//         }

//         MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), in_perm.data() + mu * nr, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);

//         for (int i = 0; i < mu; i++) {
//             if (use_permutation) {
//                 for (int j = 0; j < sizeWorld; j++) {
//                     std::copy_n(in_perm.data() + mu * nr + displs[j] + i * recvcounts[j] / mu, recvcounts[j] / mu, in_perm.data() + i * nr + displs[j] / mu);
//                 }

//                 // Permutation
//                 this->cluster_to_target_permutation(in_perm.data() + i * nr, out + i * nr);
//             } else {
//                 for (int j = 0; j < sizeWorld; j++) {
//                     std::copy_n(in_perm.data() + mu * nr + displs[j] + i * recvcounts[j] / mu, recvcounts[j] / mu, out + i * nr + displs[j] / mu);
//                 }
//             }
//         }
//     }
//     // Timing
//     infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
//     infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mvprod_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
//     double time      = MPI_Wtime();
//     bool need_delete = false;
//     if (work == nullptr) {
//         work        = new T[this->nc * mu];
//         need_delete = true;
//     }

//     int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;

//     if (!(cluster_tree_s->IsLocal()) || !(cluster_tree_t->IsLocal())) {
//         throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
//     }
//     if (mu == 1) {
//         std::vector<CoefficientPrecision> in_perm(local_size_source), out_perm(local_size);

//         // local permutation
//         if (use_permutation) {
//             // permutation
//             this->local_source_to_local_cluster(in, in_perm.data());

//             // prod
//             mymvprod_local_to_local(in_perm.data(), out_perm.data(), 1, work);

//             // permutation
//             this->local_cluster_to_local_target(out_perm.data(), out, comm);

//         } else {
//             mymvprod_local_to_local(in, out, 1, work);
//         }

//     } else {

//         std::vector<CoefficientPrecision> in_perm(local_size_source * mu);
//         std::vector<CoefficientPrecision> out_perm(local_size * mu);
//         std::vector<CoefficientPrecision> buffer(std::max(local_size_source, local_size));

//         for (int i = 0; i < mu; i++) {
//             // local permutation
//             if (use_permutation) {
//                 this->local_source_to_local_cluster(in + i * local_size_source, buffer.data());

//                 // Transpose
//                 for (int j = 0; j < local_size_source; j++) {
//                     in_perm[i + j * mu] = buffer[j];
//                 }
//             } else {
//                 // Transpose
//                 for (int j = 0; j < local_size_source; j++) {
//                     in_perm[i + j * mu] = in[j + i * local_size_source];
//                 }
//             }
//         }

//         if (m_symmetry == 'H') {
//             conj_if_complex(in_perm.data(), local_size_source * mu);
//         }

//         mymvprod_local_to_local(in_perm.data(), out_perm.data(), mu, work);

//         for (int i = 0; i < mu; i++) {
//             if (use_permutation) {
//                 // Tranpose
//                 for (int j = 0; j < local_size; j++) {
//                     buffer[j] = out_perm[i + j * mu];
//                 }

//                 // local permutation
//                 this->local_cluster_to_local_target(buffer.data(), out + i * local_size);
//             } else {
//                 // Tranpose
//                 for (int j = 0; j < local_size; j++) {
//                     out[j + i * local_size] = out_perm[i + j * mu];
//                 }
//             }
//         }

//         if (m_symmetry == 'H') {
//             conj_if_complex(out, out_perm.size());
//         }
//     }

//     if (need_delete) {
//         delete[] work;
//         work = nullptr;
//     }
//     // Timing
//     infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
//     infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mvprod_transp_global_to_global(const T *const in, T *const out, const int &mu) const {
//     double time = MPI_Wtime();
//     if (this->m_symmetry == 'S') {
//         this->mvprod_global_to_global(in, out, mu);
//         return;
//     } else if (this->m_symmetry == 'H') {
//         std::vector<CoefficientPrecision> in_conj(in, in + nr * mu);
//         conj_if_complex(in_conj.data(), nr * mu);
//         this->mvprod_global_to_global(in_conj.data(), out, mu);
//         conj_if_complex(out, mu * nc);
//         return;
//     }
//     if (mu == 1) {

//         if (use_permutation) {
//             std::vector<CoefficientPrecision> in_perm(nr), out_perm(nc);

//             // permutation
//             this->target_to_cluster_permutation(in, in_perm.data());

//             mymvprod_transp_local_to_global(in_perm.data() + local_offset, out_perm.data(), 1);

//             // permutation
//             this->cluster_to_source_permutation(out_perm.data(), out);
//         } else {
//             mymvprod_transp_local_to_global(in + local_offset, out, 1);
//         }

//     } else {

//         std::vector<CoefficientPrecision> out_perm(mu * nc);
//         std::vector<CoefficientPrecision> in_perm(local_size * mu + mu * nc);
//         std::vector<CoefficientPrecision> buffer(nr);

//         for (int i = 0; i < mu; i++) {
//             // Permutation
//             if (use_permutation) {
//                 this->target_to_cluster_permutation(in + i * nr, buffer.data());
//                 // Transpose
//                 for (int j = local_offset; j < local_offset + local_size; j++) {
//                     in_perm[i + (j - local_offset) * mu] = buffer[j];
//                 }
//             } else {
//                 // Transpose
//                 for (int j = local_offset; j < local_offset + local_size; j++) {
//                     in_perm[i + (j - local_offset) * mu] = in[j + i * nr];
//                 }
//             }
//         }

//         // It should never happen since we use mvprod_global_to_global in this case
//         // if (m_symmetry == 'H') {
//         //     conj_if_complex(in_perm.data(), local_size * mu);
//         // }

//         mymvprod_transp_local_to_global(in_perm.data(), in_perm.data() + local_size * mu, mu);

//         for (int i = 0; i < mu; i++) {
//             if (use_permutation) {
//                 // Transpose
//                 for (int j = 0; j < nc; j++) {
//                     out_perm[i * nc + j] = in_perm[i + j * mu + local_size * mu];
//                 }
//                 cluster_to_source_permutation(out_perm.data() + i * nc, out + i * nc);
//             } else {
//                 for (int j = 0; j < nc; j++) {
//                     out[i * nc + j] = in_perm[i + j * mu + local_size * mu];
//                 }
//             }
//         }

//         // It should never happen since we use mvprod_global_to_global in this case
//         // if (m_symmetry == 'H') {
//         //     conj_if_complex(out, nc * mu);
//         // }
//     }
//     // Timing
//     infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
//     infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mvprod_transp_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
//     double time           = MPI_Wtime();
//     int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;
//     if (this->m_symmetry == 'S') {
//         this->mvprod_local_to_local(in, out, mu);
//         return;
//     } else if (this->m_symmetry == 'H') {
//         std::vector<CoefficientPrecision> in_conj(in, in + local_size * mu);
//         conj_if_complex(in_conj.data(), local_size * mu);
//         this->mvprod_local_to_local(in_conj.data(), out, mu);
//         conj_if_complex(out, mu * local_size_source);
//         return;
//     }
//     bool need_delete = false;
//     if (work == nullptr) {
//         work        = new T[(this->nc + sizeWorld * this->get_source_cluster()->get_local_size()) * mu];
//         need_delete = true;
//     }

//     if (!(cluster_tree_s->IsLocal()) || !(cluster_tree_t->IsLocal())) {
//         throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
//     }

//     if (mu == 1) {
//         std::vector<CoefficientPrecision> in_perm(local_size), out_perm(local_size_source);

//         // local permutation
//         if (use_permutation) {
//             this->local_target_to_local_cluster(in, in_perm.data());

//             // prod
//             mymvprod_transp_local_to_local(in_perm.data(), out_perm.data(), 1, work);

//             // permutation
//             this->local_cluster_to_local_source(out_perm.data(), out, comm);

//         } else {
//             mymvprod_transp_local_to_local(in, out, 1, work);
//         }

//     } else {

//         std::vector<CoefficientPrecision> in_perm(local_size * mu);
//         std::vector<CoefficientPrecision> out_perm(local_size_source * mu);
//         std::vector<CoefficientPrecision> buffer(std::max(local_size_source, local_size));

//         for (int i = 0; i < mu; i++) {
//             // local permutation
//             if (use_permutation) {
//                 this->local_target_to_local_cluster(in + i * local_size, buffer.data());

//                 // Transpose
//                 for (int j = 0; j < local_size; j++) {
//                     in_perm[i + j * mu] = buffer[j];
//                 }
//             } else {
//                 // Transpose
//                 for (int j = 0; j < local_size; j++) {
//                     in_perm[i + j * mu] = in[j + i * local_size];
//                 }
//             }
//         }
//         // It should never happen since we use mvprod_global_to_global in this case
//          if (m_symmetry == 'H') {
//            conj_if_complex(in_perm.data(), local_size_source * mu);
//         }

//         mymvprod_transp_local_to_local(in_perm.data(), out_perm.data(), mu, work);

//         for (int i = 0; i < mu; i++) {
//             if (use_permutation) {
//                 // Tranpose
//                 for (int j = 0; j < local_size_source; j++) {
//                     buffer[j] = out_perm[i + j * mu];
//                 }

//                 // local permutation
//                 this->local_cluster_to_local_source(buffer.data(), out + i * local_size_source);
//             } else {
//                 // Tranpose
//                 for (int j = 0; j < local_size_source; j++) {
//                     out[j + i * local_size_source] = out_perm[i + j * mu];
//                 }
//             }
//         }
//         // It should never happen since we use mvprod_global_to_global in this case
//         if (m_symmetry == 'H') {
//            conj_if_complex(out, out_perm.size());
//     }
//     }

//     if (need_delete) {
//         delete[] work;
//         work = nullptr;
//     }
//     // Timing
//     infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
//     infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mvprod_subrhs(const T *const in, T *const out, const int &mu, const int &offset, const int &size, const int &margin) const {
//     std::fill(out, out + local_size * mu, 0);

//     // Contribution champ lointain
// #if defined(_OPENMP)
// #    pragma omp parallel
// #endif
//     {
//         std::vector<CoefficientPrecision> temp(local_size * mu, 0);
//         // To localize the rhs with multiple rhs, it is transpose. So instead of A*B, we do transpose(B)*transpose(A)
//         char transb = 'T';
//         // In case of a hermitian matrix, the rhs is conjugate transpose
//         if (m_symmetry == 'H') {
//             transb = 'C';
//         }
// #if defined(_OPENMP)
// #    pragma omp for schedule(guided)
// #endif
//         for (int b = 0; b < MyComputedBlocks.size(); b++) {
//             int offset_i = MyComputedBlocks[b]->get_target_cluster()->get_offset();
//             int offset_j = MyComputedBlocks[b]->get_source_cluster()->get_offset();
//             int size_j   = MyComputedBlocks[b]->get_source_cluster()->get_size();

//             if ((offset_j < offset + size && offset < offset_j + size_j) && (m_symmetry == 'N' || offset_i != offset_j)) {
//                 if (offset_j - offset < 0) {
//                     std::cout << "TEST "
//                               << " " << offset_j << " " << size_j << " "
//                               << " " << offset << " " << size << " "
//                               << " " << rankWorld << " "
//                               << offset_j - offset << std::endl;
//                 }
//                 MyComputedBlocks[b]->get_block_data()->add_mvprod_row_major(in + (offset_j - offset + margin) * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb);
//             }
//         }

//         // Symmetry part of the diagonal part
//         if (m_symmetry != 'N') {
//             transb      = 'N';
//             char op_sym = 'T';
//             if (m_symmetry == 'H') {
//                 op_sym = 'C';
//             }
// #if defined(_OPENMP)
// #    pragma omp for schedule(guided)
// #endif
//             for (int b = 0; b < MyDiagComputedBlocks.size(); b++) {
//                 int offset_i = MyDiagComputedBlocks[b]->get_source_cluster()->get_offset();
//                 int offset_j = MyDiagComputedBlocks[b]->get_target_cluster()->get_offset();
//                 int size_j   = MyDiagComputedBlocks[b]->get_target_cluster()->get_size();

//                 if ((offset_j < offset + size && offset < offset_j + size_j) && offset_i != offset_j) { // remove strictly diagonal blocks
//                     MyDiagComputedBlocks[b]->get_block_data()->add_mvprod_row_major(in + (offset_j - offset + margin) * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb, op_sym);
//                 }
//             }

// #if defined(_OPENMP)
// #    pragma omp for schedule(guided)
// #endif
//             for (int b = 0; b < MyStrictlyDiagNearFieldMats.size(); b++) {
//                 const Block<CoefficientPrecision> *M = MyStrictlyDiagNearFieldMats[b];
//                 int offset_i      = M->get_source_cluster()->get_offset();
//                 int offset_j      = M->get_target_cluster()->get_offset();
//                 int size_j        = M->get_source_cluster()->get_size();
//                 if (offset_j < offset + size && offset < offset_j + size_j) {
//                     M->get_dense_block_data()->add_mvprod_row_major_sym(in + (offset_j - offset + margin) * mu, temp.data() + (offset_i - local_offset) * mu, mu, this->UPLO, this->symmetry);
//                 }
//             }
//         }
// #if defined(_OPENMP)
// #    pragma omp critical
// #endif
//         std::transform(temp.begin(), temp.end(), out, out, std::plus<CoefficientPrecision>());
//     }

//     if (!(this->cluster_tree_s->get_local_offset() < offset + size && offset < this->cluster_tree_s->get_local_offset() + this->cluster_tree_s->get_local_size()) && this->OffDiagonalApproximation != nullptr) {
//         std::vector<CoefficientPrecision> off_diagonal_out(cluster_tree_t->get_local_size() * mu, 0);
//         int off_diagonal_offset = (offset < this->cluster_tree_s->get_local_offset()) ? offset : offset - this->cluster_tree_s->get_local_size();
//         if (mu > 1 && !this->OffDiagonalApproximation->IsUsingRowMajorStorage()) { // Need to transpose input and output for OffDiagonalApproximation
//             std::vector<CoefficientPrecision> off_diagonal_input_column_major(size * mu, 0);

//             for (int i = 0; i < mu; i++) {
//                 for (int j = 0; j < size; j++) {
//                     off_diagonal_input_column_major[j + i * size] = in[i + j * mu];
//                 }
//             }

//             if (m_symmetry == 'H') {
//                 conj_if_complex(off_diagonal_input_column_major.data(), size * mu);
//             }

//             this->OffDiagonalApproximation->mvprod_subrhs_to_local(off_diagonal_input_column_major.data(), off_diagonal_out.data(), mu, off_diagonal_offset, size);

//             if (m_symmetry == 'H') {
//                 conj_if_complex(off_diagonal_out.data(), off_diagonal_out.size());
//             }
//             for (int i = 0; i < mu; i++) {
//                 for (int j = 0; j < local_size; j++) {
//                     out[i + j * mu] += off_diagonal_out[i * local_size + j];
//                 }
//             }
//         } else {
//             this->OffDiagonalApproximation->mvprod_subrhs_to_local(in, off_diagonal_out.data(), mu, off_diagonal_offset, size);

//             int incx(1), incy(1), local_size_rhs(local_size * mu);
//             T da(1);

//             Blas<CoefficientPrecision>::axpy(&local_size_rhs, &da, off_diagonal_out.data(), &incx, out, &incy);
//         }
//     }
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::source_to_cluster_permutation(const T *const in, T *const out) const {
//     global_to_cluster(cluster_tree_m_source_cluster_tree->get(), in, out);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::target_to_cluster_permutation(const T *const in, T *const out) const {
//     global_to_cluster(cluster_tree_m_target_cluster_tree->get(), in, out);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::cluster_to_target_permutation(const T *const in, T *const out) const {
//     cluster_to_global(cluster_tree_m_target_cluster_tree->get(), in, out);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::cluster_to_source_permutation(const T *const in, T *const out) const {
//     cluster_to_global(cluster_tree_m_source_cluster_tree->get(), in, out);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::local_target_to_local_cluster(const T *const in, T *const out, MPI_Comm comm) const {
//     local_to_local_cluster(cluster_tree_m_target_cluster_tree->get(), in, out, comm);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::local_source_to_local_cluster(const T *const in, T *const out, MPI_Comm comm) const {
//     local_to_local_cluster(cluster_tree_m_source_cluster_tree->get(), in, out, comm);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::local_cluster_to_local_target(const T *const in, T *const out, MPI_Comm comm) const {
//     local_cluster_to_local(cluster_tree_m_target_cluster_tree->get(), in, out, comm);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::local_cluster_to_local_source(const T *const in, T *const out, MPI_Comm comm) const {
//     local_cluster_to_local(cluster_tree_m_source_cluster_tree->get(), in, out, comm);
// }

// template <typename CoefficientPrecision>
// std::vector<CoefficientPrecision> HMatrix<CoefficientPrecision,CoordinatePrecision>::operator*(const std::vector<CoefficientPrecision> &x) const {
//     assert(x.size() == nc);
//     std::vector<CoefficientPrecision> result(nr, 0);
//     mvprod_global_to_global(x.data(), result.data(), 1);
//     return result;
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// underlying_type<CoefficientPrecision> HMatrix<CoefficientPrecision, CoordinatePrecision>::compressed_size() const {

//     double my_compressed_size = 0.;
//     double nr_b, nc_b, rank;
//     const auto &low_rank_leaves = m_block_tree_properties->m_low_rank_leaves;
//     const auto &dense_leaves    = m_block_tree_properties->m_dense_leaves;
//     std::cout << dense_leaves.size() << " " << low_rank_leaves.size() << "\n";
//     for (int j = 0; j < low_rank_leaves.size(); j++) {
//         nr_b = low_rank_leaves[j]->get_target_cluster_tree().get_size();
//         nc_b = low_rank_leaves[j]->get_source_cluster_tree().get_size();
//         rank = low_rank_leaves[j]->get_low_rank_data()->rank_of();
//         my_compressed_size += rank * (nr_b + nc_b);
//     }

//     for (int j = 0; j < dense_leaves.size(); j++) {
//         nr_b = dense_leaves[j]->get_target_cluster_tree().get_size();
//         nc_b = dense_leaves[j]->get_source_cluster_tree().get_size();
//         if (dense_leaves[j]->get_target_cluster_tree().get_offset() == dense_leaves[j]->get_source_cluster_tree().get_offset() && m_block_tree_properties->m_symmetry != 'N' && nr_b == nc_b) {
//             my_compressed_size += (nr_b * (nc_b + 1)) / 2;
//         } else {
//             my_compressed_size += nr_b * nc_b;
//         }
//     }

//     return my_compressed_size;
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::print_infos() const {
//     int rankWorld;
//     MPI_Comm_rank(comm, &rankWorld);

//     if (rankWorld == 0) {
//         for (std::map<std::string, std::string>::const_iterator it = infos.begin(); it != infos.end(); ++it) {
//             std::cout << it->first << "\t" << it->second << std::endl;
//         }
//         std::cout << std::endl;
//     }
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::save_infos(const std::string &outputname, std::ios_base::openmode mode, const std::string &sep) const {
//     int rankWorld;
//     MPI_Comm_rank(comm, &rankWorld);

//     if (rankWorld == 0) {
//         std::ofstream outputfile(outputname, mode);
//         if (outputfile) {
//             for (std::map<std::string, std::string>::const_iterator it = infos.begin(); it != infos.end(); ++it) {
//                 outputfile << it->first << sep << it->second << std::endl;
//             }
//             outputfile.close();
//         } else {
//             std::cout << "Unable to create " << outputname << std::endl; // LCOV_EXCL_LINE
//         }
//     }
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::save_plot(const std::string &outputname) const {

//     std::ofstream outputfile((outputname + ".csv").c_str());

//     if (outputfile) {
//         const auto &output = get_output();
//         outputfile << m_block_tree_properties->m_root_cluster_tree_target->get_size() << "," << m_block_tree_properties->m_root_cluster_tree_source->get_size() << std::endl;
//         for (const auto &block : output) {
//             outputfile << block << "\n";
//         }
//         outputfile.close();
//     } else {
//         std::cout << "Unable to create " << outputname << std::endl; // LCOV_EXCL_LINE
//     }
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// std::vector<DisplayBlock> HMatrix<CoefficientPrecision, CoordinatePrecision>::get_output() const {

//     std::vector<DisplayBlock> output(m_block_tree_properties->m_tasks.size());
//     int index = 0;
//     for (const auto &task : m_block_tree_properties->m_tasks) {
//         output[index].target_offset = taskm_target_clusterree().get_offset() - m_block_tree_properties->m_root_cluster_tree_target->get_offset();
//         output[index].target_size   = taskm_target_clusterree().get_size();
//         output[index].source_offset = task->get_m_t.().get_offset() - m_block_tree_properties->m_root_cluster_tree_source->get_offset();
//         output[index].source_size   = task->get_m_t.().get_size();
//         output[index].rank          = task->is_dense() ? -1 : task->m_low_rank_data->rank_of();
//         index++;
//     }

//     return output;
// }

// template <typename CoefficientPrecision>
// underlying_type<CoefficientPrecision> Frobenius_absolute_error(const HMatrix<CoefficientPrecision,CoordinatePrecision> &B, const VirtualGenerator<CoefficientPrecision> &A) {
//     underlying_type<CoefficientPrecision> myerr = 0;
//     for (int j = 0; j < B.MyFarFieldMats.size(); j++) {
//         underlying_type<CoefficientPrecision> test = Frobenius_absolute_error<CoefficientPrecision>(*(B.MyFarFieldMats[j]), *(B.MyFarFieldMats[j]->get_low_rank_block_data()), A);
//         myerr += std::pow(test, 2);
//     }

//     underlying_type<CoefficientPrecision> err = 0;
//     MPI_Allreduce(&myerr, &err, 1, wrapper_mpi<CoefficientPrecision>::mpi_underlying_type(), MPI_SUM, B.comm);

//     return std::sqrt(err);
// }

// template <typename CoefficientPrecision>
// Matrix<CoefficientPrecision> HMatrix<CoefficientPrecision,CoordinatePrecision>::get_local_dense() const {
//     Matrix<CoefficientPrecision> Dense(local_size, nc);
//     // Internal dense blocks
//     for (int l = 0; l < MyNearFieldMats.size(); l++) {
//         const Block<CoefficientPrecision> *submat = MyNearFieldMats[l];
//         int local_nr           = submat->get_target_cluster()->get_size();
//         int local_nc           = submat->get_source_cluster()->get_size();
//         int offset_i           = submat->get_target_cluster()->get_offset();
//         int offset_j           = submat->get_source_cluster()->get_offset();
//         for (int k = 0; k < local_nc; k++) {
//             std::copy_n(&(submat->get_dense_block_data()->operator()(0, k)), local_nr, Dense.data() + (offset_i - local_offset) + (offset_j + k) * local_size);
//         }
//     }

//     // Internal compressed block
//     for (int l = 0; l < MyFarFieldMats.size(); l++) {
//         const Block<CoefficientPrecision> *lmat = MyFarFieldMats[l];
//         int local_nr         = lmat->get_target_cluster()->get_size();
//         int local_nc         = lmat->get_source_cluster()->get_size();
//         int offset_i         = lmat->get_target_cluster()->get_offset();
//         int offset_j         = lmat->get_source_cluster()->get_offset();
//         Matrix<CoefficientPrecision> FarFielBlock(local_nr, local_nc);
//         lmat->get_low_rank_block_data()->get_whole_matrix(&(FarFielBlock(0, 0)));
//         for (int k = 0; k < local_nc; k++) {
//             std::copy_n(&(FarFielBlock(0, k)), local_nr, Dense.data() + (offset_i - local_offset) + (offset_j + k) * local_size);
//         }
//     }
//     return Dense;
// }

// template <typename CoefficientPrecision>
// Matrix<CoefficientPrecision> HMatrix<CoefficientPrecision,CoordinatePrecision>::get_local_dense_perm() const {
//     Matrix<CoefficientPrecision> Dense(local_size, nc);
//     copy_local_dense_perm(Dense.data());
//     return Dense;
//

// void MMmat(Matrix<CoordinatePrecision> L, const HMatrix<CoefficientPrecision, CoordinatePrecision> *H, const HMatrix<CoefficientPrecision, CoordinatePrecision> *K) {
//     auto &H_child = H->get_children();
//     auto &K_child = K->get_children();
//     Matrix<CoefficientPrecision> Z(H->get_target_cluster().get_size(), K->get_source_cluster().get_size());
//     if ((H_child.size() == 0) or (K_child.size() == 0)) {
//         // on differencie tout les cas
//         if ((H_child.size() > 0) and (K_child.size() == 0)) {
//             // 2 cas hmat * lr ou hmat*full
//             if (K->is_low_rank()) {
//                 std::cout << 1 << std::endl;
//                 auto U  = K->get_low_rank_data()->Get_U();
//                 auto V  = K->get_low_rank_data()->Get_V();
//                 auto Hu = H->hmat_lr(U);
//                 Z       = Hu * V;
//                 std::cout << 11 << std::endl;

//                 // LowRankMatrix<CoefficientPrecision, CoordinatePrecision> HK(Hu, V);
//                 // L->set_low_rank_data(HK);
//             } else {
//                 std::cout << 2 << std::endl;

//                 auto k  = *K->get_dense_data();
//                 auto HK = H->hmat_lr(k);
//                 // L->set_dense_data(HK);
//                 Z = HK;
//                 std::cout << 22 << std::endl;
//             }
//         } else if ((H_child.size() == 0) and (K_child.size()) > 0) {
//             // 2 cas full * hmat ou lr * hmat
//             if (H->is_low_rank()) {
//                 std::cout << 3 << std::endl;

//                 auto U  = H->get_low_rank_data()->Get_U();
//                 auto V  = H->get_low_rank_data()->Get_V();
//                 auto vK = K->lr_hmat(V);
//                 // LowRankMatrix<CoefficientPrecision, CoordinatePrecision> HK(U, vK);
//                 // L->set_low_rank_data(HK);
//                 Z = U * vK;
//                 std::cout << 33 << std::endl;

//             } else {
//                 std::cout << 4 << std::endl;

//                 auto h  = *H->get_dense_data();
//                 auto HK = K->lr_hmat(h);
//                 // L->set_dense_data(HK);
//                 Z = HK;
//                 std::cout << 44 << std::endl;
//             }
//         } else {
//             std::cout << 5 << std::endl;
//             std::cout << H->get_target_cluster().get_size() << ',' << H->get_source_cluster().get_size() << ',' << K->get_target_cluster().get_size() << ',' << K->get_source_cluster().get_size() << std::endl;

//             Matrix<CoefficientPrecision> hh(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
//             Matrix<CoefficientPrecision> kk(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
//             if (H->is_low_rank()) {
//                 hh = H->get_low_rank_data()->Get_U() * H->get_low_rank_data()->Get_V();
//             } else if (H->is_dense()) {
//                 hh = *H->get_dense_data();
//             }
//             if (K->is_low_rank()) {
//                 kk = K->get_low_rank_data()->Get_U() * K->get_low_rank_data()->Get_V();
//             } else if (K->is_dense()) {
//                 kk = *K->get_dense_data();
//             }
//             Z = hh * kk;
//             std::cout << 55 << std::endl;

//             // L->set_dense_data(hk);
//         }
//     } else {
//         for (auto &h_child : H_child) {
//             for (auto &k_child : K_child) {
//                 if (k_child->get_target_cluster() == h_child->get_source_cluster()) {
//                     MMmat(Z, h_child.get(), k_child.get());
//                 }
//             }
//         }
//         for (int k = 0; k < H->get_target_cluster().get_size(); ++k) {
//             for (int l = 0; l < K->get_source_cluster().get_size(); ++l) {
//                 L(k + H->get_target_cluster().get_offset(), l + K->get_source_cluster().get_offset()) = Z(k, l);
//             }
//         }
//         // L = L + Z;
//     }
// }

template <typename CoefficientPrecision, typename CoordinatePrecision>
void copy_to_dense(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, CoefficientPrecision *ptr) {

    int target_offset = hmatrix.get_target_cluster().get_offset();
    int source_offset = hmatrix.get_source_cluster().get_offset();
    int target_size   = hmatrix.get_target_cluster().get_size();
    int source_size   = hmatrix.get_source_cluster().get_size();
    // const std::vector<int> &target_permutation = m_target_cluster->get_permutation();
    // const std::vector<int> &source_permutation = m_source_cluster->get_permutation();
    // std::cout << "copy_to_dense " << target_offset << " " << source_offset << " " << target_size << " " << source_size << "\n";
    if (hmatrix.is_dense()) {
        int local_nc = hmatrix.get_source_cluster().get_size();
        int local_nr = hmatrix.get_target_cluster().get_size();
        int offset_i = hmatrix.get_target_cluster().get_offset() - target_offset;
        int offset_j = hmatrix.get_source_cluster().get_offset() - source_offset;
        auto mat     = *hmatrix.get_dense_data();
        for (int k = 0; k < local_nc; ++k) {
            for (int j = 0; j < local_nr; ++j) {
                ptr[j + offset_i + (k + offset_j) * target_size] = mat(j, k);
            }
        }
    } else if (hmatrix.is_low_rank()) {
        int local_nc = hmatrix.get_source_cluster().get_size();
        int local_nr = hmatrix.get_target_cluster().get_size();
        int offset_i = hmatrix.get_target_cluster().get_offset() - target_offset;
        int offset_j = hmatrix.get_source_cluster().get_offset() - source_offset;
        auto mat     = hmatrix.get_low_rank_data()->Get_U() * hmatrix.get_low_rank_data()->Get_V();
        for (int k = 0; k < local_nc; ++k) {
            for (int j = 0; j < local_nr; ++j) {
                ptr[j + offset_i + (k + offset_j) * target_size] = mat(j, k);
            }
        }
    } else {
        for (auto leaf : hmatrix.get_leaves()) {
            int local_nr = leaf->get_target_cluster().get_size();

            int local_nc = leaf->get_source_cluster().get_size();
            int offset_i = leaf->get_target_cluster().get_offset() - target_offset;
            int offset_j = leaf->get_source_cluster().get_offset() - source_offset;
            if (leaf->is_dense()) {
                for (int k = 0; k < local_nc; k++) {
                    for (int j = 0; j < local_nr; j++) {
                        ptr[j + offset_i + (k + offset_j) * target_size] = (*leaf->get_dense_data())(j, k);
                    }
                }
            } else {

                Matrix<CoefficientPrecision> low_rank_to_dense(local_nr, local_nc);
                leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
                for (int k = 0; k < local_nc; k++) {
                    for (int j = 0; j < local_nr; j++) {
                        ptr[j + offset_i + (k + offset_j) * target_size] = low_rank_to_dense(j, k);
                    }
                }
            }
        }

        char symmetry_type = hmatrix.get_symmetry_for_leaves();
        for (auto leaf : hmatrix.get_leaves_for_symmetry()) {
            int local_nr   = leaf->get_target_cluster().get_size();
            int local_nc   = leaf->get_source_cluster().get_size();
            int col_offset = leaf->get_target_cluster().get_offset() - source_offset;
            int row_offset = leaf->get_source_cluster().get_offset() - target_offset;
            if (leaf->is_dense()) {
                for (int j = 0; j < local_nr; j++) {
                    for (int k = 0; k < local_nc; k++) {
                        ptr[k + row_offset + (j + col_offset) * target_size] = (*leaf->get_dense_data())(j, k);
                    }
                }
            } else {

                Matrix<CoefficientPrecision> low_rank_to_dense(local_nr, local_nc);
                leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
                for (int j = 0; j < local_nr; j++) {
                    for (int k = 0; k < local_nc; k++) {
                        ptr[k + row_offset + (j + col_offset) * target_size] = low_rank_to_dense(j, k);
                    }
                }
            }
            if (symmetry_type == 'H') {
                for (int k = 0; k < local_nc; k++) {
                    for (int j = 0; j < local_nr; j++) {
                        ptr[k + row_offset + (j + col_offset) * target_size] = conj_if_complex(ptr[k + row_offset + (j + col_offset) * target_size]);
                    }
                }
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void copy_diagonal(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, CoefficientPrecision *ptr) {
    if (hmatrix.get_target_cluster().get_offset() != hmatrix.get_source_cluster().get_offset() || hmatrix.get_target_cluster().get_size() != hmatrix.get_source_cluster().get_size()) {
        htool::Logger::get_instance().log(Logger::LogLevel::ERROR, "Matrix is not square a priori, get_local_diagonal cannot be used"); // LCOV_EXCL_LINE
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

// template <typename CoefficientPrecision>
// Matrix<CoefficientPrecision> HMatrix<CoefficientPrecision,CoordinatePrecision>::get_local_interaction(bool permutation) const {
//     int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;
//     Matrix<CoefficientPrecision> local_interaction(local_size, local_size_source);
//     copy_local_interaction(local_interaction.data(), permutation);
//     return local_interaction;
// }

// template <typename CoefficientPrecision>
// Matrix<CoefficientPrecision> HMatrix<CoefficientPrecision,CoordinatePrecision>::get_local_diagonal_block(bool permutation) const {
//     int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;
//     Matrix<CoefficientPrecision> diagonal_block(local_size, local_size_source);
//     copy_local_diagonal_block(diagonal_block.data(), permutation);
//     return diagonal_block;
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::copy_local_interaction(T *ptr, bool permutation) const {
//     if ((!(cluster_tree_t->IsLocal()) || !(cluster_tree_s->IsLocal())) && permutation) {
//         throw std::logic_error("[Htool error] Permutation is not local, get_local_interaction cannot be used"); // LCOV_EXCL_LINE
//     }

//     int local_offset_source = cluster_tree_s->get_masteroffset(rankWorld).first;
//     int local_size_source   = cluster_tree_s->get_masteroffset(rankWorld).second;
//     // Internal dense blocks
//     for (int i = 0; i < MyDiagNearFieldMats.size(); i++) {
//         const Block<CoefficientPrecision> *submat = MyDiagNearFieldMats[i];
//         int local_nr                              = submat->get_target_cluster()->get_size();
//         int local_nc                              = submat->get_source_cluster()->get_size();
//         int offset_i                              = submat->get_target_cluster()->get_offset() - local_offset;
//         int offset_j                              = submat->get_source_cluster()->get_offset() - local_offset_source;
//         for (int i = 0; i < local_nc; i++) {
//             std::copy_n(&(submat->get_dense_block_data()->operator()(0, i)), local_nr, ptr + offset_i + (offset_j + i) * local_size);
//         }
//     }

//     // Internal compressed block
//     for (int i = 0; i < MyDiagFarFieldMats.size(); i++) {
//         const Block<CoefficientPrecision> *lmat = MyDiagFarFieldMats[i];
//         int local_nr                            = lmat->get_target_cluster()->get_size();
//         int local_nc                            = lmat->get_source_cluster()->get_size();
//         int offset_i                            = lmat->get_target_cluster()->get_offset() - local_offset;
//         int offset_j                            = lmat->get_source_cluster()->get_offset() - local_offset_source;
//         ;
//         Matrix<CoefficientPrecision> FarFielBlock(local_nr, local_nc);
//         lmat->get_low_rank_block_data()->get_whole_matrix(&(FarFielBlock(0, 0)));
//         for (int i = 0; i < local_nc; i++) {
//             std::copy_n(&(FarFielBlock(0, i)), local_nr, ptr + offset_i + (offset_j + i) * local_size);
//         }
//     }

// // Asking for permutation while symmetry!=N means that the block is upper/lower triangular in Htool's numbering, but it is not true in User's numbering

// if (permutation && m_symmetry != 'N') {
//     if (UPLO == 'L' && m_symmetry == 'S') {
//         for (int i = 0; i < local_size; i++) {
//             for (int j = 0; j < i; j++) {
//                 ptr[j + i * local_size] = ptr[i + j * local_size];
//             }
//         }
//     }

//     if (UPLO == 'U' && m_symmetry == 'S') {
//         for (int i = 0; i < local_size; i++) {
//             for (int j = i + 1; j < local_size_source; j++) {
//                 ptr[j + i * local_size] = ptr[i + j * local_size];
//             }
//         }
//     }
//     if (UPLO == 'L' && m_symmetry == 'H') {
//         for (int i = 0; i < local_size; i++) {
//             for (int j = 0; j < i; j++) {
//                 ptr[j + i * local_size] = conj_if_complex(ptr[i + j * local_size]);
//             }
//         }
//     }

//     if (UPLO == 'U' && m_symmetry == 'H') {
//         for (int i = 0; i < local_size; i++) {
//             for (int j = i + 1; j < local_size_source; j++) {
//                 ptr[j + i * local_size] = conj_if_complex(ptr[i + j * local_size]);
//             }
//         }
//     }
// }
// // Permutations
// if (permutation) {
//     Matrix<CoefficientPrecision> diagonal_block_perm(local_size, local_size_source);
//     for (int i = 0; i < local_size; i++) {
//         for (int j = 0; j < local_size_source; j++) {
//             diagonal_block_perm(i, cluster_tree_s->get_global_perm(j + local_offset_source) - local_offset_source) = ptr[i + j * local_size];
//         }
//     }

//     for (int i = 0; i < local_size; i++) {
//         this->local_cluster_to_local_target(diagonal_block_perm.data() + i * local_size, ptr + i * local_size, comm);
//     }
// }
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::copy_local_diagonal_block(T *ptr, bool permutation) const {
//     if (cluster_tree_t != cluster_tree_s) {
//         throw std::logic_error("[Htool error] Matrix is not square a priori, get_local_diagonal_block cannot be used"); // LCOV_EXCL_LINE
//     }
//     copy_local_interaction(ptr, permutation);
// }

// template <typename CoefficientPrecision>
// std::pair<int, int> HMatrix<CoefficientPrecision,CoordinatePrecision>::get_max_size_blocks() const {
//     int local_max_size_j = 0;
//     int local_max_size_i = 0;

//     for (int i = 0; i < MyFarFieldMats.size(); i++) {
//         if (local_max_size_j < MyFarFieldMats[i]->get_source_cluster()->get_size())
//             local_max_size_j = MyFarFieldMats[i]->get_source_cluster()->get_size();
//         if (local_max_size_i < MyFarFieldMats[i]->get_target_cluster()->get_size())
//             local_max_size_i = MyFarFieldMats[i]->get_target_cluster()->get_size();
//     }
//     for (int i = 0; i < MyNearFieldMats.size(); i++) {
//         if (local_max_size_j < MyNearFieldMats[i]->get_source_cluster()->get_size())
//             local_max_size_j = MyNearFieldMats[i]->get_source_cluster()->get_size();
//         if (local_max_size_i < MyNearFieldMats[i]->get_target_cluster()->get_size())
//             local_max_size_i = MyNearFieldMats[i]->get_target_cluster()->get_size();
//     }

//     return std::pair<int, int>(local_max_size_i, local_max_size_j);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::apply_dirichlet(const std::vector<int> &boundary) {
//     // Renum
//     std::vector<int> boundary_renum(boundary.size());
//     this->source_to_cluster_permutation(boundary.data(), boundary_renum.data());

//     //
//     for (int j = 0; j < MyStrictlyDiagNearFieldMats.size(); j++) {
//         SubMatrix<CoefficientPrecision> &submat = *(MyStrictlyDiagNearFieldMats[j]);
//         int local_nr         = submat.nb_rows();
//         int local_nc         = submat.nb_cols();
//         int offset_i         = submam_target_cluster_tree->get_offset_i();
//         for (int i = offset_i; i < offset_i + std::min(local_nr, local_nc); i++) {
//             if (boundary_renum[i])
//                 submat(i - offset_i, i - offset_i) = 1e30;
//         }
//     }
// }

} // namespace htool
#endif
