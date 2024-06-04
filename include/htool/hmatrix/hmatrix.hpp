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
#include "sumexpr.hpp"
#include <htool/wrappers/wrapper_lapack.hpp>
#include <variant>

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
        Hierarchical,
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
    void recursive_build_hmatrix_triangular_product(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr, const char L, const char U);
    void recursive_build_hmatrix_product_compression(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr);

    void recursive_build_hmatrix_product_new(const SumExpression_update<CoefficientPrecision, CoordinatePrecision> &sum_expr);

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

    underlying_type<CoefficientPrecision> get_epsilon() const { return this->m_tree_data->m_epsilon; }
    // Arthur -> j'ai besoin de pouvoir accéder au low_rank generato ->  on juste passer par m_tree_data
    std::shared_ptr<VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>> get_low_rank_generator() { return this->m_tree_data->m_low_rank_generator; }
    std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> get_admissibility_condition() { return this->m_tree_data->m_admissibility_condition; }
    // HMatrix Tree setters
    const HMatrix<CoefficientPrecision, CoordinatePrecision> *get_diagonal_hmatrix() const { return this->m_tree_data->m_block_diagonal_hmatrix; }
    char get_symmetry_for_leaves() const { return m_symmetry_type_for_leaves; }

    double get_compression() const {
        double compr = 0.0;
        auto leaves  = this->m_leaves;
        // std::cout << "leaves size " << leaves.size() << std::endl;
        for (int k = 0; k < leaves.size(); ++k) {
            auto Hk = leaves[k];
            // std::cout << "denseeee" << Hk->is_dense() << ',' << Hk->is_low_rank() << std::endl;
            if (Hk->is_dense()) {
                compr += Hk->get_target_cluster().get_size() * Hk->get_source_cluster().get_size();
            } else if (Hk->is_low_rank()) {
                auto temp = (Hk->m_low_rank_data->rank_of() * (Hk->get_target_cluster().get_size() + Hk->get_source_cluster().get_size()));
                compr     = compr + temp;
            }
        }
        compr = compr * 1.0 / (this->get_target_cluster().get_size() * this->get_source_cluster().get_size());
        return 1.0 - compr;
    }

    double get_compression_triang(const char N) const {
        double compr = 0.0;
        auto leaves  = this->m_leaves;
        // std::cout << "leaves size " << leaves.size() << std::endl;
        for (int k = 0; k < leaves.size(); ++k) {
            auto Hk = leaves[k];
            // std::cout << "denseeee" << Hk->is_dense() << ',' << Hk->is_low_rank() << std::endl;
            if (N == 'U') {
                if (Hk->get_source_cluster().get_offset() >= Hk->get_target_cluster().get_offset()) {
                    if (Hk->is_dense()) {
                        compr += Hk->get_target_cluster().get_size() * Hk->get_source_cluster().get_size();
                    } else if (Hk->is_low_rank()) {
                        auto temp = (Hk->m_low_rank_data->rank_of() * (Hk->get_target_cluster().get_size() + Hk->get_source_cluster().get_size()));
                        compr     = compr + temp;
                    }
                }
            } else {
                if (Hk->get_source_cluster().get_offset() <= Hk->get_target_cluster().get_offset()) {
                    if (Hk->is_dense()) {
                        compr += Hk->get_target_cluster().get_size() * Hk->get_source_cluster().get_size();
                    } else if (Hk->is_low_rank()) {
                        auto temp = (Hk->m_low_rank_data->rank_of() * (Hk->get_target_cluster().get_size() + Hk->get_source_cluster().get_size()));
                        compr     = compr + temp;
                    }
                }
            }
        }
        compr = compr * 1.0 / (this->get_target_cluster().get_size() * this->get_source_cluster().get_size());
        return 1.0 - compr;
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

    // pour avoir le fils (t,s) , a papeler de proche en proche
    HMatrix *get_son(const int size_t, const int size_s, const int offset_t, const int offset_s) const {
        int flag = 0;
        for (auto &child : this->get_children()) {
            if ((child->get_target_cluster().get_size() == size_t) && (child->get_source_cluster().get_size() == size_s) && (child->get_target_cluster().get_offset() == offset_t) && (child->get_source_cluster().get_offset() == offset_s)) {
                return child.get();
            }
        }
        return nullptr;
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
    void save_plot(const std::string &outputname) const;
    // std::vector<DisplayBlock> get_output() const;

    // Data computation
    void compute_dense_data(const VirtualGenerator<CoefficientPrecision> &generator) {
        m_dense_data = std::make_unique<Matrix<CoefficientPrecision>>(m_target_cluster->get_size(), m_source_cluster->get_size());
        generator.copy_submatrix(m_target_cluster->get_size(), m_source_cluster->get_size(), m_target_cluster->get_offset(), m_source_cluster->get_offset(), m_dense_data->data());
        m_storage_type = StorageType::Dense;
    }

    void compute_low_rank_data(const VirtualGenerator<CoefficientPrecision> &generator, const VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision> &low_rank_generator, int reqrank, underlying_type<CoefficientPrecision> epsilon) {
        // std::cout << this->m_tree_data->m_epsilon << " " << epsilon << "\n";
        m_low_rank_data = std::make_unique<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(generator, low_rank_generator, *m_target_cluster, *m_source_cluster, reqrank, epsilon);
        m_storage_type  = StorageType::LowRank;
    }
    void clear_low_rank_data() { m_low_rank_data.reset(); }

    // je rajoute ca
    void set_low_rank_data(const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> &lrmat) {
        m_low_rank_data = std::unique_ptr<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(new LowRankMatrix<CoefficientPrecision, CoordinatePrecision>(lrmat));
        m_storage_type  = StorageType::LowRank;
    }

    void set_dense_data(const Matrix<CoefficientPrecision> &M) {
        m_dense_data   = std::unique_ptr<Matrix<CoefficientPrecision>>(new Matrix<CoefficientPrecision>(M));
        m_storage_type = StorageType::Dense;
    }

    // void set_dense_data(const Matrix<CoefficientPrecision> &M) {
    //     auto &t = this->get_target_cluster();
    //     auto &s = this->get_source_cluster();
    //     if (t.get_depth() < t.get_maximal_depth() && s.get_depth() < s.get_maximal_depth()) {
    //         this->split(M);
    //     } else {
    //         m_dense_data   = std::unique_ptr<Matrix<CoefficientPrecision>>(new Matrix<CoefficientPrecision>(M));
    //         m_storage_type = StorageType::Dense;
    //     }
    // }

    // Linear algebra
    void add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const;
    void add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const;

    HMatrix hmatrix_product(const HMatrix &B) const;
    HMatrix hmatrix_triangular_product(const HMatrix &B, const char L, const char U) const;

    HMatrix hmatrix_product_compression(const HMatrix &B) const;

    HMatrix hmatrix_product_new(const HMatrix &B) const;

    ////////////////////////
    // GET_BLOCK -> restriciton de la hmatrice a un bloc
    // je voulais vraiment pas faire cette fonction mais on est obligé de pouvoir atteindre des blocs qui sont pas forcemment des blocs fils dans ForwardM du coup je vois pas d'autres solutions
    // Il faudrait pouvoir renvoyer une hmat ou une matrice mais on j'arrive pas a utiliser std::variant et on peut pas faire un vecteur <type a , type b> donc je vois pas
    // complexité O( log(n) )\leq 2^depth -> on descend juste l'arbre avec "un seul" test a chaque niveau
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

    //////////////////////X
    ///// formatage minblock size
    ///////////
    // HMatrix<CoefficientPrecision, CoordinatePrecision> call_build(Const HMatrix<CoefficientPrecision, CoordinatePrecision> *H) {
    //     HMatrix<CoefficientPrecision, CoordinatePrecision> *res(H->get_target_cluster(), H->get_source_cluster());
    // }
    void format_minblock(const HMatrix<CoefficientPrecision, CoordinatePrecision> *H, const int &min_block_size) {
        if (this->get_target_cluster().get_size() < min_block_size + 3) {
            Matrix<CoefficientPrecision> dense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
            copy_to_dense(*H, dense.data());
            auto hk = dense.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset() - H->get_target_cluster().get_offset(), this->get_source_cluster().get_offset() - H->get_source_cluster().get_offset());

            // std::cout << "on push une feuille de norme dense  " << normFrob(hk) << std::endl;
            this->set_dense_data(hk);
        } else {
            if (H->get_children().size() == 0) {
                // on est lr
                if (H->is_low_rank()) {
                    this->set_low_rank_data(*H->get_low_rank_data());
                    // std::cout << "on push une feuille low rank de norme " << normFrob(H->get_low_rank_data()->Get_U() * H->get_low_rank_data()->Get_V()) << std::endl;

                } else {
                    // on est dense mais sur un bloc adm -> on test lr
                    bool is_admissible = false;
                    if (this->get_target_cluster().get_offset() != this->get_source_cluster().get_offset()) {
                        is_admissible = this->get_admissibility_condition()->ComputeAdmissibility(this->get_target_cluster(), this->get_source_cluster(), this->m_tree_data->m_eta);
                    }
                    auto mat = *H->get_dense_data();
                    if (mat.nb_rows() > this->get_target_cluster().get_size()) {
                        mat = mat.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset() - H->get_target_cluster().get_offset(), this->get_source_cluster().get_offset() - H->get_source_cluster().get_offset());
                    }
                    MatrixGenerator<double> gen(mat, 0, 0);
                    // MatGenerator<double> gen(*mat, )
                    if (is_admissible) {
                        LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(gen, *this->m_tree_data->m_low_rank_generator, this->get_target_cluster(), this->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
                        if ((lr.Get_U().nb_rows() == 0) or (lr.Get_U().nb_cols() == 0) or (lr.Get_V().nb_rows() == 0) or (lr.Get_V().nb_cols() == 0)) {
                            if (this->get_target_cluster().get_size() > min_block_size) {
                                for (auto &t : this->get_target_cluster().get_children()) {
                                    for (auto &s : this->get_source_cluster().get_children()) {
                                        auto child = this->add_child(t.get(), s.get());
                                        auto hk    = H->get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset());
                                        child->format_minblock(hk, min_block_size);
                                    }
                                }
                            } else {
                                // std::cout << "on push une feuille de norme dense " << normFrob(mat) << std::endl;

                                this->set_dense_data(mat);
                            }
                        } else {
                            // std::cout << "on push une feuille low rank de norme " << normFrob(lr.Get_U() * lr.Get_V()) << std::endl;

                            this->set_low_rank_data(lr);
                        }
                    } else {
                        // on est juste dense -> on descend si on peut
                        if (this->get_target_cluster().get_size() > min_block_size) {
                            for (auto &t : this->get_target_cluster().get_children()) {
                                for (auto &s : this->get_source_cluster().get_children()) {
                                    auto child = this->add_child(t.get(), s.get());
                                    auto hk    = H->get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset());
                                    child->format_minblock(hk, min_block_size);
                                }
                            }
                        }

                        else {
                            //     auto &hk = H->get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset());
                            // Matrix<CoefficientPrecision> hdense(t->get_size(), s->get_size());
                            // copy_to_dense(H, hdense.data())
                            auto hdense = H->get_dense_data()->get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                            // MatrixGenerator<double> gen(hdense, 0, 0);
                            // std::cout << "on push une feuille de norme dense " << normFrob(hdense) << std::endl;
                            this->set_dense_data(hdense);
                        }
                    }
                }
            } else {
                if (this->get_target_cluster().get_size() > min_block_size) {
                    for (auto &t : this->get_target_cluster().get_children()) {
                        for (auto &s : this->get_source_cluster().get_children()) {
                            auto child = this->add_child(t.get(), s.get());
                            auto hk    = H->get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset());
                            child->format_minblock(hk, min_block_size);
                        }
                    }
                } else {
                    // normalement lui il existe mais AU CAS OU
                    auto hdense = H->get_dense_data()->get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
                    // MatrixGenerator<double> gen(hdense, 0, 0);
                    std::cout << "on push une feuille de norme dense" << normFrob(hdense) << std::endl;

                    this->set_dense_data(hdense);
                }
            }
        }
    }

    ////////////////////
    /// ASSIGN  , = bloc t,s
    ///////////////////
    void
    assign(const HMatrix<CoefficientPrecision, CoordinatePrecision> &M, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) {
        auto Mts       = M.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        auto &children = Mts->get_children();
        if (children.size() == 0) {
            if (Mts->is_dense()) {
                this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(*Mts->get_dense_data());
                // this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->delete_children();
                this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->m_storage_type = StorageType::Dense;
            } else {
                this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_low_rank_data(*Mts->get_low_rank_data());
                // this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->delete_children();
                this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->m_storage_type = StorageType::LowRank;
            }
        } else {
            for (auto &child : children) {
                this->assign(*child, child->get_target_cluster(), child->get_source_cluster());
            }
        }
    }
    //////////////
    /// fonction extract pour récupérer le petit blocs qu'on arrive pas a choper avec le restrict
    Matrix<CoefficientPrecision> extract(int szt, int szs, int oft, int ofs) const {
        Matrix<CoefficientPrecision> dense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
        copy_to_dense(*this, dense.data());
        Matrix<CoefficientPrecision> restr(szt, szs);
        for (int k = 0; k < szt; ++k) {
            for (int l = 0; l < szs; ++l) {
                restr(k, l) = dense(k + oft - this->get_target_cluster().get_offset(), l + ofs - this->get_source_cluster().get_offset());
            }
        }
        return restr;
    }
    // Matrix<CoefficientPrecision> extract(int szt, int szs, int oft, int ofs) {
    //     if (this->is_dense()) {
    //         Matrix<CoefficientPrecision> res(szt, szs);
    //         auto dense = *(this->get_dense_data());
    //         for (int k = 0; k < szt; ++k) {
    //             for (int l = 0; l < szs; ++l) {
    //                 res(k, l) = dense(k + oft - this->get_target_cluster().get_offset(), l + ofs - this->get_source_cluster().get_offset());
    //             }
    //         }
    //         return res;
    //     }
    //     // get_block a pas marché donc on a voulu aller trop bas donc on est sur un feuille donc si on est pas full on est lr
    //     else {
    //         auto U = this->get_low_rank_data()->Get_U();
    //         auto V = this->get_low_rank_data()->Get_V();
    //         Matrix<CoefficientPrecision> u_restr(szt, U.nb_cols());
    //         Matrix<CoefficientPrecision> v_restr(V.nb_rows(), szs);
    //         for (int k = 0; k < szt; ++k) {
    //             u_restr.set_row(k, U.get_row(k + oft));
    //         }
    //         for (int l = 0; l < szs; ++l) {
    //             v_restr.set_col(l, V.get_col(l + ofs));
    //         }
    //         return u_restr * v_restr;
    //     }
    // }

    //////////////////
    // OK POUR DELTE ?

    void Get_Block(const int &szt, const int &szs, const int &oft, const int &ofs, std::pair<HMatrix<CoefficientPrecision, CoordinatePrecision> *, Matrix<CoefficientPrecision>> res) {
        if (this->get_children().size() == 0) {
            std::cout << "?" << std::endl;
            if ((this->get_target_cluster().get_size() == szt) && (this->get_target_cluster().get_offset() == oft) && (this->get_source_cluster().get_size() == szs) && (this->get_source_cluster().get_offset() == ofs)) {
                std::cout << "!" << std::endl;
                res.first = const_cast<HMatrix<CoefficientPrecision, CoordinatePrecision> *>(this);

            } else {
                Matrix<double> M(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
                copy_to_dense(*this, M.data());
                Matrix<double> m(szt, szs);
                for (int k = 0; k < szt; ++k) {
                    for (int l = 0; l < szs; ++l) {
                        m(k, l) = M(k + oft - this->get_target_cluster().get_offset(), l + ofs - this->get_source_cluster().get_offset());
                    }
                }
                res.second = m;
            }
        } else {
            // std::cout << "?" << std::endl;

            if ((this->get_target_cluster().get_size() == szt) && (this->get_target_cluster().get_offset() == oft) && (this->get_source_cluster().get_size() == szs) && (this->get_source_cluster().get_offset() == ofs)) {
                // std::cout << "!!!!!!!!!!!" << std::endl;
                res.first = const_cast<HMatrix<CoefficientPrecision, CoordinatePrecision> *>(this);
                std::cout << (res.first == nullptr) << std::endl;
            } else {
                for (auto &child : this->get_children()) {
                    auto &target_cluster = child->get_target_cluster();
                    auto &source_cluster = child->get_source_cluster();
                    if (((szt + (oft - child->get_target_cluster().get_offset()) <= child->get_target_cluster().get_size()) and (szt <= child->get_target_cluster().get_size()) and (oft >= child->get_target_cluster().get_offset()))
                        and (((szs + (ofs - child->get_source_cluster().get_offset()) <= child->get_source_cluster().get_size()) and (szs <= child->get_source_cluster().get_size()) and (ofs >= child->get_source_cluster().get_offset())))) {
                        child->Get_Block(szt, szs, oft, ofs, res);
                    }
                }
            }
        }
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
            // MatrixGenerator<CoefficientPrecision> ss(sol);
            // res.compute_dense_data(ss);
            res.set_dense_data(sol);
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
    // fin de delete?
    //////////////////////////////////////////////////////////////////////////
    /// ALGEBRE LINEAIRE SIMPLE  -> il manque la distinction low_rank dans + et -
    //////////////////

    /////////////
    /// Arthur : TO DO -> distinction low rank/ dense

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

    // A = A-B

    // void Moins(const HMatrix<CoefficientPrecision, CoordinatePrecision> &B) {
    //     auto &children = this->get_children();
    //     if (children.size() == 0) {
    //         auto &t    = this->get_target_cluster();
    //         auto &s    = this->get_source_cluster();
    //         auto Btemp = B.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //         if ((Btemp->get_target_cluster().get_size() == t.get_size()) && (Btemp->get_target_cluster().get_offset() == t.get_offset()) && (Btemp->get_source_cluster().get_size() == s.get_size()) && (Btemp->get_source_cluster().get_offset()) == s.get_offset()) {
    //             if (this->is_dense()) {
    //                 auto a = this->get_dense_data();
    //                 Matrix<CoefficientPrecision> b(t.get_size(), s.get_size());
    //                 copy_to_dense(*Btemp, b.data());
    //                 auto c = *a - b;
    //                 this->set_dense_data(c);
    //             } else {
    //                 Matrix<CoefficientPrecision> a(t.get_size(), s.get_size()), b(t.get_size(), s.get_size());
    //                 copy_to_dense(*this, a.data());
    //                 copy_to_dense(*Btemp, b.data());
    //                 auto c = a - b;
    //                 MatrixGenerator<CoefficientPrecision> mat(c, t.get_offset(), s.get_offset());
    //                 LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(mat, *this->m_tree_data->m_low_rank_generator, this->get_target_cluster(), this->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
    //                 this->set_low_rank_data(lr);
    //             }
    //         } else {
    //             Matrix<CoefficientPrecision> bdense(Btemp->get_target_cluster().get_size(), Btemp->get_source_cluster().get_size());
    //             Matrix<CoefficientPrecision> brestr(t.get_size(), s.get_size());
    //             copy_to_dense(*Btemp, bdense.data());
    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 for (int l = 0; l < s.get_size(); ++l) {
    //                     brestr(k + t.get_offset() - Btemp->get_target_cluster().get_offset(), l + s.get_offset() - Btemp->get_source_cluster().get_offset());
    //                 }
    //             }
    //             if (this->is_dense()) {
    //                 auto a = this->get_dense_data();
    //                 auto c = *a - brestr;
    //                 this->set_dense_data(c);
    //             } else {
    //                 Matrix<CoefficientPrecision> a(t.get_size(), s.get_size());
    //                 copy_to_dense(*this, a.data());
    //                 auto c = a - brestr;
    //                 MatrixGenerator<CoefficientPrecision> mat(c, t.get_offset(), s.get_offset());
    //                 LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(mat, *this->m_tree_data->m_low_rank_generator, this->get_target_cluster(), this->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
    //                 this->set_low_rank_data(lr);
    //             }
    //         }
    //     } else {
    //         for (auto &child : children) {
    //             child->Moins(B);
    //         }
    //     }
    // }*

    //// On prend la convention de garder le bloccluser tree de A poiur A=A-B  ( Ca arrive avec proba 0 que struct(A)=struct(B))
    // void Moins(const HMatrix &B, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) {
    //     auto temp = this->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //     // // Si on est sur une feuille
    //     if (temp->get_children().size() == 0) {
    //         // on récupère la feuille correspondante dans B
    //         auto Bts = B.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //         // Soit Bts est une feuille soit il faut extraire le sous blocs , dans les deux cas on va avoir besoin d'une matrice dense
    //         Matrix<CoefficientPrecision> Bdense(t.get_size(), s.get_size());
    //         Matrix<CoefficientPrecision> dense(t.get_size(), s.get_size());
    //         copy_to_dense(*temp, dense.data());
    //         if ((Bts->get_target_cluster() == t) and (Bts->get_source_cluster() == s)) {
    //             // bonne taille on copiuie to dense
    //             copy_to_dense(*Bts, Bdense.data());
    //         } else {
    //             // la il faut extraire le bon sous bloc
    //             Matrix<CoefficientPrecision> BB(Bts->get_target_cluster().get_size(), Bts->get_source_cluster().get_size());
    //             copy_to_dense(*Bts, BB.data());
    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 for (int l = 0; l < s.get_size(); ++l) {
    //                     Bdense(k, l) = BB(k + t.get_offset() - Bts->get_target_cluster().get_offset(), l + s.get_offset() - Bts->get_source_cluster().get_offset());
    //                 }
    //             }
    //         }
    //         auto res = dense - Bdense;
    //         // on regarde si la feuille est admissible
    //         auto adm = this->get_admissibility_condition();
    //         bool is_admissible =
    //             adm->ComputeAdmissibility(t, s, this->m_tree_data->m_eta);
    //         if (is_admissible) {
    //             MatrixGenerator<CoefficientPrecision> generator(res, t.get_offset(), s.get_offset());
    //             LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(generator, *this->get_low_rank_generator(), t, s, -1, this->m_tree_data->m_epsilon);
    //             temp->set_low_rank_data(lr);
    //         } else {
    //             temp->set_dense_data(res);
    //         }
    //     } else {
    //         for (auto &child : temp->get_children()) {
    //             this->Moins(B, child->get_target_cluster(), child->get_source_cluster());
    //         }
    //     }
    // }

    // fonction a appeler sur un gros bloc dense pour le split
    void split(const Matrix<CoefficientPrecision> &data) {
        auto &t     = this->get_target_cluster();
        auto &s     = this->get_source_cluster();
        int depth_t = t.get_depth();
        int depth_s = s.get_depth();

        // std::cout << depth_t << ',' << t.get_maximal_depth() << ',' << depth_s << ',' << s.get_maximal_depth() << std::endl;
        if ((depth_t < t.get_maximal_depth()) && (depth_s < s.get_maximal_depth())) {

            // bool adm = false;
            // if (t == s) {
            //     adm = false;
            // } else {
            bool adm = this->get_admissibility_condition()->ComputeAdmissibility(t, s, this->m_tree_data->m_eta);
            // }
            if (t == s) {
                adm = false;
            }
            int flag = 0;
            if (adm) {

                auto lr_generator = this->get_low_rank_generator();
                VirtualGeneratorNOPermutation<CoefficientPrecision> generator(data, t.get_offset(), s.get_offset());
                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(generator, *lr_generator, t, s, -1, this->m_tree_data->m_epsilon);
                // std::cout << "bloc admissible avec U et V" << lr.Get_U().nb_rows() << ',' << lr.Get_U().nb_cols() << ';' << lr.Get_V().nb_rows() << ',' << lr.Get_V().nb_cols() << std::endl;
                if (lr.Get_U().nb_cols() > 0) {
                    flag = 1;
                    // std::cout << "!!!!!!!!!!!!!" << std::endl;
                    this->set_low_rank_data(lr);
                }
            }
            if (flag == 0) {
                for (auto &t_child : t.get_children()) {
                    for (auto &s_child : s.get_children()) {
                        auto child                              = this->add_child(t_child.get(), s_child.get());
                        Matrix<CoefficientPrecision> data_restr = data.get_block(t_child->get_size(), s_child->get_size(), t_child->get_offset() - t.get_offset(), s_child->get_offset() - s.get_offset());
                        child->split(data_restr);
                    }
                }
            }

        } else {
            this->set_dense_data(data);
        }
    }

    void format() {
        auto &t = this->get_target_cluster();
        auto &s = this->get_source_cluster();
        if ((t.get_depth() < t.get_maximal_depth() && s.get_depth() < s.get_maximal_depth())) {
            if (this->is_dense()) {
                auto M = *this->get_dense_data();
                this->split(M);
            } else if (this->is_low_rank() == false) {
                for (auto &child : this->get_children()) {
                    if (child->is_low_rank() == false) {
                        child->format();
                    }
                }
            }
        }
    }

    void format(const HMatrix &A) {
        auto &t = this->get_target_cluster();
        auto &s = this->get_source_cluster();
        if (A.get_children().size() == 0) {
            if (A.is_low_rank()) {
                this->set_low_rank_data(*A.get_low_rank_data());
            } else {
                auto &M = *this->get_dense_data();
                if (t.get_depth() < t.get_maximal_depth() && s.get_depth() < s.get_maximal_depth()) {
                    this->split(M);
                } else {
                    this->set_dense_data(M);
                }
            }
        } else {
            for (auto &t_child : t.get_children()) {
                for (auto &s_child : s.get_children()) {
                    auto child = this->add_child(t_child.get(), s_child.get());
                    child->format(*A.get_block(t_child->get_size(), s_child->get_size(), t_child->get_offset(), s_child->get_offset()));
                }
            }
        }
    }

    /// a appelé sur unbloc admissible qui n'a pas de lr, c'est un void sur res = Hmatrix(t,s)
    void split_bloc(const Matrix<CoefficientPrecision> &data) {
        auto &t = this->get_target_cluster();
        auto &s = this->get_source_cluster();
        if (t.get_depth() > t.get_maximal_depth() && s.get_depth() > s.get_maximal_depth()) {
            for (auto &t_child : t.get_children()) {
                for (auto &s_child : s.get_children()) {
                    auto child      = this->add_child(t_child.get(), s_child.get());
                    auto data_restr = data.get_block(t_child->get_size(), s_child->get_size(), t_child->get_offset() - t.get_offset(), s_child->get_offset() - s.get_offset());
                    if (t_child == s_child) {
                        child->split_bloc(data_restr);
                    } else {
                        bool is_admissible = this->get_admissibility_condition()->ComputeAdmissibility(child->get_target_cluster(), child->get_source_cluster(), this->m_tree_data->m_eta);
                        if (is_admissible) {
                            auto lr_generator = this->get_low_rank_generator();
                            VirtualGeneratorNOPermutation<CoefficientPrecision> generator(data_restr, t_child->get_offset(), s_child->get_offset());
                            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(generator, *lr_generator, *t_child, *s_child, -1, this->m_tree_data->m_epsilon);
                            if (lr.Get_U().nb_cols() > 0) {
                                child->set_low_rank_data(lr);
                            } else {
                                child->split_bloc(data_restr);
                            }
                        } else {
                            child->split_bloc(data_restr);
                        }
                    }
                }
            }
        } else {
            this->set_dense_data(data);
        }
    }

    // void Moins(const HMatrix *A, const HMatrix *B, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix *res) {
    //     int depth_t        = t.get_depth();
    //     int depth_s        = s.get_depth();
    //     bool is_admissible = A->get_admissibility_condition()->ComputeAdmissibility(t, s, A->m_tree_data->m_eta);
    //     if (is_admissible) {
    //         auto a = A->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //         auto b = B->get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //         Matrix<CoefficientPrecision> adense, bdense;
    //         copy_to_dense(*a, adense.data());
    //         copy_to_dense(*b, bdense.data());
    //         Matrix<CoefficientPrecision> c;
    //         if (!(adense.nb_rows() == t.get_size() && adense.nb_cols() == s.get_size())) {
    //             adense = adense.get_block(t.get_size(), s.get_size(), t.get_offset() - a->get_target_cluster().get_offset(), s.get_offset() - a->get_source_cluster().get_offset());
    //         }
    //         if (!(bdense.nb_rows() == t.get_size() && bdense.nb_cols() == s.get_size())) {
    //             bdense = bdense.get_block(t.get_size(), s.gets_size(), t.get_offset() - b->get_target_cluster().get_offset(), s.get_offset() - b->get_source_cluster().get_offset());
    //         }
    //         auto c = adense - bdense;
    //         res->set_dense_data(c)
    //     }
    //     if (A->get_children() > 0 or B->get_childen() > 0) {
    //         for (auto &t_child : t.get_children()) {
    //             for (auto &s_child : s.get_children()) {
    //                 auto res_child = res->add_child(t_child.get(), s_child.get());
    //                 Moins(A->get_block(t_child->get_size(), s_child->get_size(), t_child->get_offset(), s_child->get_offset()), B->get_block(t_child->get_size(), s_child->get_size(), t_child->get_offset(), s_child->get_offset()), res_child);
    //             }
    //         }
    //     } else {
    //     }
    // }

    ///////////////////////////
    ///// MOINS QUI MARCHER MAIS C'EST VRAIMENT PAS DINGUE
    // void Moins(const HMatrix &B) {
    //     this->set_leaves_in_cache();
    //     auto leaves = this->get_leaves();
    //     for (auto &l : leaves) {
    //         auto b = B.get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
    //         if ((!(b->is_dense())) and (!(b->is_low_rank()))) {
    //             // LR+ Hmatrix
    //             Matrix<CoefficientPrecision> bdense(b->get_target_cluster().get_size(), b->get_source_cluster().get_size());
    //             Matrix<CoefficientPrecision> adense(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //             copy_to_dense(*l, adense.data());
    //             copy_to_dense(*b, bdense.data());
    //             if ((b->get_target_cluster().get_size() == l->get_target_cluster().get_size()) and (b->get_source_cluster().get_size() == l->get_source_cluster().get_size())) {
    //                 auto c = adense - bdense;
    //                 this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
    //                 // std::cout << "Moins a push un bloc de norme " << normFrob(c) << std::endl;
    //                 // this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->split(c);
    //             } else {
    //                 Matrix<CoefficientPrecision> brestr(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //                 for (int i = 0; i < l->get_target_cluster().get_size(); ++i) {
    //                     for (int j = 0; j < l->get_source_cluster().get_size(); ++j) {
    //                         brestr(i, j) = bdense(i + l->get_target_cluster().get_offset() - b->get_target_cluster().get_offset(), j + l->get_source_cluster().get_offset() - b->get_source_cluster().get_offset());
    //                     }
    //                 }
    //                 auto c = adense - brestr;
    //                 this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
    //                 // std::cout << "Moins u ush un bloc de norme " << normFrob(c) << std::endl;
    //                 // this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_targekt_cluster().get_offset(), l->get_source_cluster().get_offset())->split(c);
    //             }
    //         } else {
    //             if ((b->get_target_cluster().get_size() == l->get_target_cluster().get_size()) and (b->get_source_cluster().get_size() == l->get_source_cluster().get_size())) {
    //                 if ((b->is_dense()) or (l->is_dense())) {
    //                     Matrix<CoefficientPrecision> adense(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //                     Matrix<CoefficientPrecision> bdense(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //                     copy_to_dense(*l, adense.data());
    //                     copy_to_dense(*b, bdense.data());
    //                     auto c = adense - bdense;
    //                     this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
    //                     // std::cout << "on a push un bloc de norme : " << normFrob(c) << std::endl;
    //                     // this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->split(c);
    //                 } else {
    //                     Matrix<CoefficientPrecision> adense(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //                     Matrix<CoefficientPrecision> bdense(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //                     copy_to_dense(*l, adense.data());
    //                     copy_to_dense(*b, bdense.data());
    //                     auto c            = adense - bdense;
    //                     auto lr_generator = this->get_low_rank_generator();
    //                     VirtualGeneratorNOPermutation<CoefficientPrecision> generator(c, l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
    //                     LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(generator, *lr_generator, l->get_target_cluster(), l->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
    //                     if (lr.Get_U().nb_cols() > 0) {
    //                         this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(lr);
    //                     } else {
    //                         this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
    //                         // this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->split(c);
    //                     }
    //                     // this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(lr);
    //                     // auto uu = lr.Get_U();
    //                     // auto vv = lr.Get_V();
    //                     // std::cout << "on a push un bloc de rang faible de norme : " << normFrob(uu * vv) << std::endl;
    //                 }
    //             } else {
    //                 // cas ou le bloc n'existerait pas dasn B , ce cas ne devrait pas exister
    //                 Matrix<CoefficientPrecision> bdense(b->get_target_cluster().get_size(), b->get_source_cluster().get_size());
    //                 Matrix<CoefficientPrecision> brestr(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //                 Matrix<CoefficientPrecision> adense(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //                 copy_to_dense(*l, adense.data());
    //                 copy_to_dense(*b, bdense.data());
    //                 for (int i = 0; i < l->get_target_cluster().get_size(); ++i) {
    //                     for (int j = 0; j < l->get_source_cluster().get_size(); ++j) {
    //                         brestr(i, j) = bdense(i + l->get_target_cluster().get_offset() - b->get_target_cluster().get_offset(), j + l->get_source_cluster().get_offset() - b->get_source_cluster().get_offset());
    //                     }
    //                 }
    //                 auto c = adense - brestr;
    //                 this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
    //                 // std::cout << ",;,;,;,;,;,;,;,;,;,;,;,;,last case on a pyush avec moins un bloc de norme :" << normFrob(c) << std::endl;
    //                 // this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->split(c);
    //             }
    //         }
    //         // Je rajoute ca pour être sure de pas rater des rang faibles ;
    //         // auto bloc = this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
    //         // bool is_admissible = this->get_admissibility_condition()->ComputeAdmissibility(l->get_target_cluster(), l->get_source_cluster(), this->m_tree_data->m_eta);
    //         // // std::cout << "admissible ? " << is_admissible << std::endl;
    //         // // std::cout << "déja low rank ? " << !(bloc->get_low_rank_data() == nullptr) << std::endl;
    //         // if (is_admissible and (bloc->get_low_rank_data() == nullptr)) {
    //         //     auto lr_generator = this->get_low_rank_generator();
    //         //     // std::cout << "adm!" << std::endl;
    //         //     Matrix<CoefficientPrecision> dense_l(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //         //     copy_to_dense(*this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset()), dense_l.data());
    //         //     VirtualGeneratorNOPermutation<CoefficientPrecision> generator(dense_l, l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
    //         //     LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(generator, *lr_generator, l->get_target_cluster(), l->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
    //         //     auto uu = lr.Get_U();
    //         //     if ((uu.nb_rows() > 0) and (uu.nb_cols() > 0) and (uu.nb_cols() < l->get_target_cluster().get_size() / 2)) {
    //         //         this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(lr);
    //         //         // std::cout << "! !    !   " << std::endl;
    //         //     } else {
    //         //         // il faut descendre
    //         //     }
    //         // }
    //         // if (bloc->is_dense()) {
    //         //     bloc->split(*bloc->get_dense_data());
    //         // }
    //     }
    //     // this->format();
    // }
    /////////////////////////////////
    void Moins(const HMatrix &B) {
        this->set_leaves_in_cache();
        auto leaves = this->get_leaves();
        for (auto &l : leaves) {
            auto b = B.get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
            if ((!(b->is_dense())) and (!(b->is_low_rank()))) {
                // LR+ Hmatrix
                Matrix<CoefficientPrecision> bdense(b->get_target_cluster().get_size(), b->get_source_cluster().get_size());
                Matrix<CoefficientPrecision> adense(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                copy_to_dense(*l, adense.data());
                copy_to_dense(*b, bdense.data());
                if ((b->get_target_cluster().get_size() == l->get_target_cluster().get_size()) and (b->get_source_cluster().get_size() == l->get_source_cluster().get_size())) {
                    auto c = adense - bdense;
                    this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
                    // std::cout << "Moins a push un bloc de norme " << normFrob(c) << std::endl;
                    // this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->split(c);
                } else {
                    Matrix<CoefficientPrecision> brestr(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                    for (int i = 0; i < l->get_target_cluster().get_size(); ++i) {
                        for (int j = 0; j < l->get_source_cluster().get_size(); ++j) {
                            brestr(i, j) = bdense(i + l->get_target_cluster().get_offset() - b->get_target_cluster().get_offset(), j + l->get_source_cluster().get_offset() - b->get_source_cluster().get_offset());
                        }
                    }
                    // bdense.copy_submatrix(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset() - b->get_target_cluster().get_offset(), l->get_source_cluster().get_offset() - b->get_source_cluster().get_offset());
                    // auto c = adense - brestr;
                    Matrix<CoefficientPrecision> c(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                    moins(adense.nb_rows() * adense.nb_cols(), adense.data(), brestr.data(), c.data());
                    this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
                    // std::cout << "Moins u ush un bloc de norme " << normFrob(c) << std::endl;
                    // this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_targekt_cluster().get_offset(), l->get_source_cluster().get_offset())->split(c);
                }
            } else {
                if ((b->get_target_cluster().get_size() == l->get_target_cluster().get_size()) and (b->get_source_cluster().get_size() == l->get_source_cluster().get_size())) {
                    if ((b->is_dense()) or (l->is_dense())) {
                        Matrix<CoefficientPrecision> adense(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                        Matrix<CoefficientPrecision> bdense(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                        copy_to_dense(*l, adense.data());
                        copy_to_dense(*b, bdense.data());
                        // auto c = adense - bdense;
                        Matrix<CoefficientPrecision> c(adense.nb_rows(), adense.nb_cols());
                        moins(adense.nb_rows() * adense.nb_cols(), adense.data(), bdense.data(), c.data());
                        this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
                        // std::cout << "on a push un bloc de norme : " << normFrob(c) << std::endl;
                        // this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->split(c);
                    } else {
                        Matrix<CoefficientPrecision> adense(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                        Matrix<CoefficientPrecision> bdense(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                        copy_to_dense(*l, adense.data());
                        copy_to_dense(*b, bdense.data());
                        // auto c            = adense - bdense;
                        Matrix<CoefficientPrecision> c(adense.nb_rows(), adense.nb_cols());
                        moins(adense.nb_rows() * adense.nb_cols(), adense.data(), bdense.data(), c.data());
                        auto lr_generator = this->get_low_rank_generator();
                        VirtualGeneratorNOPermutation<CoefficientPrecision> generator(c, l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
                        LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(generator, *lr_generator, l->get_target_cluster(), l->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
                        if (lr.Get_U().nb_cols() > 0) {
                            this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(lr);
                        } else {
                            this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
                            // this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->split(c);
                        }
                        // this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(lr);
                        // auto uu = lr.Get_U();
                        // auto vv = lr.Get_V();
                        // std::cout << "on a push un bloc de rang faible de norme : " << normFrob(uu * vv) << std::endl;
                    }
                } else {
                    // cas ou le bloc n'existerait pas dasn B , ce cas ne devrait pas exister
                    Matrix<CoefficientPrecision> bdense(b->get_target_cluster().get_size(), b->get_source_cluster().get_size());
                    Matrix<CoefficientPrecision> brestr(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                    Matrix<CoefficientPrecision> adense(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
                    copy_to_dense(*l, adense.data());
                    copy_to_dense(*b, bdense.data());
                    for (int i = 0; i < l->get_target_cluster().get_size(); ++i) {
                        for (int j = 0; j < l->get_source_cluster().get_size(); ++j) {
                            brestr(i, j) = bdense(i + l->get_target_cluster().get_offset() - b->get_target_cluster().get_offset(), j + l->get_source_cluster().get_offset() - b->get_source_cluster().get_offset());
                        }
                    }
                    // auto c = adense - brestr;
                    Matrix<CoefficientPrecision> c(adense.nb_rows(), adense.nb_cols());
                    moins(adense.nb_rows() * adense.nb_cols(), adense.data(), brestr.data(), c.data());
                    this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
                    // std::cout << ",;,;,;,;,;,;,;,;,;,;,;,;,last case on a pyush avec moins un bloc de norme :" << normFrob(c) << std::endl;
                    // this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->split(c);
                }
            }
            // Je rajoute ca pour être sure de pas rater des rang faibles ;
            // auto bloc = this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
            // bool is_admissible = this->get_admissibility_condition()->ComputeAdmissibility(l->get_target_cluster(), l->get_source_cluster(), this->m_tree_data->m_eta);
            // // std::cout << "admissible ? " << is_admissible << std::endl;
            // // std::cout << "déja low rank ? " << !(bloc->get_low_rank_data() == nullptr) << std::endl;
            // if (is_admissible and (bloc->get_low_rank_data() == nullptr)) {
            //     auto lr_generator = this->get_low_rank_generator();
            //     // std::cout << "adm!" << std::endl;
            //     Matrix<CoefficientPrecision> dense_l(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
            //     copy_to_dense(*this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset()), dense_l.data());
            //     VirtualGeneratorNOPermutation<CoefficientPrecision> generator(dense_l, l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
            //     LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(generator, *lr_generator, l->get_target_cluster(), l->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
            //     auto uu = lr.Get_U();
            //     if ((uu.nb_rows() > 0) and (uu.nb_cols() > 0) and (uu.nb_cols() < l->get_target_cluster().get_size() / 2)) {
            //         this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(lr);
            //         // std::cout << "! !    !   " << std::endl;
            //     } else {
            //         // il faut descendre
            //     }
            // }
            // if (bloc->is_dense()) {
            //     bloc->split(*bloc->get_dense_data());
            // }
        }
        // this->format();
    }

    // void Moins(const HMatrix &B) {
    //     this->set_leaves_in_cache();
    //     auto leaves = this->get_leaves();
    //     for (auto &l : leaves) {
    //         auto b = B.get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
    //         // on veut faire l-b
    //         Matrix<CoefficientPrecision> bdense(b->get_target_cluster().get_size(), b->get_source_cluster().get_size());
    //         copy_to_dense(*b, bdense.data());
    //         // on regarde si elle a la même taille que l et siuinon on extrait
    //         Matrix<CoefficientPrecision> dense_leaf(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //         copy_to_dense(*l, dense_leaf.data());
    //         Matrix<CoefficientPrecision> res(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //         if ((bdense.nb_rows() == l->get_target_cluster().get_size()) and (bdense.nb_cols() == l->get_source_cluster().get_size())) {
    //             // bonne taille
    //             res = dense_leaf - bdense;
    //         } else {
    //             // on extrait le bon bloàc dans b
    //             Matrix<CoefficientPrecision> brestr(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //             for (int k = 0; k < l->get_target_cluster().get_size(); ++k) {
    //                 for (int m = 0; m < l->get_source_cluster().get_size(); ++m) {
    //                     brestr(k, m) = bdense(k + l->get_target_cluster().get_offset() - b->get_target_cluster().get_offset(), m + l->get_source_cluster().get_offset() - b->get_source_cluster().get_offset());
    //                 }
    //             }
    //             res = dense_leaf - brestr;
    //         }
    //         // On regarde si c'est admissible ou pas
    //         bool is_admissible = this->get_admissibility_condition()->ComputeAdmissibility(l->get_target_cluster(), l->get_source_cluster(), this->m_tree_data->m_eta);
    //         if (is_admissible) {
    //             auto lr_generator = this->get_low_rank_generator();
    //             VirtualGeneratorNOPermutation<CoefficientPrecision> generator(res, l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
    //             LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(generator, *lr_generator, l->get_target_cluster(), l->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
    //             this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(lr);
    //         } else {
    //             this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(res);
    //         }
    //     }
    // }
    // void Moins(const HMatrix<CoefficientPrecision, CoordinatePrecision> &B) {
    //     this->set_leaves_in_cache();
    //     auto leaves = this->get_leaves();
    //     for (auto &l : leaves) {
    //         auto a = this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
    //         auto b = B.get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
    //         if (!((b->get_target_cluster() == l->get_target_cluster()) and (b->get_source_cluster() == l->get_source_cluster()))) {
    //             Matrix<CoefficientPrecision> temp(b->get_target_cluster().get_size(), b->get_source_cluster().get_size());

    //             copy_to_dense(*b, temp.data());
    //             Matrix<CoefficientPrecision> restr(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //             for (int k = 0; k < l->get_target_cluster().get_size(); ++k) {
    //                 for (int kk = 0; kk < l->get_source_cluster().get_size(); ++kk) {
    //                     restr(k, kk) = temp(k + l->get_target_cluster().get_offset() - b->get_target_cluster().get_offset(), kk + l->get_source_cluster().get_offset() - b->get_source_cluster().get_offset());
    //                 }
    //             }
    //             Matrix<CoefficientPrecision> aa(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //             copy_to_dense(*a, aa.data());
    //             auto c = aa - restr;
    //             if (l->is_dense()) {

    //                 this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
    //             } else {
    //                 auto lr = this->get_low_rank_generator();
    //                 MatrixGenerator<CoefficientPrecision> matgen(c, l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
    //                 LowRankMatrix<CoefficientPrecision, CoordinatePrecision> llr(matgen, *lr, l->get_target_cluster(), l->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
    //                 this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(llr);
    //             }

    //         } else {

    //             Matrix<CoefficientPrecision> aa(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //             Matrix<CoefficientPrecision> bb(l->get_target_cluster().get_size(), l->get_source_cluster().get_size());
    //             copy_to_dense(*a, aa.data());
    //             copy_to_dense(*b, bb.data());
    //             auto c = aa - bb;
    //             if (l->is_dense()) {
    //                 this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(c);
    //             } else {
    //                 auto lr = this->get_low_rank_generator();
    //                 MatrixGenerator<CoefficientPrecision> matgen(c, l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset());
    //                 LowRankMatrix<CoefficientPrecision, CoordinatePrecision> llr(matgen, *lr, l->get_target_cluster(), l->get_source_cluster());
    //                 this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(llr);
    //             }
    //         }
    //     }
    // }
    ///////////////////////////////////////////////////
    //// mult -> Test PASSED : YES
    // Hmatrice * matrice
    Matrix<CoefficientPrecision> hmat_mat(const Matrix<CoefficientPrecision> &u) const {
        std::vector<CoefficientPrecision> u_row_major(u.nb_rows() * u.nb_cols());
        for (int k = 0; k < u.nb_rows(); ++k) {
            for (int kk = 0; kk < u.nb_cols(); ++kk) {
                u_row_major[k * u.nb_cols() + kk] = u.data()[k + kk * u.nb_rows()];
            }
        }
        Matrix<CoefficientPrecision> hu(this->get_target_cluster().get_size(), u.nb_cols());
        this->add_matrix_product_row_major('N', 1.0, u_row_major.data(), 0.0, hu.data(), u.nb_cols());
        Matrix<CoefficientPrecision> Hu(this->get_target_cluster().get_size(), u.nb_cols());
        for (int k = 0; k < this->get_target_cluster().get_size(); ++k) {
            for (int kk = 0; kk < u.nb_cols(); ++kk) {
                Hu.data()[kk * this->get_target_cluster().get_size() + k] = hu.data()[k * u.nb_cols() + kk];
            }
        }
        return (Hu);
    }

    // matrice * Hmatrice -> beacoup trop long a cause
    Matrix<CoefficientPrecision> mat_hmat(const Matrix<CoefficientPrecision> &v) const {
        auto v_transp = v.transp(v);
        std::vector<CoefficientPrecision> v_row_major(v.nb_cols() * v.nb_rows());
        for (int k = 0; k < v.nb_rows(); ++k) {
            for (int l = 0; l < v.nb_cols(); ++l) {
                v_row_major[l * v.nb_rows() + k] = v_transp.data()[k * v.nb_cols() + l];
            }
        }
        Matrix<CoefficientPrecision> vk(this->get_source_cluster().get_size(), v.nb_rows());
        this->add_matrix_product_row_major('T', 1.0, v_row_major.data(), 0.0, vk.data(), v.nb_rows());
        Matrix<CoefficientPrecision> vK(this->get_source_cluster().get_size(), v.nb_rows());
        for (int k = 0; k < this->get_source_cluster().get_size(); ++k) {
            for (int l = 0; l < v.nb_rows(); ++l) {
                vK.data()[l * this->get_source_cluster().get_size() + k] = vk.data()[k * v.nb_rows() + l];
            }
        }
        Matrix<CoefficientPrecision> p = vk.transp(vK);
        return (p);
    }
    ////////////////////////////

    // Matrix<CoefficientPrecision> mat_hmat(const Matrix<CoefficientPrecision> &v) const {
    //     Matrix<CoefficientPrecision> dense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
    //     copy_to_dense(*this, dense.data());
    //     return (dense * v);
    // }

    //////////////////////////////
    ///// FONCTION INUTILES SERVANT JUSTE A FAIRE DES TESTS SUR FORWARD ET BACKWARD
    ///// DELETE STATE : WAIT
    void Get_U(const HMatrix &M) {
        if (M.get_children().size() == 0) {
            if (M.get_target_cluster() == M.get_source_cluster()) {
                Matrix<CoefficientPrecision> dense_diag(M.get_target_cluster().get_size(), M.get_target_cluster().get_size());
                copy_to_dense(M, dense_diag.data());
                Matrix<CoefficientPrecision> dense_up(M.get_target_cluster().get_size(), M.get_target_cluster().get_size());
                for (int k = 0; k < M.get_target_cluster().get_size(); ++k) {
                    for (int l = k; l < M.get_target_cluster().get_size(); ++l) {
                        dense_up(k, l) = dense_diag(k, l);
                    }
                }
                this->set_dense_data(dense_up);
            } else {
                if (M.is_dense()) {
                    this->set_dense_data(*M.get_dense_data());
                } else {
                    this->set_low_rank_data(*M.get_low_rank_data());
                }
            }
        } else {
            for (auto &child : M.get_children()) {
                auto child_U = this->add_child(&child->get_target_cluster(), &child->get_source_cluster());
                if (child->get_target_cluster().get_offset() <= child->get_source_cluster().get_offset()) {
                    child_U->Get_U(*child);
                } else {
                    Matrix<CoefficientPrecision> zero(child->get_target_cluster().get_size(), child->get_source_cluster().get_size());
                    child_U->set_dense_data(zero);
                }
            }
        }
    }

    void Get_L(const HMatrix &M) {
        if (M.get_children().size() == 0) {
            if (M.get_target_cluster() == M.get_source_cluster()) {
                Matrix<CoefficientPrecision> dense_diag(M.get_target_cluster().get_size(), M.get_source_cluster().get_size());
                copy_to_dense(M, dense_diag.data());
                Matrix<CoefficientPrecision> dense_down(M.get_target_cluster().get_size(), M.get_source_cluster().get_size());
                for (int k = 0; k < M.get_target_cluster().get_size(); ++k) {
                    dense_down(k, k) = 1.0;
                    for (int l = 0; l < k; ++l) {
                        dense_down(k, l) = dense_diag(k, l);
                    }
                }
                this->set_dense_data(dense_down);
            } else {
                if (M.is_dense()) {
                    this->set_dense_data(*M.get_dense_data());
                } else {
                    this->set_low_rank_data(*M.get_low_rank_data());
                }
            }
        } else {
            for (auto &child : M.get_children()) {
                auto child_L = this->add_child(&child->get_target_cluster(), &child->get_source_cluster());
                if (child->get_target_cluster().get_offset() >= child->get_source_cluster().get_offset()) {
                    child_L->Get_L(*child);
                } else {
                    Matrix<CoefficientPrecision> zero(child->get_target_cluster().get_size(), child->get_source_cluster().get_size());
                    child_L->set_dense_data(zero);
                }
            }
        }
    }
    // void Get_U(const HMatrix &M) {
    //     // auto Lts       = this->get_block(M.get_target_cluster().get_size(), M.get_source_cluster().get_size(), M.get_target_cluster().get_offset(), M.get_source_cluster().get_offset());
    //     auto &children = M.get_children();
    //     if (children.size() == 0) {
    //         if (M.get_target_cluster() == M.get_source_cluster()) {
    //             Matrix<CoefficientPrecision> up(M.get_target_cluster().get_size(), M.get_target_cluster().get_size());
    //             auto data = *M.get_dense_data();
    //             for (int k = 0; k < M.get_target_cluster().get_size(); ++k) {
    //                 for (int l = k; l < M.get_target_cluster().get_size(); ++l) {
    //                     up(k, l) = data(k, l);
    //                 }
    //             }
    //             this->set_dense_data(up);
    //         } else {
    //             if (M.is_dense()) {
    //                 Matrix<CoefficientPrecision> data = *M.get_dense_data();
    //                 this->set_dense_data(data);
    //             } else {
    //                 auto lr = M.get_low_rank_data();
    //                 this->set_low_rank_data(*lr);
    //             }
    //         }
    //     } else {
    //         for (auto &child : children) {
    //             if (child->get_target_cluster().get_offset() <= child->get_source_cluster().get_offset()) {
    //                 auto Uson = this->add_child(&child->get_target_cluster(), &child->get_source_cluster());
    //                 Uson->Get_U(*child);
    //             }
    //         }
    //     }
    // }

    // void Get_L(const HMatrix &M) {
    //     // auto Lts       = this->get_block(M.get_target_cluster().get_size(), M.get_source_cluster().get_size(), M.get_target_cluster().get_offset(), M.get_source_cluster().get_offset());
    //     auto &children = M.get_children();
    //     if (children.size() == 0) {
    //         if (M.get_target_cluster() == M.get_source_cluster()) {
    //             Matrix<CoefficientPrecision> down(M.get_target_cluster().get_size(), M.get_target_cluster().get_size());
    //             auto data = *M.get_dense_data();
    //             for (int k = 0; k < M.get_target_cluster().get_size(); ++k) {
    //                 down(k, k) = 1.0;
    //                 for (int l = 0; l < k; ++l) {
    //                     down(k, l) = data(k, l);
    //                 }
    //             }
    //             this->set_dense_data(down);
    //         } else {
    //             if (M.is_dense()) {
    //                 Matrix<CoefficientPrecision> data = *M.get_dense_data();
    //                 this->set_dense_data(data);
    //             } else {
    //                 auto lr = M.get_low_rank_data();
    //                 this->set_low_rank_data(*lr);
    //             }
    //         }
    //     } else {
    //         for (auto &child : children) {
    //             if (child->get_target_cluster().get_offset() >= child->get_source_cluster().get_offset()) {
    //                 auto Lson = this->add_child(&child->get_target_cluster(), &child->get_source_cluster());
    //                 Lson->Get_L(*child);
    //             }
    //         }
    //     }
    // }

    // void Get_U(HMatrix &M) const {
    //     if (this->get_children().size() > 0) {
    //         for (auto &child : this->get_children()) {
    //             if (child->get_target_cluster().get_offset() <= child->get_source_cluster().get_offset()) {
    //                 child->Get_U(M);
    //             }
    //         }
    //     } else {
    //         if (this->get_target_cluster() == this->get_source_cluster()) {
    //             auto m = *(this->get_dense_data());
    //             Matrix<CoefficientPrecision> mm(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
    //             for (int k = 0; k < this->get_target_cluster().get_size(); ++k) {
    //                 for (int l = k; l < this->get_source_cluster().get_size(); ++l) {
    //                     mm(k, l) = m(k, l);
    //                 }
    //             }
    //             M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset())->set_dense_data(mm);
    //             M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset())->delete_children();
    //         } else {

    //             auto &sub = *this->get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
    //             auto temp = M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
    //             temp      = &sub;
    //             M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset())->delete_children();
    //         }
    //     }
    // }

    // void Get_L(HMatrix &M) const {
    //     // std::cout << " cluster " << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << ',' << this->get_target_cluster().get_offset() << ',' << this->get_source_cluster().get_offset() << std::endl;
    //     if (this->get_children().size() > 0) {
    //         for (auto &child : this->get_children()) {
    //             if (child->get_target_cluster().get_offset() <= child->get_source_cluster().get_offset()) {
    //                 child->Get_L(M);
    //             }
    //         }
    //     } else {
    //         if (this->get_target_cluster() == this->get_source_cluster()) {
    //             auto m = *(this->get_dense_data());
    //             Matrix<CoefficientPrecision> mm(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
    //             for (int k = 0; k < this->get_target_cluster().get_size(); ++k) {
    //                 mm(k, k) = 1;
    //                 for (int l = 0; l < k; ++l) {
    //                     mm(k, l) = m(k, l);
    //                 }
    //             }
    //             M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset())->set_dense_data(mm);
    //             // M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset())->delete_children();

    //         } else if (this->get_target_cluster().get_offset() < this->get_source_cluster().get_offset()) {

    //             auto &sub = *this->get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
    //             auto temp = M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset());
    //             temp      = &sub;
    //             // std::cout << " cluster delete" << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << ',' << this->get_target_cluster().get_offset() << ',' << this->get_source_cluster().get_offset() << std::endl;
    //             // M.get_block(this->get_target_cluster().get_size(), this->get_source_cluster().get_size(), this->get_target_cluster().get_offset(), this->get_source_cluster().get_offset())->delete_children();
    //         }
    //     }
    // }

    ////////////////
    /// TARNSPOSE
    /// DELETE STAIT : ?
    void transp(const HMatrix &H) {
        if (H.get_children().size() > 0) {
            for (auto &child : H.get_children()) {
                auto bloc = this->get_block(H.get_source_cluster().get_size(), H.get_target_cluster().get_size(), H.get_source_cluster().get_offset(), H.get_target_cluster().get_offset())->add_child(&child->get_source_cluster(), &child->get_target_cluster());
                bloc->transp(*child);
            }
        } else {
            if (H.is_dense()) {
                auto M = *H.get_dense_data();
                auto N = M.transp(M);
                this->set_dense_data(N);
            } else {
                auto u  = H.get_low_rank_data()->Get_U();
                auto v  = H.get_low_rank_data()->Get_V();
                auto uu = u.transp(u);
                auto vv = v.transp(v);
                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> uv(vv, uu);
                this->set_low_rank_data(uv);
            }
        }
    }

    /////////////////////////
    /// FORWARD_SUBSTITUTION
    /// trouver x tq LX = y
    /// TESTS PASSED : YES

    void forward_substitution_extract(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &b) const {
        auto Lt = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Lt == nullptr) {
            Matrix<CoefficientPrecision> l_dense(t.get_size(), t.get_size());
            if (t.get_offset() + 16 / 32 == 0) {
                auto l_big = *this->get_block(2 * t.get_size(), 2 * t.get_size(), t.get_offset() - 16, t.get_offset() - 16)->get_dense_data();
                for (int k = 0; k < t.get_size(); ++k) {
                    for (int l = 0; l < t.get_size(); ++l) {
                        l_dense(k, l) = l_big(k + 16, l + 16);
                    }
                }
            } else {
                auto l_big = *this->get_block(2 * t.get_size(), 2 * t.get_size(), t.get_offset(), t.get_offset())->get_dense_data();
                for (int k = 0; k < t.get_size(); ++k) {
                    for (int l = 0; l < t.get_size(); ++l) {
                        l_dense(k, l) = l_big(k, l);
                    }
                }
            }
            for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
                y[j] = b[j];
                for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
                    b[i] = b[i] - l_dense(i - t.get_offset(), j - t.get_offset()) * y[j];
                }
            }
        } else {
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
    }

    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////// VRAI FORWARD ET FORWARD T (08/12)
    // void forward_substitution_s(const Cluster<CoordinatePrecision> &t, const int &offset_0, std::vector<CoefficientPrecision> &y_in, std::vector<CoefficientPrecision> &x_out) const {
    //     auto L = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
    //     if ((L->get_target_cluster().get_size() == t.get_size()) and (L->get_source_cluster().get_size() == t.get_size())) {
    //         if (L->is_dense()) {
    //             auto ldense = *L->get_dense_data();
    //             for (int j = 0; j < t.get_size(); ++j) {
    //                 x_out[j + t.get_offset() - offset_0] = y_in[j + t.get_offset() - offset_0];
    //                 std::cout << "leaf 0: " << norm2(y_in) << ',' << norm2(x_out) << " l " << normFrob(ldense) << "   et  " << ldense.nb_rows() << ',' << ldense.nb_cols() << std::endl;
    //                 for (int i = j + 1; i < t.get_size(); ++i) {
    //                     // std::cout << "   leaf 1 : " << i << "     " << norm2(y_in) << ',' << norm2(x_out) << " l " << normFrob(ldense) << "   et  " << ldense.nb_rows() << ',' << ldense.nb_cols() << std::endl;
    //                     // std::cout << " :::::::::::: " << norm2(y_in) << "     et " << y_in[i + t.get_offset() - offset_0] << "    ,    " << x_out[j + t.get_offset() - offset_0] << "       ,      " << ldense(i, j) << "--------------->" << ldense(i, j) * x_out[j + t.get_offset() - offset_0] << "     ,     " << y_in[i + t.get_offset() - offset_0] - ldense(i, j) * x_out[j + t.get_offset() - offset_0] << std::endl;
    //                     auto r = y_in[i + t.get_offset() - offset_0] - ldense(i, j) * x_out[j + t.get_offset() - offset_0];
    //                     std::cout << r << "  et   " << y_in[i + t.get_offset() - offset_0] << std::endl;
    //                     y_in[i + t.get_offset() - offset_0] = r;
    //                     std::cout << "      leaf 2 : " << norm2(y_in) << ',' << norm2(x_out) << " l " << normFrob(ldense) << "   et  " << ldense.nb_rows() << ',' << ldense.nb_cols() << std::endl;
    //                     std::cout << std::endl;
    //                 }
    //             }

    //         } else {
    //             auto &t_children = t.get_children();
    //             for (int j = 0; j < t_children.size(); ++j) {
    //                 int offset_j = t_children[j]->get_offset() - offset_0;
    //                 L->forward_substitution_s(*t_children[j], offset_0, y_in, x_out);
    //                 std::vector<CoefficientPrecision> x_j(x_out.begin() + offset_j, x_out.begin() + offset_j + t_children[j]->get_size());
    //                 for (int i = j + 1; i < t_children.size(); ++i) {
    //                     int offset_i = t_children[i]->get_offset() - offset_0;
    //                     std::vector<CoefficientPrecision> y_i(y_in.begin() + offset_i, y_in.begin() + offset_i + t_children[i]->get_size());
    //                     // std::cout << "avant " << norm2(y_in) << ',' << norm2(x_out) << std::endl;
    //                     // auto yy = y_i;
    //                     L->get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset())->add_vector_product('N', -1.0, x_j.data(), 1.0, y_i.data());
    //                     // Matrix<CoefficientPrecision> test(t_children[i]->get_size(), t_children[j]->get_size());
    //                     // copy_to_dense(*L->get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset()), test.data());

    //                     // std::cout << "???" << normFrob(test) << "::::" << norm2(y_i - (yy - test * x_j)) << std::endl;
    //                     // std::cout << "après  " << norm2(y_in) << ',' << norm2(x_out) << std::endl;

    //                     // std::cout << "aprés " << norm2(y_i) << ',' << norm2(x_j) << std::endl;
    //                     std::copy(y_i.begin(), y_i.end(), y_in.begin() + offset_i);
    //                 }
    //             }
    //         }
    //     } else {
    //         // cas chelou parce qu'il peut il y avoir des rand blocs dense
    //         std::cout << "APPELL de forward sur un bloc trop petit" << std::endl;
    //         Matrix<CoefficientPrecision> ldense = *L->get_dense_data();
    //         int offset                          = t.get_offset() - L->get_target_cluster().get_offset();
    //         Matrix<CoefficientPrecision> lrestr(t.get_size(), t.get_size());
    //         for (int k = 0; k < t.get_size(); ++k) {
    //             lrestr(k, k) = 1.0;
    //             for (int l = 0; l < k; ++l) {
    //                 lrestr(k, l) = ldense(k + offset, l + offset);
    //             }
    //         }
    //         for (int j = 0; j < t.get_size(); ++j) {
    //             x_out[j + t.get_offset() - offset_0] = y_in[j + t.get_offset() - offset_0];
    //             for (int i = j + 1; i < t.get_size(); ++i) {
    //                 y_in[i + t.get_offset() - offset_0] = y_in[i + t.get_offset() - offset_0] - lrestr(i, j) * x_out[j + t.get_offset() - offset_0];
    //             }
    //         }
    //     }
    // }
    ////////////////////////////////////////////
    ////// forward substitution
    void forward_substitution_s(const Cluster<CoordinatePrecision> &t, const int &offset_0, std::vector<CoefficientPrecision> &y_in, std::vector<CoefficientPrecision> &x_out) const {
        auto Lt = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if ((Lt->get_target_cluster().get_size() == t.get_size()) and (Lt->get_source_cluster().get_size() == t.get_size())) {
            if (Lt->is_dense()) {
                auto ldense = *Lt->get_dense_data();
                for (int j = 0; j < t.get_size(); ++j) {
                    x_out[j + t.get_offset() - offset_0] = y_in[j + t.get_offset() - offset_0];
                    for (int i = j + 1; i < t.get_size(); ++i) {
                        y_in[i + t.get_offset() - offset_0] = y_in[i + t.get_offset() - offset_0] - ldense(i, j) * x_out[j + t.get_offset() - offset_0];
                    }
                }
            } else {
                auto &t_children = t.get_children();
                for (int j = 0; j < t_children.size(); ++j) {
                    int offset_j = t_children[j]->get_offset() - offset_0;
                    Lt->forward_substitution_s(*t_children[j], offset_0, y_in, x_out);
                    std::vector<CoefficientPrecision> x_j(x_out.begin() + offset_j, x_out.begin() + offset_j + t_children[j]->get_size());
                    for (int i = j + 1; i < t_children.size(); ++i) {
                        int offset_i = t_children[i]->get_offset() - offset_0;
                        std::vector<CoefficientPrecision> y_i(y_in.begin() + offset_i, y_in.begin() + offset_i + t_children[i]->get_size());
                        Lt->get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset())->add_vector_product('N', -1.0, x_j.data(), 1.0, y_i.data());
                        std::copy(y_i.begin(), y_i.end(), y_in.begin() + offset_i);
                    }
                }
            }
        } else {
            std::cout << "noooooooooooooooooooooo" << std::endl;
            std::cout << this->get_target_cluster().get_size() << ',' << Lt->get_target_cluster().get_size() << ',' << t.get_size() << std::endl;
            std::cout << Lt->get_children().size() << std::endl;
        }
    }

    void forward_substitution_T_s(const Cluster<CoordinatePrecision> &t, const int &offset_0, std::vector<CoefficientPrecision> &y_in, std::vector<CoefficientPrecision> &x_out) const {
        auto Ut = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if ((Ut->get_target_cluster().get_size() == t.get_size()) and (Ut->get_source_cluster().get_size() == t.get_size())) {
            if (Ut->is_dense()) {
                auto udense = *Ut->get_dense_data();
                for (int j = 0; j < t.get_size(); ++j) {
                    x_out[j + t.get_offset() - offset_0] = y_in[j + t.get_offset() - offset_0] / udense(j, j);
                    for (int i = j + 1; i < t.get_size(); ++i) {
                        y_in[i + t.get_offset() - offset_0] = y_in[i + t.get_offset() - offset_0] - udense(j, i) * x_out[j + t.get_offset() - offset_0];
                    }
                }
            } else {
                auto &t_children = t.get_children();
                for (int j = 0; j < t_children.size(); ++j) {
                    int offset_j = t_children[j]->get_offset() - offset_0;
                    Ut->forward_substitution_T_s(*t_children[j], offset_0, y_in, x_out);
                    std::vector<CoefficientPrecision> x_j(x_out.begin() + offset_j, x_out.begin() + offset_j + t_children[j]->get_size());
                    for (int i = j + 1; i < t_children.size(); ++i) {
                        int offset_i = t_children[i]->get_offset() - offset_0;
                        std::vector<CoefficientPrecision> y_i(y_in.begin() + offset_i, y_in.begin() + offset_i + t_children[i]->get_size());
                        Ut->get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset())->add_vector_product('T', -1.0, x_j.data(), 1.0, y_i.data());
                        std::copy(y_i.begin(), y_i.end(), y_in.begin() + offset_i);
                    }
                }
            }
        }
    }

    void backward_substitution_s(const Cluster<CoordinatePrecision> &t, const int &offset_0, std::vector<CoefficientPrecision> &y_in, std::vector<CoefficientPrecision> &x_out) const {
        auto Ut = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if ((Ut->get_target_cluster().get_size() == t.get_size()) and (Ut->get_source_cluster().get_size() == t.get_size())) {
            if (Ut->is_dense()) {
                auto udense = *Ut->get_dense_data();
                for (int j = t.get_size() - 1; j > -1; --j) {
                    x_out[j + t.get_offset() - offset_0] = y_in[j + t.get_offset() - offset_0] / udense(j, j);
                    for (int i = 0; i < j; ++i) {
                        y_in[i + t.get_offset() - offset_0] = y_in[i + t.get_offset() - offset_0] - udense(i, j) * x_out[j + t.get_offset() - offset_0];
                    }
                }
            } else {
                auto &t_children = t.get_children();
                for (int j = t_children.size() - 1; j > -1; --j) {
                    int offset_j = t_children[j]->get_offset() - offset_0;
                    Ut->backward_substitution_s(*t_children[j], offset_0, y_in, x_out);
                    std::vector<CoefficientPrecision> x_j(x_out.begin() + offset_j, x_out.begin() + offset_j + t_children[j]->get_size());
                    for (int i = 0; i < j; ++i) {
                        int offset_i = t_children[i]->get_offset() - offset_0;
                        std::vector<CoefficientPrecision> y_i(y_in.begin() + offset_i, y_in.begin() + offset_i + t_children[i]->get_size());
                        Ut->get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset())->add_vector_product('N', -1.0, x_j.data(), 1.0, y_i.data());
                        std::copy(y_i.begin(), y_i.end(), y_in.begin() + offset_i);
                    }
                }
            }
        } else {
            std::cout << "noooooooooooooooooooooo" << std::endl;
            std::cout << this->get_target_cluster().get_size() << ',' << Ut->get_target_cluster().get_size() << ',' << t.get_size() << std::endl;
            std::cout << Ut->get_children().size() << std::endl;
        }
    }

    void forward_substitution(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &b) const {
        auto Lt = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (!((Lt->get_target_cluster() == t) and (Lt->get_source_cluster() == t))) {
            Matrix<CoefficientPrecision> l_dense(Lt->get_target_cluster().get_size(), Lt->get_source_cluster().get_size());
            copy_to_dense(*Lt, l_dense.data());
            Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                for (int l = 0; l < t.get_size(); ++l) {
                    ll(k, l) = l_dense(k - Lt->get_target_cluster().get_offset() + t.get_offset(), l - Lt->get_source_cluster().get_offset() + t.get_offset());
                }
            }

            for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
                y[j] = b[j];
                for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
                    b[i] = b[i] - ll(i - t.get_offset(), j - t.get_offset()) * y[j];
                }
            }
        } else {
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
                    this->forward_substitution(*tj, y, b);
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
    }

    void forward_substitution_T(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &b) const {
        auto Ut = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (!((Ut->get_target_cluster() == t) and (Ut->get_source_cluster() == t))) {
            Matrix<CoefficientPrecision> u_dense(Ut->get_target_cluster().get_size(), Ut->get_source_cluster().get_size());
            copy_to_dense(*Ut, u_dense.data());
            Matrix<CoefficientPrecision> uu(t.get_size(), t.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                for (int l = 0; l < t.get_size(); ++l) {
                    uu(k, l) = u_dense(k - Ut->get_target_cluster().get_offset() + t.get_offset(), l - Ut->get_source_cluster().get_offset() + t.get_offset());
                }
            }
            uu = uu.transp(uu);
            for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
                y[j] = b[j] / uu(j, j);
                for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
                    b[i] = b[i] - uu(i - t.get_offset(), j - t.get_offset()) * y[j];
                }
            }
        } else {
            if (Ut->get_children().size() == 0) {
                Matrix<CoefficientPrecision> M(t.get_size(), t.get_size());
                copy_to_dense(*Ut, M.data());
                // std::cout << "normfrob de m " << normFrob(M) << " bloc = " << t.get_size() << ',' << t.get_size() << std::endl;
                Matrix<double> transp(M.nb_cols(), M.nb_cols());
                transp = M.transp(M);

                for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
                    double diag = transp(j - t.get_offset(), j - t.get_offset());
                    // if (diag == 0) {
                    //     std::cout << "                      wtf                   " << std::endl;
                    // }
                    y[j] = b[j] / diag;
                    for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
                        b[i] = b[i] - transp(i - t.get_offset(), j - t.get_offset()) * y[j];
                    }
                }
            } else {
                for (auto &tj : t.get_children()) {
                    this->forward_substitution_T(*tj, y, b);
                    for (auto &ti : t.get_children()) {
                        if (ti->get_offset() > tj->get_offset()) {
                            auto Uij = this->get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
                            std::vector<CoefficientPrecision> bti(ti->get_size(), 0.0);
                            for (int k = 0; k < ti->get_size(); ++k) {
                                bti[k] = b[k + ti->get_offset()];
                            }
                            std::vector<CoefficientPrecision> ytj(tj->get_size(), 0.0);
                            for (int l = 0; l < tj->get_size(); ++l) {
                                ytj[l] = y[l + tj->get_offset()];
                            }
                            Uij->add_vector_product('T', -1.0, ytj.data(), 1.0, bti.data());

                            for (int k = 0; k < ti->get_size(); ++k) {
                                b[k + ti->get_offset()] = bti[k];
                            }
                        }
                    }
                }
            }
        }
    }
    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////

    // void forward_substitution(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &b, const int &reference_offset) const {
    //     auto Lt = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
    //     if (!((Lt->get_target_cluster() == t) and (Lt->get_source_cluster() == t))) {
    //         Matrix<CoefficientPrecision> l_dense(Lt->get_target_cluster().get_size(), Lt->get_source_cluster().get_size());
    //         copy_to_dense(*Lt, l_dense.data());
    //         Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
    //         for (int k = 0; k < t.get_size(); ++k) {
    //             for (int l = 0; l < t.get_size(); ++l) {
    //                 ll(k, l) = l_dense(k - Lt->get_target_cluster().get_offset() + t.get_offset(), l - Lt->get_source_cluster().get_offset() + t.get_offset());
    //             }
    //         }

    //         for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
    //             y[j - reference_offset] = b[j - reference_offset];
    //             for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
    //                 b[i - reference_offset] = b[i - reference_offset] - ll(i - t.get_offset(), j - t.get_offset()) * y[j - reference_offset];
    //             }
    //         }
    //     } else {
    //         if (Lt->get_children().size() == 0) {
    //             auto M = (*Lt->get_dense_data());
    //             for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
    //                 y[j - reference_offset] = b[j - reference_offset];
    //                 for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
    //                     b[i - reference_offset] = b[i - reference_offset] - M(i - t.get_offset(), j - t.get_offset()) * y[j - reference_offset];
    //                 }
    //             }
    //         } else {
    //             for (auto &tj : t.get_children()) {
    //                 this->forward_substitution(*tj, y, b, reference_offset);
    //                 for (auto &ti : t.get_children()) {
    //                     if (ti->get_offset() > tj->get_offset()) {
    //                         auto Lij = this->get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset());
    //                         std::vector<CoefficientPrecision> bti(ti->get_size(), 0.0);
    //                         for (int k = 0; k < ti->get_size(); ++k) {
    //                             bti[k] = b[k + ti->get_offset() - reference_offset];
    //                         }
    //                         std::vector<CoefficientPrecision> ytj(tj->get_size(), 0.0);
    //                         for (int l = 0; l < tj->get_size(); ++l) {
    //                             ytj[l] = y[l + tj->get_offset() - reference_offset];
    //                         }
    //                         Lij->add_vector_product('N', -1.0, ytj.data(), 1.0, bti.data());

    //                         for (int k = 0; k < ti->get_size(); ++k) {
    //                             b[k + ti->get_offset() - reference_offset] = bti[k];
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    // ///////////////////////////////
    // // FORWARD : LX= y ------------------------> OK (05/12)
    // //////////////////
    // void forward_substitution(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) const {
    //     std::cout << "t : " << t.get_size() << ',' << t.get_offset() << " y et x " << y.size() << ',' << x.size() << std::endl;
    //     auto ll = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
    //     // std::cout << "ll" << ll->get_target_cluster().get_size() << ',' << ll->get_source_cluster().get_size() << std::endl;
    //     if (ll->is_dense()) {
    //         Matrix<CoefficientPrecision> ldense(ll->get_target_cluster().get_size(), ll->get_source_cluster().get_size());
    //         copy_to_dense(*ll, ldense.data());
    //         Matrix<CoefficientPrecision> lrestr(t.get_size(), t.get_size());
    //         if ((ldense.nb_rows() == t.get_size()) && (ldense.nb_cols() == t.get_size())) {
    //             lrestr = ldense;
    //         } else {
    //             // std::cout << "forward tronqué" << std::endl;
    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 for (int l = 0; l < t.get_size(); ++l) {
    //                     lrestr(k, l) = ldense(k + t.get_offset() - ll->get_target_cluster().get_offset(), l + t.get_offset() - ll->get_source_cluster().get_offset());
    //                 }
    //             }
    //         }
    //         for (int j = 0; j < t.get_size(); ++j) {
    //             x[j + t.get_offset()] = y[j + t.get_offset()];
    //             for (int i = j + 1; i < t.get_size(); ++i) {
    //                 y[i + t.get_offset()] = y[i + t.get_offset()] - lrestr(i, j) * x[j + t.get_offset()];
    //             }
    //         }
    //     } else {
    //         auto &t_children = t.get_children();
    //         for (int j = 0; j < t_children.size(); ++j) {
    //             auto &tj = t_children[j];
    //             this->forward_substitution(*tj, y, x);
    //             for (int i = j + 1; i < t_children.size(); ++i) {
    //                 auto &ti = t_children[i];
    //                 std::vector<CoefficientPrecision> y_restr(tj->get_size());
    //                 for (int k = 0; k < y_restr.size(); ++k) {
    //                     y_restr[k] = y[k + tj->get_offset()];
    //                 }
    //                 std::vector<CoefficientPrecision> x_restr(ti->get_size());
    //                 for (int k = 0; k < x_restr.size(); ++k) {
    //                     x_restr[k] = x[k + ti->get_offset()];
    //                 }
    //                 this->get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset())->add_vector_product('N', -1.0, x_restr.data(), 1.0, y_restr.data());
    //                 for (int k = 0; k < tj->get_size(); ++k) {
    //                     y[k + tj->get_offset()] = y_restr[k];
    //                 }
    //             }
    //         }
    //     }
    // }

    ///////////////////////////////
    // BACKWARD : UX = y ----------------------------------------> OK (05/12)
    ///////////////////////////
    void backward_substitution(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) const {
        auto uu = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (uu->is_dense()) {
            Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
            copy_to_dense(*uu, udense.data());
            for (int j = t.get_size() - 1; j > -1; --j) {
                x[j + t.get_offset()] = y[j + t.get_offset()] / udense(j, j);
                for (int i = 0; i < j; ++i) {
                    y[i + t.get_offset()] = y[i + t.get_offset()] - udense(i, j) * x[j + t.get_offset()];
                }
            }
        } else {
            auto &t_children = t.get_children();
            for (int j = t_children.size() - 1; j > -1; --j) {
                auto &tj = t_children[j];
                this->backward_substitution(*tj, y, x);
                for (int i = 0; i < j; ++i) {
                    auto &ti = t_children[i];
                    std::vector<CoefficientPrecision> y_restr(tj->get_size());
                    for (int k = 0; k < y_restr.size(); ++k) {
                        y_restr[k] = y[k + tj->get_offset()];
                    }
                    std::vector<CoefficientPrecision> x_restr(ti->get_size());
                    for (int k = 0; k < x_restr.size(); ++k) {
                        x_restr[k] = x[k + ti->get_offset()];
                    }
                    this->get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset())->add_vector_product('N', -1.0, x_restr.data(), 1.0, y_restr.data());
                    for (int k = 0; k < tj->get_size(); ++k) {
                        y[k + tj->get_offset()] = y_restr[k];
                    }
                }
            }
        }
    }
    //////////////////////////////////////
    /// FORWARD T : x^T U = y^T --------------------------------------------------> OK (05/12)
    ////////////////////////////////////////
    // void forward_substitution_T(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) const {
    //     auto utt = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
    //     if (utt->is_dense()) {
    //         // Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
    //         // copy_to_dense(*utt, udense.data());
    //         Matrix<CoefficientPrecision> udense = *utt->get_dense_data();
    //         for (int j = 0; j < t.get_size(); ++j) {
    //             x[j + t.get_offset()] = y[j + t.get_offset()] / udense(j, j);
    //             for (int i = j + 1; i < t.get_size(); ++i) {
    //                 y[i + t.get_offset()] = y[i + t.get_offset()] - udense(j, i) * x[j + t.get_offset()];
    //             }
    //         }
    //     } else {
    //         auto &t_children = t.get_children();
    //         for (int j = 0; j < t_children.size(); ++j) {
    //             auto &tj = t_children[j];
    //             this->forward_substitution_T(*tj, y, x);
    //             for (int i = j + 1; i < t_children.size(); ++i) {
    //                 auto &ti = t_children[i];
    //                 std::vector<CoefficientPrecision> y_restr(tj->get_size());
    //                 for (int k = 0; k < y_restr.size(); ++k) {
    //                     y_restr[k] = y[k + tj->get_offset()];
    //                 }
    //                 std::vector<CoefficientPrecision> x_restr(ti->get_size());
    //                 for (int k = 0; k < x_restr.size(); ++k) {
    //                     x_restr[k] = x[k + ti->get_offset()];
    //                 }
    //                 this->get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset())->add_vector_product('T', -1.0, x_restr.data(), 1.0, y_restr.data());
    //                 for (int k = 0; k < tj->get_size(); ++k) {
    //                     y[k + tj->get_offset()] = y_restr[k];
    //                 }
    //             }
    //         }
    //     }
    // }
    ////////////////////////////////////////////////
    /////// L*U*y = x -----------------------------------> OK (05/12)
    //////////////////////////////:
    void solve_LU(const HMatrix &L, const HMatrix &U, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &x) const {
        std::vector<CoefficientPrecision> xtemp(L.get_target_cluster().get_size());
        L.forward_substitution_s(L.get_target_cluster(), 0, y, xtemp);
        U.backward_substitution_s(U.get_target_cluster(), 0, xtemp, x);
    }

    //////////////////////////////////////////////////////////
    /// DELTETE : YES , TEST run sans: ?
    /// DELETE STAIT : WAIT

    void forward_substitution_extract_new(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &b) const {
        auto Lt = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (!((Lt->get_target_cluster() == t) and (Lt->get_source_cluster() == t))) {
            Matrix<CoefficientPrecision> l_dense(Lt->get_target_cluster().get_size(), Lt->get_source_cluster().get_size());
            copy_to_dense(*Lt, l_dense.data());
            Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                for (int l = 0; l < t.get_size(); ++l) {
                    ll(k, l) = l_dense(k - Lt->get_target_cluster().get_offset() + t.get_offset(), l - Lt->get_source_cluster().get_offset() + t.get_offset());
                }
            }

            for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
                y[j] = b[j];
                for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
                    b[i] = b[i] - ll(i - t.get_offset(), j - t.get_offset()) * y[j];
                }
            }
        } else {
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
                    this->forward_substitution_extract_new(*tj, y, b);
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
    }
    // END DELETE WAIT
    ////////////////
    // DELETE STATE WAIT
    void forward_substitution_extract_transp(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &b) const {
        auto Lt = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Lt->get_children().size() == 0) {
            auto M = (*Lt->get_dense_data());
            for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
                y[j] = b[j] / M(j, j);
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

    // END DELETE STAIT

    /////////////////////////////////
    /// U x = y

    // Ils sont tous "faux" ils utilisent un anciens get_bloc ( nullptr si on coupe trop bas)
    // Arthur : TO DO changer le get_block -> on s'en sert jamais pour l'instant

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

    void backward_substitution_extract_new(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &x, std::vector<CoefficientPrecision> &y) const {
        auto Uk = this->extract_block(t, t);

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

    void forward_substitution_extract_T(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &b) const {
        auto Ut = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Ut == nullptr) {
            Matrix<CoefficientPrecision> u_dense(t.get_size(), t.get_size());
            if (t.get_offset() + 16 / 32 == 0) {
                std::cout << "!" << std::endl;
                auto u_big = *this->get_block(2 * t.get_size(), 2 * t.get_size(), t.get_offset() - 16, t.get_offset() - 16)->get_dense_data();
                for (int k = 0; k < t.get_size(); ++k) {
                    for (int l = 0; l < t.get_size(); ++l) {
                        u_dense(k, l) = u_big(k + 16, l + 16);
                    }
                }
            } else {
                std::cout << "!!" << std::endl;
                auto u_big = *this->get_block(2 * t.get_size(), 2 * t.get_size(), t.get_offset(), t.get_offset())->get_dense_data();
                for (int k = 0; k < t.get_size(); ++k) {
                    for (int l = 0; l < t.get_size(); ++l) {
                        u_dense(k, l) = u_big(k, l);
                    }
                }
            }
            // Matrix<CoefficientPrecision> u_dense = this->get_mat(t, t);
            std::cout << "!!!" << u_dense.nb_rows() << ',' << u_dense.nb_cols() << std::endl;
            for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
                y[j] = b[j] / u_dense(j - t.get_offset(), j - t.get_offset());
                for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
                    b[i] = b[i] - u_dense(j - t.get_offset(), i - t.get_offset()) * y[j];
                }
            }
        } else {

            if (Ut->get_children().size() == 0) {
                auto M = (*Ut->get_dense_data());
                for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
                    y[j] = b[j] / M(j - t.get_offset(), j - t.get_offset());
                    for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
                        b[i] = b[i] - M(j - t.get_offset(), i - t.get_offset()) * y[j];
                    }
                }
                // std::cout << "on était sur une feuille et maintenant ||b|| = " << norm2(b) << std::endl;
                // std::cout << " et ||y|| = " << norm2(y) << std::endl;
            } else {
                for (auto &tj : t.get_children()) {
                    this->forward_substitution_extract_T(*tj, y, b);
                    for (auto &ti : t.get_children()) {
                        if (ti->get_offset() > tj->get_offset()) {
                            auto Uij = this->get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
                            std::vector<CoefficientPrecision> btj(tj->get_size(), 0.0);
                            for (int k = 0; k < tj->get_size(); ++k) {
                                btj[k] = b[k + tj->get_offset()];
                            }
                            std::vector<CoefficientPrecision> yti(ti->get_size(), 0.0);
                            for (int l = 0; l < ti->get_size(); ++l) {
                                yti[l] = y[l + ti->get_offset()];
                            }
                            Uij->add_vector_product('T', -1.0, yti.data(), 1.0, btj.data());

                            for (int k = 0; k < tj->get_size(); ++k) {
                                b[k + tj->get_offset()] = btj[k];
                            }
                        }
                    }
                }
            }
        }
    }

    /////////////////////////////////
    //// trouver x tel que xU = y
    /// TESTS PASSED : YES
    // void forward_substitution_T(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &b, const int &reference_offset) const {
    //     auto Ut = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
    //     if (!((Ut->get_target_cluster() == t) and (Ut->get_source_cluster() == t)) or (Ut->get_children().size() == 0)) {
    //         Matrix<CoefficientPrecision> M(t.get_size(), t.get_size());
    //         if (!((Ut->get_target_cluster() == t) and (Ut->get_source_cluster() == t))) {
    //             M = Ut->extract(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
    //         } else {
    //             M = *Ut->get_dense_data();
    //         }
    //         for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
    //             y[j] = b[j] / M(j - t.get_offset(), j - t.get_offset());
    //             for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
    //                 b[i - reference_offset] = b[i - reference_offset] - M(j - t.get_offset(), i - t.get_offset()) * y[j - reference_offset];
    //             }
    //         }
    //     } else {
    //         for (auto &tj : t.get_children()) {
    //             this->forward_substitution_T(*tj, y, b, reference_offset);
    //             for (auto &ti : t.get_children()) {
    //                 if (ti->get_offset() > tj->get_offset()) {
    //                     auto Uij = this->get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
    //                     std::vector<CoefficientPrecision> btj(tj->get_size(), 0.0);
    //                     for (int k = 0; k < tj->get_size(); ++k) {
    //                         btj[k] = b[k + tj->get_offset() - reference_offset];
    //                     }
    //                     std::vector<CoefficientPrecision> yti(ti->get_size(), 0.0);
    //                     for (int l = 0; l < ti->get_size(); ++l) {
    //                         yti[l] = y[l + ti->get_offset() - reference_offset];
    //                     }

    //                     Uij->add_vector_product('T', -1.0, yti.data(), 1.0, btj.data());

    //                     for (int k = 0; k < tj->get_size(); ++k) {
    //                         b[k + tj->get_offset()] = btj[k - reference_offset];
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    //////////////////////////////////////////////////////////////////////
    void forward_substitution_extract_T_new(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &b) const {
        auto Ut = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        std::cout << "! " << std::endl;
        if (!((Ut->get_target_cluster() == t) and (Ut->get_source_cluster() == t))) {
            std::cout << "! " << std::endl;

            Matrix<CoefficientPrecision> u_dense(Ut->get_target_cluster().get_size(), Ut->get_source_cluster().get_size());
            copy_to_dense(*Ut, u_dense.data());
            Matrix<CoefficientPrecision> uu(t.get_size(), t.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                for (int l = 0; l < t.get_size(); ++l) {
                    uu(k, l) = u_dense(k - Ut->get_target_cluster().get_offset() + t.get_offset(), l - Ut->get_source_cluster().get_offset() + t.get_offset());
                }
            }

            for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
                y[j] = b[j] / uu(j - t.get_offset(), j - t.get_offset());
                for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
                    b[i] = b[i] - uu(j - t.get_offset(), i - t.get_offset()) * y[j];
                }
            }
            std::cout << "! ok " << std::endl;

        } else {
            std::cout << "!! " << std::endl;

            if (Ut->get_children().size() == 0) {
                auto M = (*Ut->get_dense_data());
                for (int j = t.get_offset(); j < t.get_offset() + t.get_size(); ++j) {
                    y[j] = b[j] / M(j - t.get_offset(), j - t.get_offset());
                    for (int i = j + 1; i < t.get_offset() + t.get_size(); ++i) {
                        b[i] = b[i] - M(j - t.get_offset(), i - t.get_offset()) * y[j];
                    }
                }
                // std::cout << "on était sur une feuille et maintenant ||b|| = " << norm2(b) << std::endl;
                // std::cout << " et ||y|| = " << norm2(y) << std::endl;
            } else {
                for (auto &tj : t.get_children()) {
                    this->forward_substitution_extract_T(*tj, y, b);
                    for (auto &ti : t.get_children()) {
                        if (ti->get_offset() > tj->get_offset()) {
                            auto Uij = this->get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
                            std::vector<CoefficientPrecision> btj(tj->get_size(), 0.0);
                            for (int k = 0; k < tj->get_size(); ++k) {
                                btj[k] = b[k + tj->get_offset()];
                            }
                            std::vector<CoefficientPrecision> yti(ti->get_size(), 0.0);
                            for (int l = 0; l < ti->get_size(); ++l) {
                                yti[l] = y[l + ti->get_offset()];
                            }
                            Uij->add_vector_product('T', -1.0, yti.data(), 1.0, btj.data());

                            for (int k = 0; k < tj->get_size(); ++k) {
                                b[k + tj->get_offset()] = btj[k];
                            }
                        }
                    }
                }
            }
            std::cout << "! ok" << std::endl;
        }
    }

    ////////////////////////
    /// BACKWARD et pas FORWARD T
    //     void backward(const Cluster<CoordinatePrecision> &t, std::vector<CoefficientPrecision> &y, std::vector<CoefficientPrecision> &b) {
    //         auto Ut = this->get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
    //         if (Ut->is_dense()) {
    //             Matrix<CoefficientPrecision> udense(t.get_size(), s.get_size());
    //             copy_to_dense(*Ut, udense.data());
    //             for (int i = t.get_size() - 1; i > -1; --j) {
    //                 CoefficientPrecision alpha = y[i + t.get_offset()];
    //                 for (int j = i + 1; j < t.get_size(); ++j) {
    //                     alpha = alpha - udense(i, j) * x[t.get_offset() + j]
    //                 }
    //                 x[i + t.get_offset()] = alpha / udense(i, i);
    //             }
    //         } else if (Ut->is_low_rank()) {
    //             auto lr = Ut->get_low_rank_data();
    //             auto uu = lr->Get_U();
    //             auto vv = lr->Get_V();
    //         }
    //     }
    // }

    //////////////////////////////////
    //// trouver x tq LU x = y
    //// obsolete : get_block
    //// Arthur TO DO -> extract =extract_new
    /// TEST PASSED : NO

    friend void
    forward_backward(const HMatrix &L, const HMatrix &U, const Cluster<CoordinatePrecision> &root, std::vector<CoefficientPrecision> &x, std::vector<CoordinatePrecision> &y) {
        std::vector<CoefficientPrecision> ux(L.get_source_cluster().get_size(), 0.0);
        L.forward_substitution_extract(root, ux, y);
        U.backward_substitution_extract(root, x, ux);
    }

    /////////////////////////////////////
    ///// A remplacer par un constructeur vide qui prend bloc_cluster_tree d'une hmatrice
    void copy_zero(const HMatrix &H) {
        if (H.get_children().size() > 0) {
            for (auto &child : H.get_children()) {
                auto bloc = this->add_child(&child->get_target_cluster(), &child->get_source_cluster());
                bloc->copy_zero(*child);
            }
        } else {
            Matrix<CoefficientPrecision> zero(H.get_target_cluster().get_size(), H.get_source_cluster().get_size());
            this->set_dense_data(zero);
            bool temp = true;
        }
    }

    /////////////////////////
    //// DELETE STAIT : WAIT

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
    //////////////////////////////////////////////
    /// DELETE STAIT : WAIT
    // friend void forward_substitution_dense(const HMatrix &L, HMatrix &Z, Matrix<CoefficientPrecision> z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, Matrix<CoefficientPrecision> &X, HMatrix &XX) {
    //     auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());

    //     if (Zts->is_dense()) {
    //         Matrix<CoefficientPrecision> xdense(X.nb_rows(), X.nb_cols());
    //         copy_to_dense(XX, xdense.data());
    //         auto Zdense = *Zts->get_dense_data();
    //         Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
    //         auto ltemp = L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
    //         copy_to_dense(*ltemp, ll.data());
    //         Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
    //         std::cout << "avant la boucle : " << normFrob(X - xdense) / normFrob(X) << std::endl;

    //         for (int j = 0; j < s.get_size(); ++j) {
    //             std::vector<CoefficientPrecision> Xtj(L.get_source_cluster().get_size(), 0.0);
    //             std::vector<CoefficientPrecision> Ztj(Z.get_target_cluster().get_size(), 0.0);
    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 Ztj[k + t.get_offset()] = z(k + t.get_offset(), j + s.get_offset());
    //                 // Ztj[k + t.get_offset()] = Zdense(k, j);
    //             }
    //             L.forward_substitution_extract(t, Xtj, Ztj);
    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 X(k + t.get_offset(), j + s.get_offset()) = Xtj[k + t.get_offset()];
    //                 Xupdate(k, j)                             = Xtj[k + t.get_offset()];
    //             }
    //         }
    //         std::cout << "aprés la boucle : " << normFrob(X - xdense) / normFrob(X) << std::endl;
    //         MatrixGenerator<CoefficientPrecision> gen(Xupdate, t.get_offset(), s.get_offset());
    //         std::cout << "avant compute : " << normFrob(xdense) << std::endl;
    //         // XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->compute_dense_data(gen);
    //         XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xupdate);
    //         std::cout << XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_children().size() << std::endl;
    //         // XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->delete_children();
    //         // std::cout << XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_children().size() << std::endl;
    //         // std::cout << XX.get_children().size() << std::endl;
    //         // copy_to_dense(XX, xdense.data());
    //         //  std::cout << "aprés compute : " << normFrob(xdense) << std::endl;
    //         //  std::cout << "alors qu'on a rajourté une feuille de norme " << normFrob(*XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data()) << std::endl;

    //         // /////test/////
    //         // std::cout << " erreur sur le blocs rajouté :" << normFrob((*XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data()) - Xupdate) / normFrob(Xupdate) << std::endl;
    //         // std::cout << "norme de la feuille " << normFrob((*XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data())) << ',' << normFrob(Xupdate) << std::endl;
    //         // auto Xtemp = XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //         // std::cout << "  get block ? " << Xtemp->get_target_cluster().get_size() << ',' << Xtemp->get_source_cluster().get_size() << ',' << Xtemp->get_target_cluster().get_offset() << ',' << Xtemp->get_source_cluster().get_offset() << std::endl;
    //         // std::cout << "   alors que " << t.get_size() << ',' << s.get_size() << ',' << t.get_offset() << ',' << s.get_offset() << std::endl;
    //         // Matrix<CoefficientPrecision> xxx(X.nb_rows(), X.nb_cols());
    //         // copy_to_dense(XX, xxx.data());
    //         // std::cout << "    erreur sur X : " << normFrob(X - xxx) / normFrob(X) << std::endl;
    //         // std::cout << "     wtf : " << normFrob(xxx) << ',' << normFrob(X) << std::endl;
    //         // // std::cout << "compute dense avant - aprés : " << normFrob(xxx - xdense) << "alors que apré =" << normFrob(xxx) << " , avant = " << normFrob(xdense) << " et on a rajouté " << normFrob((*XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data()) - Xupdate) << std::endl;
    //         // std::cout << "norme de la feuille " << normFrob(*XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data()) << std::endl;
    //     } else {
    //         for (int i = 0; i < t.get_children().size(); ++i) {
    //             auto &child_t = t.get_children()[i];
    //             for (auto &child_s : s.get_children()) {
    //                 // std::cout << "avant forward " << std::endl;
    //                 // Matrix<CoefficientPrecision> t1(X.nb_rows(), X.nb_cols());
    //                 // Matrix<CoefficientPrecision> t2(z.nb_rows(), z.nb_cols());
    //                 // copy_to_dense(XX, t1.data());
    //                 // copy_to_dense(Z, t2.data());
    //                 // std::cout << " X : " << normFrob(t1 - X) / normFrob(X) << " , Z : " << normFrob(z - t2) / normFrob(z) << std::endl;
    //                 forward_substitution_dense(L, Z, z, *child_t, *child_s, X, XX);
    //                 // std::cout << "aprés forward " << std::endl;
    //                 // Matrix<CoefficientPrecision> t11(X.nb_rows(), X.nb_cols());
    //                 // Matrix<CoefficientPrecision> t22(z.nb_rows(), z.nb_cols());
    //                 // copy_to_dense(XX, t11.data());
    //                 // copy_to_dense(Z, t22.data());
    //                 // std::cout << " X : " << normFrob(t11 - X) / normFrob(X) << " , Z : " << normFrob(z - t22) / normFrob(z) << std::endl;
    //                 for (int j = i + 1; j < t.get_children().size(); ++j) {
    //                     auto &child_tt = t.get_children()[j];
    //                     auto Ztemp     = Z.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
    //                     auto Ltemp     = L.get_block(child_t->get_size(), child_tt->get_size(), child_t->get_offset(), child_tt->get_offset());
    //                     auto Xtemp     = XX.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
    //                     Matrix<CoefficientPrecision> ll(Ltemp->get_target_cluster().get_size(), Ltemp->get_source_cluster().get_size());
    //                     copy_to_dense(*Ltemp, ll.data());
    //                     Matrix<CoefficientPrecision> xx(child_tt->get_size(), child_s->get_size());
    //                     for (int k = 0; k < child_tt->get_size(); ++k) {
    //                         for (int l = 0; l < child_s->get_size(); ++l) {
    //                             xx(k, l) = X(k + child_tt->get_offset(), l + child_s->get_offset());
    //                         }
    //                     }
    //                     auto M = ll * xx;
    //                     for (int k = 0; k < child_t->get_size(); ++k) {
    //                         for (int l = 0; l < child_s->get_size(); ++l) {
    //                             z(k + child_t->get_offset(), l + child_s->get_offset()) = z(k + child_t->get_offset(), l + child_s->get_offset()) - M(k, l);
    //                         }
    //                     }
    //                     // auto l = Ltemp->hmatrix_product(*Xtemp);
    //                     // Matrix<CoefficientPrecision> ltest(child_t->get_size(), child_s->get_size());
    //                     // copy_to_dense(l, ltest.data());
    //                     // std::cout << "erreur L* x :" << normFrob(ltest - M) << normFrob(M) << std::endl;
    //                     Ztemp->Moins(Ltemp->hmatrix_product(*Xtemp));
    //                     // // tests//
    //                     Matrix<CoefficientPrecision> ztest(Z.get_target_cluster().get_size(), Z.get_source_cluster().get_size());
    //                     copy_to_dense(Z, ztest.data());
    //                     std::cout << "erreur sur z : " << normFrob(z - ztest) / normFrob(z) << std::endl;
    //                 }
    //             }
    //         }
    //     }
    // }
    ///////////////////////////////////////////////////////
    /////// FORWARD MATRIX -----------------------------> OK (11/12)
    ///////////////////////////////////////////////////
    friend void Forward_Matrix_s(const HMatrix &L, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &Z, HMatrix &X) {
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        if ((Zts->get_target_cluster().get_size() == t.get_size()) and (Zts->get_target_cluster().get_size())) {
            if (Zts->is_dense()) {
                auto zdense = Zts->get_dense_data();
                for (int j = 0; j < s.get_size(); ++j) {
                    std::vector<CoefficientPrecision> col_j(t.get_size(), 0.0);
                }
            }
        }
    }
    friend void Forward_Matrix(const HMatrix &L, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &Z, HMatrix &X) {
        // std::cout << "APPEL FORWARD_M : trouver X tq LX = Z sur t=  (" << t.get_size() << "," << t.get_offset() << "),(" << s.get_size() << ',' << s.get_offset() << std::endl;
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        // Matrix<CoefficientPrecision> zz(t.get_size(), s.get_size());
        // copy_to_dense(*Zts, zz.data());
        // std::cout << "forward sur Z de norme " << normFrob(zz) << std::endl;
        // std::cout << Zts->is_dense() << ',' << Zts->is_low_rank() << ',' << Zts->get_children().size() << std::endl;

        // std::cout << " forward -> pour trouver une matrice trieagulaire supérieursu le bloc t,s = " << '(' << t.get_size() << ',' << t.get_offset() << "),(" << s.get_size() << ',' << s.get_offset() << ')' << " de n,orme : " << normFrob(zz) << std::endl;
        /// on regarde si L est pas dense sinon ca sert a rien de descendre :
        auto Ltt = L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Ltt->is_dense()) {
            Matrix<CoefficientPrecision> Zdense(t.get_size(), s.get_size());
            // std::cout << "Zdense ? " << Zts->get_target_cluster().get_size() << ',' << Zts->get_source_cluster().get_size() << "= " << t.get_size() << ',' << s.get_size() << std::endl;
            copy_to_dense(*Zts, Zdense.data());
            auto ZZ = Zdense;
            // std::cout << "normFrob Zdense : " << normFrob(Zdense) << std::endl;
            Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
            for (int j = 0; j < s.get_size(); ++j) {
                std::vector<CoefficientPrecision> Xtj(L.get_source_cluster().get_size(), 0.0);
                std::vector<CoefficientPrecision> Ztj(L.get_target_cluster().get_size(), 0.0);
                for (int k = 0; k < t.get_size(); ++k) {
                    Ztj[k + t.get_offset()] = Zdense(k, j);
                }
                L.forward_substitution(t, Xtj, Ztj);
                for (int k = 0; k < t.get_size(); ++k) {
                    Xupdate(k, j) = Xtj[k + t.get_offset()];
                }
            }
            Matrix<CoefficientPrecision> Ldense(t.get_size(), t.get_size());
            copy_to_dense(*Ltt, Ldense.data());
            // std::cout << "erreur sur le bloc calculé Z-LX  =:" << normFrob(Ldense * Xupdate - ZZ) / normFrob(ZZ) << std::endl;
            X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xupdate);
            X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->delete_children();
        } else {
            if (Zts->get_children().size() > 0) {
                auto &t_children = t.get_children();
                for (int i = 0; i < t_children.size(); ++i) {
                    auto &ti = t_children[i];
                    if (L.get_block(ti->get_size(), ti->get_size(), ti->get_offset(), ti->get_offset()) == nullptr) {
                        Matrix<CoefficientPrecision> Zdense(t.get_size(), s.get_size());
                        copy_to_dense(*Zts, Zdense.data());
                        Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
                        for (int j = 0; j < s.get_size(); ++j) {
                            std::vector<CoefficientPrecision> Xtj(L.get_source_cluster().get_size(), 0.0);
                            std::vector<CoefficientPrecision> Ztj(L.get_target_cluster().get_size(), 0.0);
                            for (int k = 0; k < t.get_size(); ++k) {
                                Ztj[k + t.get_offset() - L.get_target_cluster().get_offset()] = Zdense(k, j);
                            }
                            L.forward_substitution(t, Xtj, Ztj);
                            for (int k = 0; k < t.get_size(); ++k) {
                                Xupdate(k, j) = Xtj[k + t.get_offset() - L.get_source_cluster().get_offset()];
                            }
                        }
                        X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xupdate);
                    } else {

                        auto &s_children = s.get_children();
                        for (auto &s_child : s_children) {

                            Forward_Matrix(L, *ti, *s_child, Z, X);
                            for (int j = i + 1; j < t_children.size(); ++j) {
                                auto &tj  = t_children[j];
                                auto Ztjs = Z.get_block(tj->get_size(), s_child->get_size(), tj->get_offset(), s_child->get_offset());
                                auto Ltji = L.get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
                                auto Xis  = X.get_block(ti->get_size(), s_child->get_size(), ti->get_offset(), s_child->get_offset());
                                if ((Ztjs->get_target_cluster().get_size() == tj->get_size()) && (Ztjs->get_source_cluster().get_size() == s_child->get_size()) && (Ltji->get_target_cluster().get_size() == tj->get_size()) && (Ltji->get_source_cluster().get_size() == ti->get_size()) && (Xis->get_target_cluster().get_size() == ti->get_size()) && (Xis->get_source_cluster().get_size() == s_child->get_size())) {
                                    // std::cout << "yoooooooooooooo" << std::endl;
                                    Ztjs->Moins(Ltji->hmatrix_product(*Xis));
                                    // Ztjs->Moins(Ltji->hmatrix_product(*Xis), *t_children[j], *s_child);

                                } else {
                                    std::cout << "yiiiiiiiiiiiiiiiiiiiii" << std::endl;
                                    Matrix<CoefficientPrecision> Zdense(Ztjs->get_target_cluster().get_size(), Ztjs->get_source_cluster().get_size());
                                    Matrix<CoefficientPrecision> Xdense(Xis->get_target_cluster().get_size(), Xis->get_source_cluster().get_size());
                                    Matrix<CoefficientPrecision> Ldense(Ltji->get_target_cluster().get_size(), Ltji->get_source_cluster().get_size());
                                    Matrix<CoefficientPrecision> Zrestr(tj->get_size(), s_child->get_size());
                                    Matrix<CoefficientPrecision> Xrestr(ti->get_size(), s_child->get_size());
                                    Matrix<CoefficientPrecision> Lrestr(tj->get_size(), ti->get_size());
                                    copy_to_dense(*Ztjs, Zdense.data());
                                    if (Ztjs->is_dense()) {
                                        Zdense = *Ztjs->get_dense_data();
                                    } else if (Ztjs->is_low_rank()) {
                                        Zdense = Ztjs->get_low_rank_data()->Get_U() * Ztjs->get_low_rank_data()->Get_V();
                                    }
                                    copy_to_dense(*Xis, Xdense.data());
                                    copy_to_dense(*Ltji, Ldense.data());
                                    std::cout << Zdense.nb_rows() << ',' << Zdense.nb_cols() << '|' << tj->get_size() << ',' << tj->get_offset() << '+' << s_child->get_size() << ',' << s_child->get_offset() << std::endl;
                                    std::cout << Xdense.nb_rows() << ',' << Xdense.nb_cols() << '|' << ti->get_size() << ',' << s_child->get_size() << std::endl;
                                    std::cout << Ldense.nb_rows() << ',' << Ldense.nb_cols() << '|' << tj->get_size() << ',' << ti->get_size() << std::endl;
                                    std::cout << normFrob(Zdense) << ',' << normFrob(Xdense) << ',' << normFrob(Ldense) << std::endl;

                                    for (int k = 0; k < tj->get_size(); ++k) {
                                        for (int l = 0; l < s_child->get_size(); ++l) {
                                            Zrestr(k, l) = Zdense(k + tj->get_offset() - Ztjs->get_target_cluster().get_offset(), l + s_child->get_offset() - Ztjs->get_source_cluster().get_offset());
                                        }
                                        for (int l = 0; l < ti->get_size(); ++l) {
                                            Lrestr(k, l) = Ldense(k + tj->get_offset() - Ltji->get_target_cluster().get_offset(), l + ti->get_offset() - Ltji->get_source_cluster().get_offset());
                                        }
                                    }
                                    for (int k = 0; k < ti->get_size(); ++k) {
                                        for (int l = 0; l < s_child->get_size(); ++l) {
                                            Xrestr(k, l) = Xdense(k + ti->get_offset() - Xis->get_target_cluster().get_offset(), l + s_child->get_offset() - Xis->get_source_cluster().get_size());
                                        }
                                    }
                                    std::cout << normFrob(Zrestr) << ',' << normFrob(Xrestr) << ',' << normFrob(Lrestr) << std::endl;
                                    Zrestr = Zrestr - Lrestr * Xrestr;
                                    Ztjs->set_dense_data(Zrestr);
                                }
                            }
                        }
                    }
                }
            } else {
                if (Zts->is_dense()) {
                    // Matrix<CoefficientPrecision> Ldense(t.get_size(), t.get_size());
                    // copy_to_dense(*L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), Ldense.data());
                    auto Zdense = *Zts->get_dense_data();
                    // auto ZZ     = Zdense;
                    // std::cout << "ldense zdense : " << normFrob(Ldense) << ',' << normFrob(ZZ) << std::endl;
                    Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
                    for (int j = 0; j < s.get_size(); ++j) {
                        std::vector<CoefficientPrecision> Xtj(L.get_source_cluster().get_size(), 0.0);
                        std::vector<CoefficientPrecision> Ztj(L.get_target_cluster().get_size(), 0.0);
                        for (int k = 0; k < t.get_size(); ++k) {
                            Ztj[k + t.get_offset() - L.get_target_cluster().get_offset()] = Zdense(k, j);
                        }
                        L.forward_substitution(t, Xtj, Ztj);
                        for (int k = 0; k < t.get_size(); ++k) {
                            Xupdate(k, j) = Xtj[k + t.get_offset() - L.get_source_cluster().get_offset()];
                        }
                    }
                    // std::cout << "erreur sur le bloc calculé Z-LX  =:" << normFrob(Ldense * Xupdate - ZZ) / normFrob(ZZ) << std::endl;
                    X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xupdate);
                    X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->delete_children();
                    // copy_to_dense(*X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset()), Xupdate.data());
                    // std::cout << "erreur sur le bloc push Z-LX  =:" << normFrob(Ldense * Xupdate - ZZ) / normFrob(ZZ) << std::endl;

                } else {
                    // std::cout << "!!!!!!" << std::endl;
                    auto U = Zts->get_low_rank_data()->Get_U();
                    auto V = Zts->get_low_rank_data()->Get_V();
                    Matrix<CoefficientPrecision> Xu(t.get_size(), Zts->get_low_rank_data()->rank_of());
                    for (int j = 0; j < Zts->get_low_rank_data()->rank_of(); ++j) {
                        std::vector<CoefficientPrecision> Uj(L.get_target_cluster().get_size());
                        std::vector<CoefficientPrecision> Aj(L.get_target_cluster().get_size());
                        for (int k = 0; k < t.get_size(); ++k) {
                            Uj[k + t.get_offset() - L.get_target_cluster().get_offset()] = U(k, j);
                        }
                        L.forward_substitution(t, Aj, Uj);
                        for (int k = 0; k < t.get_size(); ++k) {
                            Xu(k, j) = Aj[k + t.get_offset() - L.get_target_cluster().get_offset()];
                        }
                    }
                    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Xu, V);
                    X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_low_rank_data(Xlr);
                }
            }
        }
    }

    // On calcule Zts = LttXts
    friend void FM(const HMatrix &L, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &Z, HMatrix &X) {
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        auto Ltt = L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Zts->get_children().size() == 0) {
            if (Zts->is_dense()) {
                Matrix<CoefficientPrecision> Xts(t.get_size(), s.get_size());
                auto Zdense = *Zts->get_dense_data();
                for (int j = 0; j < Zdense.nb_cols(); ++j) {
                    // on extrait les colonnes
                    std::vector<CoefficientPrecision> col_Z_j(Zdense.nb_rows());
                    for (int i = 0; i < Zdense.nb_rows(); ++i) {
                        col_Z_j[i] = Zdense(i, j);
                    }
                    std::vector<CoefficientPrecision> col_X_j(t.get_size());
                    L.forward_substitution_s(t, t.get_offset(), col_Z_j, col_X_j);
                    // On recopiie colonne par colonne dans X
                    for (int i = 0; i < t.get_size(); ++i) {
                        Xts(i, j) = col_X_j[i];
                    }
                }
                X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xts);
                // pour être sure qu'il yait pas des trucs bizzard
                X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->delete_children();
                // X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->split(Xts);
            } else {
                /// UV = LX -> U = LXu , V=Xv
                auto U = Zts->get_low_rank_data()->Get_U();
                auto V = Zts->get_low_rank_data()->Get_V();
                Matrix<CoefficientPrecision> Xu(t.get_size(), U.nb_cols());
                for (int j = 0; j < U.nb_cols(); ++j) {
                    // on extrait la colonne de U
                    std::vector<CoefficientPrecision> col_U_j(U.nb_rows());
                    for (int i = 0; i < t.get_size(); ++i) {
                        col_U_j[i] = U(i, j);
                    }
                    std::vector<CoefficientPrecision> col_XU_j(t.get_size());
                    L.forward_substitution_s(t, t.get_offset(), col_U_j, col_XU_j);
                    // On recopiie colonne par colonne dans X
                    for (int i = 0; i < t.get_size(); ++i) {
                        Xu(i, j) = col_XU_j[i];
                    }
                }
                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lrX(Xu, V);
                X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_low_rank_data(lrX);
                // pour être sure qu'il yait pas des trucs bizzard
                X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->delete_children();
            }
        } else if ((Ltt->get_children().size() == 0) and (Zts->get_children().size() > 0)) {
            // mauvais cas on doit densifier Z
            Matrix<CoefficientPrecision> Xts(t.get_size(), s.get_size());
            Matrix<CoefficientPrecision> Zdense(t.get_size(), s.get_size());
            copy_to_dense(*Zts, Zdense.data());
            for (int j = 0; j < Zdense.nb_cols(); ++j) {
                // on extrait les colonnes
                std::vector<CoefficientPrecision> col_Z_j(Zdense.nb_rows());
                for (int i = 0; i < Zdense.nb_rows(); ++i) {
                    col_Z_j[i] = Zdense(i, j);
                }
                std::vector<CoefficientPrecision> col_X_j(t.get_size());
                L.forward_substitution_s(t, t.get_offset(), col_Z_j, col_X_j);
                // On recopiie colonne par colonne dans X
                for (int i = 0; i < t.get_size(); ++i) {
                    Xts(i, j) = col_X_j[i];
                }
            }
            X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xts);
            // pour être sure qu'il yait pas des trucs bizzard
            X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->delete_children();
        } else if ((Ltt->get_children().size() > 0) and (Zts->get_children().size() > 0)) {
            auto &t_children = t.get_children();
            auto &s_children = s.get_children();
            // on regarde si on peut descendre
            for (int i = 0; i < t_children.size(); ++i) {
                for (auto &s_child : s_children) {
                    // auto x_child = X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->add_child(*t_children[i], *s_child);
                    FM(L, *t_children[i], *s_child, Z, X);
                    for (int j = i + 1; j < s_children.size(); ++j) {
                        auto Zjs = Z.get_block(t_children[j]->get_size(), s_child->get_size(), t_children[j]->get_offset(), s_child->get_offset());
                        auto Lji = L.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset());
                        auto Xis = X.get_block(t_children[i]->get_size(), s_child->get_size(), t_children[i]->get_offset(), s_child->get_offset());

                        // on verifie que les blocs existent vraimet
                        if ((Lji->get_target_cluster().get_size() == t_children[j]->get_size()) and (Lji->get_source_cluster().get_size() == t_children[i]->get_size())) {
                            Zjs->Moins(Lji->hmatrix_product(*Xis));
                            // Zjs->Moins(Lji->hmatrix_product_new(*Xis));

                        } else {
                            Matrix<CoefficientPrecision> ldense(Lji->get_target_cluster().get_size(), Lji->get_source_cluster().get_size());
                            Matrix<CoefficientPrecision> lrestr(t_children[j]->get_size(), t_children[i]->get_size());
                            for (int k = 0; k < t_children[j]->get_size(); ++k) {
                                for (int l = 0; l < t_children[i]->get_size(); ++l) {
                                    lrestr(k, l) = ldense(k + t_children[j]->get_offset() - Lji->get_target_cluster().get_offset(), l + t_children[i]->get_offset() - Lji->get_source_cluster().get_offset());
                                }
                            }
                            Matrix<CoefficientPrecision> xdense(t_children[i]->get_size(), s_child->get_size());
                            auto zdense = *Zjs->get_dense_data();
                            copy_to_dense(*Xis, xdense.data());
                            zdense = zdense - lrestr * xdense;
                            Zjs->set_dense_data(zdense);
                            // Zjs->split(zdense);
                        }
                    }
                }
            }
        }
    }
    ////////////////////////////////////////////////////////////
    /////// Backward matrix : Find Xts such that UttXts= Yts //////////////
    ///////////////////////////////////////////////////////////
    ///// ca sert a rien c'est juste pour les tests

    // friend void BM(const HMatrix &U, const cluster<CoordinatePrecision> &t, const cluster<CoordinatePrecision> &s, HMatrix &Y, HMatrix &X) {
    //     auto Yts = Y.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //     if (Yts->is_dense()) {
    //         for (int k = 0; k < s.get_size(); ++k) {
    //             std::vector<CoefficientPrecision> col_k(U.get_target_cluster().get_size(), U.get_target_cluster().get_size());
    //             auto data = Yts->get_dense_data()->get_col(k).data().begin();
    //             std::copy(data.begin(), data.end(), col_k.begin() + t.get_offset());
    //             std::vector<CoefficientPrecision> res(t.get_size());
    //             auto U.backward_substitution(t, col_k, res) ;

    //         }
    //     }
    // }

    friend void FM_build(const HMatrix &L, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &Z, HMatrix &X) {
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        auto Ltt = L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Zts->get_children().size() == 0) {
            if (Zts->is_dense()) {
                Matrix<CoefficientPrecision> Xts(t.get_size(), s.get_size());
                auto Zdense = *Zts->get_dense_data();
                for (int j = 0; j < Zdense.nb_cols(); ++j) {
                    // on extrait les colonnes
                    std::vector<CoefficientPrecision> col_Z_j(Zdense.nb_rows());
                    for (int i = 0; i < Zdense.nb_rows(); ++i) {
                        col_Z_j[i] = Zdense(i, j);
                    }
                    std::vector<CoefficientPrecision> col_X_j(t.get_size());
                    L.forward_substitution_s(t, t.get_offset(), col_Z_j, col_X_j);
                    // On recopiie colonne par colonne dans X
                    for (int i = 0; i < t.get_size(); ++i) {
                        Xts(i, j) = col_X_j[i];
                    }
                }
                X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xts);
                // pour être sure qu'il yait pas des trucs bizzard
                X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->delete_children();
            } else {
                /// UV = LX -> U = LXu , V=Xv
                auto U = Zts->get_low_rank_data()->Get_U();
                auto V = Zts->get_low_rank_data()->Get_V();
                Matrix<CoefficientPrecision> Xu(t.get_size(), U.nb_cols());
                for (int j = 0; j < U.nb_cols(); ++j) {
                    // on extrait la colonne de U
                    std::vector<CoefficientPrecision> col_U_j(U.nb_rows());
                    for (int i = 0; i < t.get_size(); ++i) {
                        col_U_j[i] = U(i, j);
                    }
                    std::vector<CoefficientPrecision> col_XU_j(t.get_size());
                    L.forward_substitution_s(t, t.get_offset(), col_U_j, col_XU_j);
                    // On recopiie colonne par colonne dans X
                    for (int i = 0; i < t.get_size(); ++i) {
                        Xu(i, j) = col_XU_j[i];
                    }
                }
                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lrX(Xu, V);
                X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_low_rank_data(lrX);
                // pour être sure qu'il yait pas des trucs bizzard
                X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->delete_children();
            }
        } else {
            auto &t_children = t.get_children();
            auto &s_children = s.get_children();
            // on regarde si on peut descendre
            for (int i = 0; i < t_children.size(); ++i) {
                for (auto &s_child : s_children) {
                    auto x_child = X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->add_child(t_children[i].get(), s_child.get());
                    FM_build(L, *t_children[i], *s_child, Z, *x_child);
                    X.assign(*x_child, *t_children[i], *s_child);
                    for (int j = i + 1; j < s_children.size(); ++j) {
                        auto Zjs = Z.get_block(t_children[j]->get_size(), s_child->get_size(), t_children[j]->get_offset(), s_child->get_offset());
                        auto Lji = L.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset());
                        auto Xis = X.get_block(t_children[i]->get_size(), s_child->get_size(), t_children[i]->get_offset(), s_child->get_offset());
                        Zjs->Moins(Lji->hmatrix_product_new(*x_child));
                    }
                }
            }
        }
    }

    friend void FM_T(const HMatrix &U, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &Z, HMatrix &X) {
        auto Zst = Z.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset());
        auto Utt = U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Zst->get_children().size() == 0) {
            if (Zst->is_dense()) {
                Matrix<CoefficientPrecision> Zdense(s.get_size(), t.get_size());
                copy_to_dense(*Zst, Zdense.data());
                Matrix<CoefficientPrecision> Xupdate(s.get_size(), t.get_size());
                for (int j = 0; j < s.get_size(); ++j) {
                    std::vector<CoefficientPrecision> Xjs(t.get_size(), 0.0);
                    std::vector<CoefficientPrecision> Zjs(t.get_size(), 0.0);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Zjs[k] = Zdense(j, k);
                    }
                    U.forward_substitution_T_s(t, t.get_offset(), Zjs, Xjs);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Xupdate(j, k) = Xjs[k];
                    }
                }
                X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->set_dense_data(Xupdate);
                X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->delete_children();
                // X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->split(Xupdate);

            } else {
                Matrix<double> zz(s.get_size(), t.get_size());
                copy_to_dense(*Zst, zz.data());
                auto Ur = Zst->get_low_rank_data()->Get_U();
                auto Vr = Zst->get_low_rank_data()->Get_V();
                Matrix<CoefficientPrecision> Xv(Ur.nb_cols(), t.get_size());

                for (int j = 0; j < Zst->get_low_rank_data()->rank_of(); ++j) {
                    std::vector<CoefficientPrecision> Vj(t.get_size());
                    std::vector<CoefficientPrecision> Aj(t.get_size());
                    for (int k = 0; k < t.get_size(); ++k) {
                        Vj[k] = Vr(j, k);
                    }

                    U.forward_substitution_T_s(t, t.get_offset(), Vj, Aj);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Xv(j, k) = Aj[k];
                    }
                }
                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Ur, Xv);
                X.set_epsilon(Z.m_tree_data->m_epsilon);
                X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->set_low_rank_data(Xlr);
                X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->delete_children();
            }
        } else if ((Utt->get_children().size() == 0) and (Zst->get_children().size() > 0)) {
            Matrix<CoefficientPrecision> Zdense(s.get_size(), t.get_size());
            copy_to_dense(*Zst, Zdense.data());
            Matrix<CoefficientPrecision> Xupdate(s.get_size(), t.get_size());
            for (int j = 0; j < s.get_size(); ++j) {
                std::vector<CoefficientPrecision> Xjs(t.get_size(), 0.0);
                std::vector<CoefficientPrecision> Zjs(t.get_size(), 0.0);
                for (int k = 0; k < t.get_size(); ++k) {
                    Zjs[k] = Zdense(j, k);
                }
                U.forward_substitution_T_s(t, t.get_offset(), Zjs, Xjs);
                for (int k = 0; k < t.get_size(); ++k) {
                    Xupdate(j, k) = Xjs[k];
                }
            }
            X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->set_dense_data(Xupdate);
            X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->delete_children();
        } else if ((Utt->get_children().size() > 0) and (Zst->get_children().size() > 0)) {
            auto &t_children = t.get_children();
            auto &s_children = s.get_children();
            // on regarde si on peut descendre
            for (int i = 0; i < t_children.size(); ++i) {
                for (auto &s_child : s_children) {
                    // auto x_child = X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->add_child(t_children[i].get(), s_child.get());
                    FM_T(U, *t_children[i], *s_child, Z, X);
                    for (int j = i + 1; j < s_children.size(); ++j) {
                        auto Zsj = Z.get_block(s_child->get_size(), t_children[j]->get_size(), s_child->get_offset(), t_children[j]->get_offset());
                        auto Uij = U.get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset());
                        auto Xsi = X.get_block(s_child->get_size(), t_children[i]->get_size(), s_child->get_offset(), t_children[i]->get_offset());
                        Xsi->set_admissibility_condition(Zsj->get_admissibility_condition());
                        Xsi->set_low_rank_generator(Zsj->get_low_rank_generator());
                        // std::cout << Xsi->get_target_cluster().get_size() << ',' << Xsi->get_source_cluster().get_size() << "    " << Uij->get_target_cluster().get_size() << ',' << Uij->get_source_cluster().get_size() << std::endl;
                        // std::cout << Uij->is_dense() << ',' << Uij->is_low_rank() << std::endl;
                        if ((Uij->get_target_cluster().get_size() != t_children[i]->get_size()) or (Uij->get_source_cluster().get_size() != t_children[j]->get_size())) {
                            Matrix<double> udense(Uij->get_target_cluster().get_size(), Uij->get_source_cluster().get_size());
                            copy_to_dense(*Uij, udense.data());
                            Matrix<double> urestr(t_children[i]->get_size(), t_children[j]->get_size());
                            for (int k = 0; k < t_children[i]->get_size(); ++k) {
                                for (int l = 0; l < t_children[j]->get_size(); ++l) {
                                    urestr(k, l) = udense(k + t_children[i]->get_offset() - Uij->get_target_cluster().get_offset(), l + t_children[j]->get_offset() - Uij->get_source_cluster().get_offset());
                                }
                            }
                            Matrix<double> zdense(s_child->get_size(), t_children[j]->get_size());
                            Matrix<double> xdense(s_child->get_size(), t_children[i]->get_size());
                            copy_to_dense(*Zsj, zdense.data());
                            copy_to_dense(*Xsi, xdense.data());
                            zdense = zdense - urestr * xdense;
                            Zsj->set_dense_data(zdense);
                            // Zsj->split(zdense);
                        } else {
                            Zsj->Moins(Xsi->hmatrix_product(*Uij));
                        }
                    }
                }
            }
        }
    }

    friend void FM_T_build(const HMatrix &U, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &Z, HMatrix &X) {
        auto Zst = Z.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset());
        auto Utt = U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Zst->get_children().size() == 0) {
            if (Zst->is_dense()) {
                Matrix<CoefficientPrecision> Zdense(s.get_size(), t.get_size());
                copy_to_dense(*Zst, Zdense.data());
                Matrix<CoefficientPrecision> Xupdate(s.get_size(), t.get_size());
                for (int j = 0; j < s.get_size(); ++j) {
                    std::vector<CoefficientPrecision> Xjs(t.get_size(), 0.0);
                    std::vector<CoefficientPrecision> Zjs(t.get_size(), 0.0);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Zjs[k] = Zdense(j, k);
                    }
                    U.forward_substitution_T_s(t, t.get_offset(), Zjs, Xjs);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Xupdate(j, k) = Xjs[k];
                    }
                }
                X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->set_dense_data(Xupdate);
                X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->delete_children();

            } else {
                Matrix<double> zz(s.get_size(), t.get_size());
                copy_to_dense(*Zst, zz.data());
                auto Ur = Zst->get_low_rank_data()->Get_U();
                auto Vr = Zst->get_low_rank_data()->Get_V();
                Matrix<CoefficientPrecision> Xv(Ur.nb_cols(), s.get_size());

                for (int j = 0; j < Zst->get_low_rank_data()->rank_of(); ++j) {
                    std::vector<CoefficientPrecision> Vj(s.get_size());
                    std::vector<CoefficientPrecision> Aj(s.get_size());
                    for (int k = 0; k < t.get_size(); ++k) {
                        Vj[k] = Vr(j, k);
                    }

                    U.forward_substitution_T_s(t, t.get_offset(), Vj, Aj);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Xv(j, k) = Aj[k];
                    }
                }
                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Ur, Xv);
                X.set_epsilon(Z.m_tree_data->m_epsilon);
                X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->set_low_rank_data(Xlr);
                X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->delete_children();
            }
        } else {
            auto &t_children = t.get_children();
            auto &s_children = s.get_children();
            // on regarde si on peut descendre
            for (int i = 0; i < t_children.size(); ++i) {
                for (auto &s_child : s_children) {
                    auto x_child = X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->add_child(s_child.get(), t_children[i].get());
                    FM_T_build(U, *t_children[i], *s_child, Z, X);
                    for (int j = i + 1; j < s_children.size(); ++j) {
                        auto Zsj = Z.get_block(s_child->get_size(), t_children[j]->get_size(), s_child->get_offset(), t_children[j]->get_offset());
                        auto Uij = U.get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset());
                        auto Xsi = X.get_block(s_child->get_size(), t_children[i]->get_size(), s_child->get_offset(), t_children[i]->get_offset());
                        Xsi->set_admissibility_condition(Zsj->get_admissibility_condition());
                        Xsi->set_low_rank_generator(Zsj->get_low_rank_generator());
                        // std::cout << Xsi->get_target_cluster().get_size() << ',' << Xsi->get_source_cluster().get_size() << "    " << Uij->get_target_cluster().get_size() << ',' << Uij->get_source_cluster().get_size() << std::endl;
                        // std::cout << Uij->is_dense() << ',' << Uij->is_low_rank() << std::endl;
                        if ((Uij->get_target_cluster().get_size() != t_children[i]->get_size()) or (Uij->get_source_cluster().get_size() != t_children[j]->get_size())) {
                            Matrix<double> udense(Uij->get_target_cluster().get_size(), Uij->get_source_cluster().get_size());
                            copy_to_dense(*Uij, udense.data());
                            Matrix<double> urestr(t_children[i]->get_size(), t_children[j]->get_size());
                            for (int k = 0; k < t_children[i]->get_size(); ++k) {
                                for (int l = 0; l < t_children[j]->get_size(); ++l) {
                                    urestr(k, l) = udense(k + t_children[i]->get_offset() - Uij->get_target_cluster().get_offset(), l + t_children[j]->get_offset() - Uij->get_source_cluster().get_offset());
                                }
                            }
                            Matrix<double> zdense(s_child->get_size(), t_children[j]->get_size());
                            Matrix<double> xdense(s_child->get_size(), t_children[i]->get_size());
                            copy_to_dense(*Zsj, zdense.data());
                            copy_to_dense(*Xsi, xdense.data());
                            zdense = zdense - urestr * xdense;
                            Zsj->set_dense_data(zdense);
                        } else {
                            Zsj->Moins(Xsi->hmatrix_product_new(*Uij));
                        }
                    }
                }
            }
        }
    }

    friend void
    FM_TT(const HMatrix &U, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &Z, HMatrix &X) {
        Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
        Matrix<CoefficientPrecision> zdense(s.get_size(), s.get_size());
        auto Zts = Z.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset());
        auto Ut  = U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Zts->get_children().size() > 0) {
            auto &t_children = t.get_children();
            for (int i = 0; i < t_children.size(); ++i) {
                auto &ti         = t_children[i];
                auto &s_children = s.get_children();
                for (auto &s_child : s_children) {
                    FM_T(U, *ti, *s_child, Z, X);
                    for (int j = i + 1; j < t_children.size(); ++j) {
                        auto &tj   = t_children[j];
                        auto Zstj  = Z.get_block(s_child->get_size(), tj->get_size(), s_child->get_offset(), tj->get_offset());
                        auto Utitj = U.get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset());
                        auto Xsi   = X.get_block(s_child->get_size(), ti->get_size(), s_child->get_offset(), ti->get_offset());
                        Xsi->set_admissibility_condition(Z.get_admissibility_condition());
                        Xsi->set_low_rank_generator(Z.get_low_rank_generator());
                        Zstj->Moins(Xsi->hmatrix_product(*Utitj));
                    }
                }
            }
        } else {
            if (Zts->is_dense()) {
                Matrix<CoefficientPrecision> Zdense(s.get_size(), t.get_size());
                copy_to_dense(*Zts, Zdense.data());
                Matrix<CoefficientPrecision> Xupdate(s.get_size(), t.get_size());
                for (int j = 0; j < s.get_size(); ++j) {
                    std::vector<CoefficientPrecision> Xjs(t.get_size(), 0.0);
                    std::vector<CoefficientPrecision> Zjs(t.get_size(), 0.0);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Zjs[k] = Zdense(j, k);
                    }
                    U.forward_substitution_T_s(t, t.get_offset(), Zjs, Xjs);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Xupdate(j, k) = Xjs[k];
                    }
                }
                X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->set_dense_data(Xupdate);
                X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->delete_children();
            } else {
                auto Ur = Zts->get_low_rank_data()->Get_U();
                auto Vr = Zts->get_low_rank_data()->Get_V();
                std::cout << "U : " << Ur.nb_rows() << ',' << Ur.nb_cols() << " de norme " << normFrob(Ur) << " et V : " << Vr.nb_rows() << ',' << Vr.nb_cols() << " de norme " << normFrob(Vr) << std::endl;
                Matrix<double> ttest(s.get_size(), t.get_size());
                copy_to_dense(*Zts, ttest.data());
                std::cout << " Zts ?? " << Zts->get_children().size() << ',' << Zts->is_dense() << ',' << Zts->is_low_rank() << " de norme =  " << normFrob(ttest) << std::endl;
                Matrix<CoefficientPrecision> Xv(Ur.nb_cols(), s.get_size());

                for (int j = 0; j < Zts->get_low_rank_data()->rank_of(); ++j) {
                    std::vector<CoefficientPrecision> Vj(s.get_size());
                    std::vector<CoefficientPrecision> Aj(s.get_size());
                    for (int k = 0; k < s.get_size(); ++k) {
                        Vj[k] = Vr(j, k);
                    }
                    Matrix<CoefficientPrecision> us(t.get_size(), t.get_size());
                    copy_to_dense(*U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), us.data());

                    U.forward_substitution_T_s(t, t.get_offset(), Vj, Aj);
                    for (int k = 0; k < s.get_size(); ++k) {
                        Xv(j, k) = Aj[k];
                    }
                }
                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Ur, Xv);
                X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->set_low_rank_data(Xlr);
                X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->delete_children();
            }
        }
    }

    ///////////////////////////////////////////////
    //// FORWARD_MATRIX_T : XU=Z ----------------------------------------->
    /// f(s,t)-> trouve Z|s,t ^T = Utt^T * Xts
    //////////////////////////

    // friend void Forward_Matrix_T(const HMatrix &U, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &Z, HMatrix &X) {
    //     Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
    //     Matrix<CoefficientPrecision> zdense(s.get_size(), s.get_size());
    //     auto Zts = Z.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset());

    //     if (Zts->get_children().size() > 0) {
    //         auto &t_children = t.get_children();
    //         for (int i = 0; i < t_children.size(); ++i) {
    //             auto &ti         = t_children[i];
    //             auto &s_children = s.get_children();
    //             for (auto &s_child : s_children) {
    //                 Forward_Matrix_T(U, *ti, *s_child, Z, X);
    //                 for (int j = i + 1; j < t_children.size(); ++j) {
    //                     auto &tj   = t_children[j];
    //                     auto Zstj  = Z.get_block(s_child->get_size(), tj->get_size(), s_child->get_offset(), tj->get_offset());
    //                     auto Utitj = U.get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset());
    //                     auto Xsi   = X.get_block(s_child->get_size(), ti->get_size(), s_child->get_offset(), ti->get_offset());
    //                     // std::cout << "Z : " << Zstj->get_target_cluster().get_size() << ',' << Zstj->get_source_cluster().get_size() << " | et on demande :" << s_child->get_size() << ',' << tj->get_size() << std::endl;
    //                     // std::cout << "X : " << Xsi->get_target_cluster().get_size() << ',' << Xsi->get_source_cluster().get_size() << "| et on demande :" << s_child->get_size() << ',' << ti->get_size() << std::endl;
    //                     // std::cout << "U :" << Utitj->get_target_cluster().get_size() << ',' << Utitj->get_source_cluster().get_size() << "| et on demande :" << ti->get_size() << ',' << tj->get_size() << std::endl;
    //                     if ((Utitj->get_target_cluster().get_size() == ti->get_size()) and (Utitj->get_source_cluster().get_size() == tj->get_size()) and (Xsi->get_target_cluster().get_size() == s_child->get_size()) and (Xsi->get_source_cluster().get_size() == ti->get_size())) {
    //                         Xsi->set_admissibility_condition(Z.get_admissibility_condition());
    //                         Xsi->set_low_rank_generator(Z.get_low_rank_generator());
    //                         Zstj->Moins(Xsi->hmatrix_product(*Utitj));
    //                     } else {
    //                         auto u = Utitj->extract(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset());
    //                         auto z = Zstj->extract(s_child->get_size(), tj->get_size(), s_child->get_offset(), tj->get_offset());
    //                         auto x = Xsi->extract(s_child->get_size(), ti->get_size(), s_child->get_offset(), ti->get_offset());
    //                         std::cout << "norme :" << normFrob(x) << "," << normFrob(z) << ',' << normFrob(u) << std::endl;
    //                         z = z - x * u;
    //                         Zstj->set_dense_data(z);
    //                         Z.assign(*Zstj, *s_child, *tj);
    //                     }
    //                     // std::cout << "Uis == nulptr ? :" << (Uis == nullptr) << std::endl;
    //                     // if ((Ztjs->get_target_cluster().get_size() == tj->get_size()) && (Ztjs->get_source_cluster().get_size() == s_child->get_size()) && (Xtji->get_target_cluster().get_size() == tj->get_size()) && (Xtji->get_source_cluster().get_size() == ti->get_size()) && (Uis->get_target_cluster().get_size() == ti->get_size()) && (Uis->get_source_cluster().get_size() == s_child->get_size())) {
    //                     // if ((Ztjs == nullptr) or (Uis == nullptr) or (Xtji == nullptr))
    //                     // std::cout << "yoooooooooooooo" << std::endl;
    //                     // std::cout << (Xsi->get_admissibility_condition() == nullptr) << std::endl;
    //                     // if ((Xsi->get_source_cluster() == Utitj->get_target_cluster()) and (Xsi->get_target_cluster() == Zstj->get_target_cluster()) and (Utitj->get_source_cluster() == Zstj->get_source_cluster())) {
    //                     // Xsi->set_admissibility_condition(Z.get_admissibility_condition());
    //                     // Xsi->set_low_rank_generator(Z.get_low_rank_generator());
    //                     // std::cout << "même taille ? " << Xsi->get_target_cluster().get_size() << ',' << Xsi->get_source_cluster().get_size() << ',' << Utitj->get_target_cluster().get_size() << ',' << Utitj->get_source_cluster().get_size() << std::endl;
    //                     // std::cout << " ? = " << Zstj->get_target_cluster().get_size() << ',' << Zstj->get_source_cluster().get_size() << std::endl;
    //                     // Zstj->Moins(Xsi->hmatrix_product(*Utitj));
    //                     // } else {
    //                     //     auto xx = Xsi->extract(s_child->get_size(), ti->get_size(), s_child->get_offset(), ti->get_offset());
    //                     //     auto uu = Utitj->extract(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset());
    //                     //     Matrix<CoefficientPrecision> zz(s_child->get_size(), tj->get_size());
    //                     //     copy_to_dense(*Zstj, zz.data());
    //                     //     zz = zz - xx * uu;
    //                     //     std::cout << "zz :" << normFrob(zz) << std::endl;
    //                     //     Zstj->set_dense_data(zz);
    //                     // }
    //                     // } else {
    //                     //     // std::cout << "yiiiiiiiiiiiiiiiiiiiii" << std::endl;
    //                     //     Matrix<CoefficientPrecision> Zdense(Ztjs->get_target_cluster().get_size(), Ztjs->get_source_cluster().get_size());
    //                     //     Matrix<CoefficientPrecision> Xdense(Xtji->get_target_cluster().get_size(), Xtji->get_source_cluster().get_size());
    //                     //     Matrix<CoefficientPrecision> Udense(Uis->get_target_cluster().get_size(), Uis->get_source_cluster().get_size());
    //                     //     Matrix<CoefficientPrecision> Zrestr(tj->get_size(), s_child->get_size());
    //                     //     Matrix<CoefficientPrecision> Xrestr(tj->get_size(), ti->get_size());
    //                     //     Matrix<CoefficientPrecision> Urestr(ti->get_size(), s_child->get_size());
    //                     //     copy_to_dense(*Ztjs, Zdense.data());
    //                     //     copy_to_dense(*Xtji, Xdense.data());
    //                     //     copy_to_dense(*Uis, Udense.data());
    //                     //     for (int k = 0; k < tj->get_size(); ++k) {
    //                     //         for (int l = 0; l < s_child->get_size(); ++l) {
    //                     //             Zrestr(k, l) = Zdense(k + tj->get_offset() - Ztjs->get_target_cluster().get_offset(), l + s_child->get_offset() - Ztjs->get_source_cluster().get_offset());
    //                     //         }
    //                     //         for (int l = 0; l < ti->get_size(); ++l) {
    //                     //             Xrestr(k, l) = Xdense(k + tj->get_offset() - Xtji->get_target_cluster().get_offset(), l + ti->get_offset() - Xtji->get_source_cluster().get_offset());
    //                     //         }
    //                     //     }
    //                     //     for (int k = 0; k < ti->get_size(); ++k) {
    //                     //         for (int l = 0; l < s_child->get_size(); ++l) {
    //                     //             Urestr(k, l) = Udense(k + ti->get_offset() - Uis->get_target_cluster().get_offset(), l + s_child->get_offset() - Uis->get_source_cluster().get_size());
    //                     //         }
    //                     //     }
    //                     //     Zrestr = Zrestr - Xrestr * Urestr;
    //                     //     Ztjs->set_dense_data(Zrestr);
    //                     // }
    //                 }
    //             }
    //         }
    //     } else {
    //         if (Zts->is_dense()) {
    //             auto Ut = U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
    //             Matrix<CoefficientPrecision> Udense(Ut->get_target_cluster().get_size(), Ut->get_source_cluster().get_size());

    //             copy_to_dense(*Ut, Udense.data());
    //             std::cout << t.get_size() << ',' << U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_target_cluster().get_size() << ',' << U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_source_cluster().get_size();

    //             auto Zdense = *Zts->get_dense_data();
    //             auto ZZ     = Zdense;
    //             Matrix<CoefficientPrecision> Xupdate(s.get_size(), t.get_size());
    //             for (int j = 0; j < s.get_size(); ++j) {
    //                 std::vector<CoefficientPrecision> Xjs(U.get_target_cluster().get_size(), 0.0);
    //                 std::vector<CoefficientPrecision> Zjs(U.get_source_cluster().get_size(), 0.0);
    //                 for (int k = 0; k < t.get_size(); ++k) {
    //                     Zjs[k + t.get_offset() - U.get_target_cluster().get_offset()] = Zdense(j, k);
    //                 }
    //                 for (int k = 0; k < t.get_size(); ++k) {
    //                     Xupdate(j, k) = Xjs[k + t.get_offset() - U.get_target_cluster().get_offset()];
    //                 }
    //             }
    //             X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->set_dense_data(Xupdate);
    //             X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->delete_children();
    //             // std::cout << " on affecte une valeur de taille : " << Xupdate.nb_rows() << "," << Xupdate.nb_cols() << std::endl;
    //             // std::cout << " sur un bloc de taille " << X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->get_target_cluster().get_size() << ',' << X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->get_source_cluster().get_size() << std::endl;

    //             copy_to_dense(*X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset()), Xupdate.data());
    //             std::cout << "erreur sur le bloc push Z-XU  =:" << normFrob(Xupdate * Udense - ZZ) / normFrob(ZZ) << std::endl;

    //         } else {
    //             // std::cout << "!!!!!!" << std::endl;
    //             auto Ur = Zts->get_low_rank_data()->Get_U();
    //             auto Vr = Zts->get_low_rank_data()->Get_V();
    //             Matrix<CoefficientPrecision> Xv(Zts->get_low_rank_data()->rank_of(), s.get_size());
    //             for (int j = 0; j < Zts->get_low_rank_data()->rank_of(); ++j) {
    //                 std::vector<CoefficientPrecision> Vj(U.get_source_cluster().get_size());
    //                 std::vector<CoefficientPrecision> Aj(U.get_source_cluster().get_size());
    //                 for (int k = 0; k < s.get_size(); ++k) {
    //                     Vj[k + s.get_offset() - U.get_source_cluster().get_offset()] = Vr(j, k);
    //                 }
    //                 U.forward_substitution_T(s, Aj, Vj);
    //                 for (int k = 0; k < s.get_size(); ++k) {
    //                     Xv(j, k) = Aj[k + s.get_offset() - U.get_target_cluster().get_offset()];
    //                 }
    //             }
    //             LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Ur, Xv);
    //             X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->set_low_rank_data(Xlr);
    //             X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->delete_children();
    //         }
    //     }
    // }

    friend void
    Forward_Matrix_T(const HMatrix &U, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &Z, HMatrix &X) {
        // Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
        // Matrix<CoefficientPrecision> zdense(s.get_size(), s.get_size());
        auto Zts = Z.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset());
        // copy_to_dense(*Zts, zdense.data());
        // copy_to_dense(*U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), udense.data());
        // std::cout << ":::::::::::::::::::::::::::::::::::" << std::endl;
        // std::cout << "               appel de Forward_T pour trouver X tq XU=H avec norme de U et H =" << normFrob(udense) << "      et     " << normFrob(zdense) << std::endl;
        // std::cout << " sur   t s = " << '(' << s.get_size() << ',' << s.get_offset() << ')' << ';' << '(' << t.get_size() << ',' << t.get_offset() << ')' << std::endl;
        auto Ut = U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Ut->is_dense()) {
            Matrix<CoefficientPrecision> Zdense(s.get_size(), t.get_size());
            copy_to_dense(*Zts, Zdense.data());
            // auto ZZ = Zdense;
            Matrix<CoefficientPrecision> Xupdate(s.get_size(), t.get_size());
            for (int j = 0; j < s.get_size(); ++j) {
                std::vector<CoefficientPrecision> Xjs(U.get_target_cluster().get_size(), 0.0);
                std::vector<CoefficientPrecision> Zjs(U.get_source_cluster().get_size(), 0.0);
                for (int k = 0; k < t.get_size(); ++k) {
                    Zjs[k + t.get_offset() - U.get_target_cluster().get_offset()] = Zdense(j, k);
                }
                U.forward_substitution_T(t, Xjs, Zjs);
                for (int k = 0; k < t.get_size(); ++k) {
                    Xupdate(j, k) = Xjs[k + t.get_offset() - U.get_target_cluster().get_offset()];
                }
            }

            X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->set_dense_data(Xupdate);
            X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->delete_children();
            // Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
            // Matrix<CoefficientPrecision> xdense(s.get_size(), t.get_size());
            // copy_to_dense(*X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset()), xdense.data());
            // std::cout << "backwaard a push un bloc de norme " << normFrob(Xupdate) << std::endl;

            // copy_to_dense(*U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), udense.data());
            // std::cout << "------------------backward cas 1 : erreur Z-XU" << normFrob(ZZ - xdense * udense) / normFrob(ZZ) << std::endl;
            // std::cout << "avec Z de norme " << normFrob(ZZ) << " bloc ts = "
            //           << "(" << s.get_size() << ',' << s.get_offset() << "),(" << t.get_size() << ',' << t.get_offset() << std::endl;
            // if (Zts->is_low_rank()) {
            //     std::cout << "pourtant low rank avec norme de U norme de V = " << normFrob(Zts->get_low_rank_data()->Get_U()) << ',' << normFrob(Zts->get_low_rank_data()->Get_V()) << "et le produi t : " << normFrob(Zts->get_low_rank_data()->Get_U() * Zts->get_low_rank_data()->Get_V()) << std::endl;
            // }
            // std::cout << "et Zts de taille " << Zts->get_target_cluster().get_size() << ',' << Zts->get_source_cluster().get_size() << std::endl;
            // std::cout << "nb child : " << Zts->get_children().size() << ',' << "is dense ?" << Zts->is_dense() << std::endl;
            // if (Zts->is_dense()) {
            //     auto l = *Zts->get_dense_data();
            //     for (int k = 0; k < l.nb_rows(); ++k) {
            //         for (int m = 0; m < l.nb_cols() - 1; ++m) {
            //             std::cout << l(k, m) << ',';
            //         }
            //         std::cout << l(k, l.nb_cols() - 1) << std::endl;
            //     }
            // }
        } else {
            if (Zts->get_children().size() > 0) {
                auto &t_children = t.get_children();
                for (int i = 0; i < t_children.size(); ++i) {
                    auto &ti         = t_children[i];
                    auto &s_children = s.get_children();
                    for (auto &s_child : s_children) {
                        Forward_Matrix_T(U, *ti, *s_child, Z, X);
                        for (int j = i + 1; j < t_children.size(); ++j) {
                            auto &tj   = t_children[j];
                            auto Zstj  = Z.get_block(s_child->get_size(), tj->get_size(), s_child->get_offset(), tj->get_offset());
                            auto Utitj = U.get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset());
                            auto Xsi   = X.get_block(s_child->get_size(), ti->get_size(), s_child->get_offset(), ti->get_offset());
                            // std::cout << "Uis == nulptr ? :" << (Uis == nullptr) << std::endl;
                            // if ((Ztjs->get_target_cluster().get_size() == tj->get_size()) && (Ztjs->get_source_cluster().get_size() == s_child->get_size()) && (Xtji->get_target_cluster().get_size() == tj->get_size()) && (Xtji->get_source_cluster().get_size() == ti->get_size()) && (Uis->get_target_cluster().get_size() == ti->get_size()) && (Uis->get_source_cluster().get_size() == s_child->get_size())) {
                            // if ((Ztjs == nullptr) or (Uis == nullptr) or (Xtji == nullptr))
                            // std::cout << "yoooooooooooooo" << std::endl;
                            // std::cout << (Xsi->get_admissibility_condition() == nullptr) << std::endl;
                            // if ((Xsi->get_source_cluster() == Utitj->get_target_cluster()) and (Xsi->get_target_cluster() == Zstj->get_target_cluster()) and (Utitj->get_source_cluster() == Zstj->get_source_cluster())) {
                            Matrix<CoefficientPrecision> zdense(s_child->get_size(), t_children[j]->get_size());
                            copy_to_dense(*Zstj, zdense.data());
                            Xsi->set_admissibility_condition(Z.get_admissibility_condition());
                            Xsi->set_low_rank_generator(Z.get_low_rank_generator());
                            // std::cout << "même taille ? " << Xsi->get_target_cluster().get_size() << ',' << Xsi->get_source_cluster().get_size() << ',' << Utitj->get_target_cluster().get_size() << ',' << Utitj->get_source_cluster().get_size() << std::endl;
                            // std::cout << " ? = " << Zstj->get_target_cluster().get_size() << ',' << Zstj->get_source_cluster().get_size() << std::endl;
                            // std::cout << "norme avent le moins " << normFrob(zdense) << std::endl;
                            // Matrix<CoefficientPrecision> dense_prod(s_child->get_size(), t_children[j]->get_size());
                            // copy_to_dense(Xsi->hmatrix_product(*Utitj), dense_prod.data());
                            // std::cout << Zstj->get_target_cluster().get_size() << ',' << Zstj->get_source_cluster().get_size() << " = (" << Xsi->get_target_cluster().get_size() << ',' << Xsi->get_source_cluster().get_size() << ")x(" << Utitj->get_target_cluster().get_size() << ',' << Utitj->get_source_cluster().get_size() << ")" << std::endl;
                            // Matrix<CoefficientPrecision> xdense(s_child->get_size(), t_children[j]->get_size());
                            // Matrix<CoefficientPrecision> udense(t_children[i]->get_size(), t_children[j]->get_size());
                            // copy_to_dense(*Utitj, udense.data());
                            // copy_to_dense(*Xsi, xdense.data());
                            // std::cout << "norme du produit XU, X et U de Z-Xu  avant le moins: " << normFrob(dense_prod) << ',' << normFrob(xdense) << ',' << normFrob(udense) << std::endl;
                            Zstj->Moins(Xsi->hmatrix_product(*Utitj));
                            // Zstj->Moins(Xsi->hmatrix_product(*Utitj), *s_child, *t_children[i]);
                            // Z.assign(*Zstj, *s_child, *tj);
                            // copy_to_dense(*Zstj, zdense.data());
                            // std::cout << "norme aprés Moins de forward" << normFrob(zdense) << std::endl;
                            // } else {
                            //     auto xx = Xsi->extract(s_child->get_size(), ti->get_size(), s_child->get_offset(), ti->get_offset());
                            //     auto uu = Utitj->extract(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset());
                            //     Matrix<CoefficientPrecision> zz(s_child->get_size(), tj->get_size());
                            //     copy_to_dense(*Zstj, zz.data());
                            //     zz = zz - xx * uu;
                            //     std::cout << "zz :" << normFrob(zz) << std::endl;
                            //     Zstj->set_dense_data(zz);
                            // }
                            // } else {
                            //     // std::cout << "yiiiiiiiiiiiiiiiiiiiii" << std::endl;
                            //     Matrix<CoefficientPrecision> Zdense(Ztjs->get_target_cluster().get_size(), Ztjs->get_source_cluster().get_size());
                            //     Matrix<CoefficientPrecision> Xdense(Xtji->get_target_cluster().get_size(), Xtji->get_source_cluster().get_size());
                            //     Matrix<CoefficientPrecision> Udense(Uis->get_target_cluster().get_size(), Uis->get_source_cluster().get_size());
                            //     Matrix<CoefficientPrecision> Zrestr(tj->get_size(), s_child->get_size());
                            //     Matrix<CoefficientPrecision> Xrestr(tj->get_size(), ti->get_size());
                            //     Matrix<CoefficientPrecision> Urestr(ti->get_size(), s_child->get_size());
                            //     copy_to_dense(*Ztjs, Zdense.data());
                            //     copy_to_dense(*Xtji, Xdense.data());
                            //     copy_to_dense(*Uis, Udense.data());
                            //     for (int k = 0; k < tj->get_size(); ++k) {
                            //         for (int l = 0; l < s_child->get_size(); ++l) {
                            //             Zrestr(k, l) = Zdense(k + tj->get_offset() - Ztjs->get_target_cluster().get_offset(), l + s_child->get_offset() - Ztjs->get_source_cluster().get_offset());
                            //         }
                            //         for (int l = 0; l < ti->get_size(); ++l) {
                            //             Xrestr(k, l) = Xdense(k + tj->get_offset() - Xtji->get_target_cluster().get_offset(), l + ti->get_offset() - Xtji->get_source_cluster().get_offset());
                            //         }
                            //     }
                            //     for (int k = 0; k < ti->get_size(); ++k) {
                            //         for (int l = 0; l < s_child->get_size(); ++l) {
                            //             Urestr(k, l) = Udense(k + ti->get_offset() - Uis->get_target_cluster().get_offset(), l + s_child->get_offset() - Uis->get_source_cluster().get_size());
                            //         }
                            //     }
                            //     Zrestr = Zrestr - Xrestr * Urestr;
                            //     Ztjs->set_dense_data(Zrestr);
                            // }
                        }
                    }
                }
            } else {
                if (Zts->is_dense()) {
                    // Matrix<CoefficientPrecision> Udense(t.get_size(), t.get_size());
                    // copy_to_dense(*U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), Udense.data());
                    auto Zdense = *Zts->get_dense_data();
                    // auto ZZ     = Zdense;
                    // std::cout << "forward  sur le bloc "
                    //           << "(" << s.get_size() << ',' << s.get_offset() << "),(" << t.get_size() << ',' << t.get_offset() << " avec  Z est dense de norme " << normFrob(Zdense) << std::endl;

                    Matrix<CoefficientPrecision> Xupdate(s.get_size(), t.get_size());
                    Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
                    copy_to_dense(*U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), udense.data());
                    for (int j = 0; j < s.get_size(); ++j) {
                        std::vector<CoefficientPrecision> Xjs(U.get_target_cluster().get_size(), 0.0);
                        std::vector<CoefficientPrecision> Zjs(U.get_source_cluster().get_size(), 0.0);
                        std::vector<CoefficientPrecision> zz(t.get_size(), 0.0);
                        for (int k = 0; k < t.get_size(); ++k) {
                            Zjs[k + t.get_offset() - U.get_target_cluster().get_offset()] = Zdense(j, k);
                            zz[k]                                                         = Zdense(j, k);
                        }
                        U.forward_substitution_T(t, Xjs, Zjs);
                        std::vector<CoefficientPrecision> testt(t.get_size());
                        for (int k = 0; k < t.get_size(); ++k) {
                            Xupdate(j, k) = Xjs[k + t.get_offset() - U.get_target_cluster().get_offset()];
                            testt[k]      = Xjs[k + t.get_offset() - U.get_target_cluster().get_offset()];
                        }
                        // std::cout << "erreur veceur xU :" << norm2(zz - udense.transp(udense) * testt) / norm2(zz) << " avec zz = " << norm2(zz) << std::endl;
                    }
                    X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->set_dense_data(Xupdate);
                    X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->delete_children();

                    // copy_to_dense(*X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset()), Xupdate.data());
                    // Matrix<CoefficientPrecision> xdense(s.get_size(), t.get_size());
                    // copy_to_dense(*X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset()), xdense.data());

                    // std::cout << "------------------backward cas 2 : erreur Z-XU" << normFrob(ZZ - xdense * udense) / normFrob(ZZ) << std::endl;
                } else {
                    // std::cout << "!!!!!!" << std::endl;
                    auto Ur = Zts->get_low_rank_data()->Get_U();
                    auto Vr = Zts->get_low_rank_data()->Get_V();
                    // std::cout << "/////////////   U et V \\\\\\\\\\\\\\ " << Ur.nb_rows() << ',' << Ur.nb_cols() << "! " << Vr.nb_rows() << ',' << Vr.nb_cols() << std::endl;
                    Matrix<CoefficientPrecision> Xv(Zts->get_low_rank_data()->rank_of(), s.get_size());

                    for (int j = 0; j < Zts->get_low_rank_data()->rank_of(); ++j) {
                        std::vector<CoefficientPrecision> Vj(U.get_source_cluster().get_size());
                        std::vector<CoefficientPrecision> Aj(U.get_source_cluster().get_size());
                        for (int k = 0; k < s.get_size(); ++k) {
                            Vj[k + s.get_offset() - U.get_source_cluster().get_offset()] = Vr(j, k);
                        }
                        // std::cout << "on appelle forward_T sur un vecteur de norme " << norm2(Vj) << std::endl;
                        Matrix<CoefficientPrecision> us(t.get_size(), t.get_size());
                        copy_to_dense(*U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), us.data());
                        Matrix<CoefficientPrecision> udense(U.get_target_cluster().get_size(), U.get_source_cluster().get_size());
                        copy_to_dense(U, udense.data());
                        // std::cout << "on appelle forward_T sur un vecteur de norme " << norm2(Vj) << "et Utt de norme " << normFrob(us) << "   pour U de norme " << normFrob(udense) << std::endl;
                        // std::cout << " bonne taille " << t.get_size() << ',' << t.get_size() << "= ? " <<

                        U.forward_substitution_T(t, Aj, Vj);
                        // std::cout << "forward T nous sort un vecteur de norme " << norm2(Aj) << std::endl;
                        for (int k = 0; k < s.get_size(); ++k) {
                            Xv(j, k) = Aj[k + s.get_offset() - U.get_target_cluster().get_offset()];
                        }
                    }
                    // std::cout << normFrob(Ur) << " et " << normFrob(Xv) << std::endl;
                    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Ur, Xv);
                    X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->set_low_rank_data(Xlr);
                    X.get_block(s.get_size(), t.get_size(), s.get_offset(), t.get_offset())->delete_children();
                    // std::cout << "backward lr a push une feuille de norme " << normFrob(Ur * Xv) << std::endl;
                }
            }
        }
    }

    // friend void Forward_Matrix_T(const HMatrix &U, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &Z, HMatrix &X) {
    //     auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //     Matrix<CoefficientPrecision> udense(s.get_size(), s.get_size());
    //     Matrix<CoefficientPrecision> zdense(t.get_size(), s.get_size());
    //     Matrix<CoefficientPrecision> xdense(t.get_size(), s.get_size());
    //     copy_to_dense(*Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset()), zdense.data());
    //     copy_to_dense(*X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset()), xdense.data());
    //     copy_to_dense(*U.get_block(s.get_size(), s.get_size(), s.get_offset(), s.get_offset()), udense.data());
    //     std::cout << "on veut résoudre X(" << t.get_size() << ',' << t.get_offset() << '|' << s.get_size() << ',' << s.get_offset() << ")U(" << s.get_size() << ',' << s.get_offset() << '|' << s.get_size() << ',' << s.get_offset() << ") = Z(" << t.get_size() << ',' << t.get_offset() << '|' << s.get_size() << ',' << s.get_offset() << ") avec norme(X) =" << normFrob(xdense) << ", norm(U) = " << normFrob(xdense) << ", norm(Z) = " << normFrob(zdense) << std::endl;
    //     if (Zts->get_children().size() == 0) {
    //         if (Zts->is_dense()) {
    //             auto z = *Zts->get_dense_data();
    //             Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
    //             for (int i = 0; i < t.get_size(); ++i) {
    //                 std::vector<CoefficientPrecision> Z_line_i(X.get_source_cluster().get_size());
    //                 for (int j = 0; j < s.get_size(); ++j) {
    //                     Z_line_i[j + s.get_offset()] = z(i, j);
    //                 }
    //                 std::vector<CoefficientPrecision> x_line_i(X.get_source_cluster().get_size());
    //                 // std::cout << "on appelle Uforward T" << std::endl;
    //                 U.forward_substitution_T(s, Z_line_i, x_line_i);
    //                 // std::cout << "uforward T ok" << std::endl;
    //                 for (int j = 0; j < s.get_size(); ++j) {
    //                     Xupdate(i, j) = x_line_i[j + s.get_offset()];
    //                 }
    //             }
    //             X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xupdate);
    //         } else {
    //             auto A = Zts->get_low_rank_data()->Get_U();
    //             auto B = Zts->get_low_rank_data()->Get_V();
    //             Matrix<CoefficientPrecision> Bupdate(B.nb_rows(), s.get_size());
    //             for (int i = 0; i < B.nb_rows(); ++i) {
    //                 std::vector<CoefficientPrecision> B_line_i(X.get_source_cluster().get_size());
    //                 for (int j = 0; j < s.get_size(); ++j) {
    //                     B_line_i[j + s.get_offset()] = B(i, j);
    //                 }
    //                 std::vector<CoefficientPrecision> b_line_i(X.get_source_cluster().get_size());
    //                 U.forward_substitution_T(s, B_line_i, b_line_i);
    //                 for (int j = 0; j < s.get_size(); ++j) {
    //                     Bupdate(i, j) = b_line_i[j + s.get_offset()];
    //                 }
    //             }
    //             LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(A, Bupdate);
    //             X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_low_rank_data(lr);
    //         }
    //     } else {
    //         auto &t_children = t.get_children();
    //         for (int i = 0; i < t_children.size(); ++i) {
    //             auto &child_t = t.get_children()[i];
    //             for (auto &child_s : s.get_children()) {
    //                 Forward_Matrix_T(U, *child_t, *child_s, Z, X);
    //                 for (auto &child_tt : s.get_children()) {
    //                     Matrix<double> ztest(child_t->get_size(), child_s->get_size());
    //                     auto Ztemp = Z.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
    //                     auto Utemp = U.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
    //                     auto Xtemp = X.get_block(child_t->get_size(), child_tt->get_size(), child_t->get_offset(), child_tt->get_offset());
    //                     // Je sais poas pourquoi mais parfois les blocs sont trop gros et il faut les restreindre
    //                     if ((Ztemp->get_target_cluster().get_size() == child_t->get_size()) && (Ztemp->get_source_cluster().get_size() == child_s->get_size()) && (Utemp->get_target_cluster().get_size() == child_tt->get_size()) && (Utemp->get_source_cluster().get_size() == child_s->get_size()) && (Xtemp->get_target_cluster().get_size() == child_t->get_size()) && (Xtemp->get_source_cluster().get_size() == child_tt->get_size())) {
    //                         // copy_to_dense(*Ztemp, ztest.data());
    //                         // std::cout << "avant : " << normFrob(ztest) << std::endl;
    //                         auto prod = Xtemp->hmatrix_product(*Utemp);
    //                         Ztemp->Moins(prod);
    //                         // copy_to_dense(*Ztemp, ztest.data());
    //                         // std::cout << "aprés : " << normFrob(ztest) << std::endl;
    //                     } else {
    //                         Matrix<CoefficientPrecision> zrestr(child_t->get_size(), child_s->get_size());
    //                         Matrix<CoefficientPrecision> urestr(child_tt->get_size(), child_s->get_size());
    //                         Matrix<CoefficientPrecision> xrestr(child_t->get_size(), child_tt->get_size());
    //                         Matrix<CoefficientPrecision> zdense(Ztemp->get_target_cluster().get_size(), Ztemp->get_source_cluster().get_size());
    //                         Matrix<CoefficientPrecision> udense(Utemp->get_target_cluster().get_size(), Utemp->get_source_cluster().get_size());
    //                         Matrix<CoefficientPrecision> xdense(Xtemp->get_target_cluster().get_size(), Xtemp->get_source_cluster().get_size());
    //                         copy_to_dense(*Ztemp, zdense.data());
    //                         copy_to_dense(*Utemp, udense.data());
    //                         copy_to_dense(*Xtemp, xdense.data());
    //                         for (int k = 0; k < child_t->get_size(); ++k) {
    //                             for (int l = 0; l < child_s->get_size(); ++l) {
    //                                 zrestr(k, l) = zdense(child_t->get_offset() - Ztemp->get_target_cluster().get_offset() + k, child_s->get_offset() - Ztemp->get_source_cluster().get_offset() + l);
    //                             }
    //                             for (int l = 0; l < child_tt->get_size(); ++l) {
    //                                 xrestr(k, l) = xdense(k + child_t->get_offset() - Xtemp->get_target_cluster().get_offset(), l + child_tt->get_size() - Xtemp->get_source_cluster().get_offset());
    //                             }
    //                         }
    //                         for (int k = 0; k < child_tt->get_size(); ++k) {
    //                             for (int l = 0; l < child_s->get_size(); ++l) {
    //                                 urestr(k, l) = udense(k + child_tt->get_offset() - Utemp->get_target_cluster().get_offset(), l + child_s->get_offset() - Utemp->get_source_cluster().get_offset());
    //                             }
    //                         }
    //                         zrestr = zrestr - xrestr * urestr;
    //                         Ztemp->set_dense_data(zrestr);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    /////////////////////////////////////////////////////
    friend void
    HMatrix_LU(HMatrix &H, const Cluster<CoordinatePrecision> &t, HMatrix &L, HMatrix &U) {
        auto Htt = H.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        // Si c'est dense on calcule LU
        // std::cout << "______________________________" << std::endl;
        // std::cout << "en haut de LU avec t=" << t.get_size() << ',' << t.get_offset() << std::endl;
        if (Htt->is_dense()) {
            // std::cout << "  on fait du dense : " << std::endl;
            Matrix<CoefficientPrecision> Hdense = *Htt->get_dense_data();
            auto hdense                         = Hdense;
            std::vector<int> vrai_pivot(t.get_size(), 0.0);
            Matrix<CoefficientPrecision> l(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> u(t.get_size(), t.get_size());
            get_lu_factorisation(hdense, l, u, vrai_pivot);
            Matrix<CoefficientPrecision> cyclmat(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> inv_cyclmat(t.get_size(), t.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                cyclmat(vrai_pivot[k] - 1, k)     = 1.0;
                inv_cyclmat(k, vrai_pivot[k] - 1) = 1.0;
            }
            Matrix<double> id(t.get_size(), t.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                id(k, k) = 1.0;
            }
            // std::cout << "  norm(permutaion-id)  " << normFrob(cyclmat - id) << std::endl;
            // std::cout << "  erreur invperm(M) -LU " << normFrob(inv_cyclmat * hdense - l * u) / normFrob(hdense) << std::endl;

            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            Matrix<CoefficientPrecision> ldense(t.get_size(), t.get_size());
            copy_to_dense(*L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), ldense.data());
            Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
            copy_to_dense(*U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), udense.data());
            std::cout << "  erreur sur la feuille " << t.get_size() << ',' << t.get_offset() << ':' << normFrob(inv_cyclmat * hdense - ldense * udense) / normFrob(hdense) << std::endl;
        } else {
            // std::cout << "  appel récursif" << std::endl;
            auto &t_children = t.get_children();
            for (int i = 0; i < t_children.size(); ++i) {
                HMatrix_LU(H, *t_children[i], L, U);
                for (int j = i + 1; j < t_children.size(); ++j) {
                    Matrix<CoefficientPrecision> ldense(t_children[j]->get_size(), t_children[i]->get_size());
                    Matrix<CoefficientPrecision> udense(t_children[i]->get_size(), t_children[j]->get_size());
                    Matrix<CoefficientPrecision> h12(t_children[i]->get_size(), t_children[j]->get_size());
                    Matrix<CoefficientPrecision> h21(t_children[j]->get_size(), t_children[i]->get_size());
                    Matrix<CoefficientPrecision> l11(t_children[i]->get_size(), t_children[i]->get_size());
                    Matrix<CoefficientPrecision> u11(t_children[i]->get_size(), t_children[i]->get_size());
                    copy_to_dense(*U.get_block(t_children[i]->get_size(), t_children[i]->get_size(), t_children[i]->get_offset(), t_children[i]->get_offset()), u11.data());
                    copy_to_dense(*L.get_block(t_children[i]->get_size(), t_children[i]->get_size(), t_children[i]->get_offset(), t_children[i]->get_offset()), l11.data());
                    copy_to_dense(*H.get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset()), h12.data());
                    // std::cout << "  appel de Forward M avec u11 et l11 de norme :" << normFrob(u11) << ',' << normFrob(l11) << std::endl;

                    // std::cout << "  M avec ts = " << t_children[j]->get_size() << ',' << t_children[i]->get_size() << ',' << t_children[j]->get_offset() << ',' << t_children[i]->get_offset() << std::endl;
                    // std::cout << "  sur une matrice  h12 de norme :" << normFrob(h12) << std::endl;
                    Forward_Matrix(L, *t_children[i], *t_children[j], H, U);
                    copy_to_dense(*U.get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset()), udense.data());
                    std::cout << "  --------------> erreur surn forward M : " << normFrob(l11 * udense - h12) / normFrob(h12) << std::endl;

                    // std::cout << "  norme de U12 de forward_M :      " << normFrob(udense) << std::endl;
                    copy_to_dense(*H.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset()), h21.data());
                    // std::cout << "  MT avec st = " << t_children[j]->get_size() << ',' << t_children[i]->get_size() << ',' << t_children[j]->get_offset() << ',' << t_children[i]->get_offset() << std::endl;
                    std::cout << "   h21 de norme :" << normFrob(h21) << std::endl;
                    Forward_Matrix_T(U, *t_children[i], *t_children[j], H, L);
                    copy_to_dense(*L.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset()), ldense.data());
                    std::cout << "  norme de L21 de forward_M_T :          " << normFrob(ldense) << std::endl;
                    std::cout << "  --------------> erreur sur forward M _T: " << normFrob(ldense * u11 - h21) / normFrob(h21) << std::endl;

                    for (int r = i + 1; r < t_children.size(); ++r) {
                        auto Htemp = H.get_block(t_children[j]->get_size(), t_children[r]->get_size(), t_children[j]->get_offset(), t_children[r]->get_offset());
                        auto Ltemp = L.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset());
                        auto Utemp = U.get_block(t_children[i]->get_size(), t_children[r]->get_size(), t_children[i]->get_offset(), t_children[r]->get_offset());
                        Htemp->Moins(Ltemp->hmatrix_product(*Utemp));
                        // Matrix<CoefficientPrecision> ldense1(t_children[j]->get_size(), t_children[i]->get_size());
                        // Matrix<CoefficientPrecision> udense1(t_children[i]->get_size(), t_children[r]->get_size());
                        // Matrix<CoefficientPrecision> hdense1(t_children[j]->get_size(), t_children[r]->get_size());
                        // copy_to_dense(*Utemp, udense1.data());
                        // copy_to_dense(*Ltemp, ldense1.data());
                        // copy_to_dense(*Htemp, hdense1.data());
                        // std::cout << "    normes facteur de prod : " << normFrob(ldense1) << ',' << normFrob(udense1) << std::endl;
                        // std::cout << "    norme de h-lu = " << normFrob(hdense1) << std::endl;
                    }
                }
            }
        }
    }
    /////////////// pour permuter les lignes des blocs STRICTEMENT INFERIEURS ,  perm de taille txt
    void permutation_down(const Matrix<double> perm, const Cluster<CoordinatePrecision> &t) {
        auto leaves = this->get_leaves();
        for (auto &l : leaves) {
            if ((l->get_target_cluster().get_offset() > t.get_offset()) and l->get_source_cluster().get_offset() < t.get_offset()) {
                if (l->get_target_cluster() == t) {
                    if (l->is_dense()) {
                        auto l_perm = perm * (*l->get_dense_data());
                        this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(l_perm);
                    } else {
                        auto U_perm = perm * l->get_low_rank_data()->Get_U();
                        LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_perm(U_perm, l->get_low_rank_data()->Get_V());
                        this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(lr_perm);
                    }
                } else {
                    Matrix<double> perm_restr(l->get_target_cluster().get_size(), l->get_target_cluster().get_size());
                    for (int i = 0; i < l->get_target_cluster().get_size(); ++i) {
                        for (int j = 0; j < l->get_target_cluster().get_size(); ++j) {
                            perm_restr(i, j) = perm(i + l->get_target_cluster().get_offset() - t.get_offset(), j + l->get_target_cluster().get_offset() - t.get_offset());
                        }
                    }
                    if (l->is_dense()) {
                        auto l_perm = perm_restr * (*l->get_dense_data());
                        this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(l_perm);
                    } else {
                        auto U_perm = perm_restr * l->get_low_rank_data()->Get_U();
                        LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_perm(U_perm, l->get_low_rank_data()->Get_V());
                        this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(lr_perm);
                    }
                }
            }
        }
    }

    /////// la on permute toute la matrice
    void permutation(const Matrix<double> perm, const Cluster<CoordinatePrecision> &t) {
        auto leaves = this->get_leaves();
        for (auto &l : leaves) {
            if (l->get_target_cluster().get_offset() < t.get_offset()) {
                if (l->get_target_cluster() == t) {
                    if (l->is_dense()) {
                        auto l_perm = perm * (*l->get_dense_data());
                        this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(l_perm);
                    } else {
                        auto U_perm = perm * l->get_low_rank_data()->Get_U();
                        LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_perm(U_perm, l->get_low_rank_data()->Get_V());
                        this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(lr_perm);
                    }
                } else {
                    Matrix<double> perm_restr(l->get_target_cluster().get_size(), l->get_target_cluster().get_size());
                    for (int i = 0; i < l->get_target_cluster().get_size(); ++i) {
                        for (int j = 0; j < l->get_target_cluster().get_size(); ++j) {
                            perm_restr(i, j) = perm(i + l->get_target_cluster().get_offset() - t.get_offset(), j + l->get_target_cluster().get_offset() - t.get_offset());
                        }
                    }
                    if (l->is_dense()) {
                        auto l_perm = perm_restr * (*l->get_dense_data());
                        this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_dense_data(l_perm);
                    } else {
                        auto U_perm = perm_restr * l->get_low_rank_data()->Get_U();
                        LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_perm(U_perm, l->get_low_rank_data()->Get_V());
                        this->get_block(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset())->set_low_rank_data(lr_perm);
                    }
                }
            }
        }
    }

    /////////////////////////////////////////////////////////
    //     ____________________
    //     |         |         |
    //     |   1     |         |
    //     |_________|_________|
    //     |         |         |
    //     |    2    |   3     |
    //     |_________|_________|
    //              4
    //          <------ permutation

    friend void HMatrix_PLU(HMatrix &H, const Cluster<CoordinatePrecision> &t, HMatrix &L, HMatrix &U, HMatrix &permutation, HMatrix &unpermutation) {
        auto Htt = H.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        // auto test = H.get_block(32, 32, 320, 64);
        // std::cout << "!!!!!!!!!!!!!!!!!!!!!! appel avec t= " << t.get_size() << ',' << t.get_offset() << std::endl;
        // Matrix<CoefficientPrecision> dense(32, 32);
        // copy_to_dense(*test, dense.data());
        // std::cout << "!!!!!!!!!!!!!!!!!!!!!! bloc (32, 320) (32, 64)"
        //           << " dense ?" << test->is_dense() << ",low rank ? " << test->is_low_rank() << " de norme : " << normFrob(dense) << std::endl;

        if (Htt->is_dense()) {
            Matrix<CoefficientPrecision> Hdense = *Htt->get_dense_data();
            auto hdense                         = Hdense;
            std::vector<int> pivot(t.get_size(), 0.0);
            Matrix<CoefficientPrecision> l(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> u(t.get_size(), t.get_size());
            get_lu_factorisation(hdense, l, u, pivot);
            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            Matrix<CoefficientPrecision> inv_cyclmat(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> cyclmat(t.get_size(), t.get_size());

            for (int k = 0; k < t.get_size(); ++k) {
                inv_cyclmat(pivot[k] - 1, k) = 1.0;
                cyclmat(k, pivot[k] - 1)     = 1.0;
            }
            permutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(inv_cyclmat);
            permutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            unpermutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(cyclmat);
            unpermutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            // Matrix<CoefficientPrecision> ldense(t.get_size(), t.get_size());
            // copy_to_dense(*L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), ldense.data());
            // Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
            // copy_to_dense(*U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), udense.data());
            // std::cout << "erreur sur la feuille " << t.get_size() << ',' << t.get_offset() << ':' << normFrob(hdense - inv_cyclmat * ldense * udense) / normFrob(hdense) << std::endl;
            // std::cout << "erreur sur la feuille unperm" << t.get_size() << ',' << t.get_offset() << ':' << normFrob(hdense - ldense * udense) / normFrob(hdense) << std::endl;
        } else {
            // std::cout << "  appel récursif" << std::endl;
            auto &t_children = t.get_children();
            if (permutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_children().size() == 0) {
                for (auto &ti : t_children) {
                    for (auto &tj : t_children) {
                        auto child = permutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->add_child(ti.get(), tj.get());
                    }
                }
            }
            if (unpermutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_children().size() == 0) {
                for (auto &ti : t_children) {
                    for (auto &tj : t_children) {
                        auto child = unpermutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->add_child(ti.get(), tj.get());
                    }
                }
            }
            for (int i = 0; i < t_children.size(); ++i) {
                HMatrix_PLU(H, *t_children[i], L, U, permutation, unpermutation);

                for (int j = i + 1; j < t_children.size(); ++j) {

                    auto unt   = unpermutation.get_block(t_children[i]->get_size(), t_children[i]->get_size(), t_children[i]->get_offset(), t_children[i]->get_offset());
                    auto ht    = H.get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset());
                    auto Hperm = unt->hmatrix_product(*ht);

                    // FM_build(L, *t_children[i], *t_children[j], Hperm, U);

                    FM(L, *t_children[i], *t_children[j], Hperm, U);

                    /// ----------------> Ca nous donne U12
                    ///// La on écrit pas la bonne valeur dans L mais c'est celle dont on a besoin pour le moins du coup il faut permuter aprés -> quand on fera LU dessus
                    // Matrix<CoefficientPrecision> hdense(t_children[j]->get_size(), t_children[i]->get_size());
                    // copy_to_dense(*H.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset()), hdense.data());
                    // Matrix<CoefficientPrecision> uudense(t_children[i]->get_size(), t_children[i]->get_size());
                    // copy_to_dense(*U.get_block(t_children[i]->get_size(), t_children[i]->get_size(), t_children[i]->get_offset(), t_children[i]->get_offset()), uudense.data());

                    FM_T_build(U, *t_children[i], *t_children[j], H, L);

                    // // FM_T(U, *t_children[i], *t_children[j], H, L);

                    // Matrix<CoefficientPrecision> lldense(t_children[j]->get_size(), t_children[i]->get_size());
                    // copy_to_dense(*L.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset()), lldense.data());
                    // std::cout << "      erreur backward : " << normFrob(lldense * uudense - hdense) / normFrob(hdense) << std::endl;
                    // copy_to_dense(*ht, htemp.data());
                    // std::cout << "                                                      norme de Z après backward : " << normFrob(htemp) << std::endl;
                    //// -----------------> Ca nous donne (PL)21 du coup il faut permuter dés qu'on a accés a la matrice de permutation

                    for (int r = i + 1; r < t_children.size(); ++r) {
                        auto Htemp = H.get_block(t_children[j]->get_size(), t_children[r]->get_size(), t_children[j]->get_offset(), t_children[r]->get_offset());
                        auto Ltemp = L.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset());
                        auto Utemp = U.get_block(t_children[i]->get_size(), t_children[r]->get_size(), t_children[i]->get_offset(), t_children[r]->get_offset());
                        Matrix<CoefficientPrecision> hh(t_children[j]->get_size(), t_children[r]->get_size());
                        Matrix<CoefficientPrecision> hmoins(t_children[j]->get_size(), t_children[r]->get_size());
                        // std::cout << "bonne taille pour le moins ? " << std::endl;

                        // std::cout << "H : " << Htemp->get_target_cluster().get_size() << ',' << Htemp->get_source_cluster().get_size() << "?= " << t_children[j]->get_size() << ',' << t_children[r]->get_size() << std::endl;
                        // std::cout << "L : " << Ltemp->get_target_cluster().get_size() << ',' << Ltemp->get_source_cluster().get_size() << "?=" << t_children[j]->get_size() << ',' << t_children[i]->get_size() << std::endl;
                        // std::cout << "U : " << Utemp->get_target_cluster().get_size() << ',' << Utemp->get_source_cluster().get_size() << "?= " << t_children[r]->get_size() << ',' << t_children[r]->get_size() << std::endl;
                        Matrix<CoefficientPrecision> ll(t_children[j]->get_size(), t_children[i]->get_size());
                        Matrix<CoefficientPrecision> uu(t_children[i]->get_size(), t_children[r]->get_size());
                        copy_to_dense(*Htemp, hh.data());
                        copy_to_dense(*Utemp, uu.data());
                        copy_to_dense(*Ltemp, ll.data());
                        Htemp->Moins(Ltemp->hmatrix_product(*Utemp));
                        // Ltemp->set_epsilon(H.m_tree_data->m_epsilon);
                        // Htemp->Moins(Ltemp->hmatrix_product_new(*Utemp));

                        // copy_to_dense(*Htemp, hmoins.data());
                        // std::cout << "                                  erreur MOINS :" << normFrob(hmoins - (hh - ll * uu)) << std::endl;
                    }
                }
                if (i > 0) {
                    auto local_perm = permutation.get_block(t_children[i]->get_size(), t_children[i]->get_size(), t_children[i]->get_offset(), t_children[i]->get_offset());
                    /// si on a plus de 2 fils par étage ca risque de tout cassé du coup on doit prendre tout les blocs a gauche de la diagonale ( et pas juste L21)
                    for (int k = 0; k < i; ++k) {
                        auto Lki = L.get_block(t_children[i]->get_size(), t_children[k]->get_size(), t_children[i]->get_offset(), t_children[k]->get_offset());

                        // Matrix<double> ldense(t_children[i]->get_size(), t_children[k]->get_size());
                        // Matrix<double> ldense2(t_children[i]->get_size(), t_children[k]->get_size());
                        // Matrix<double> permdense(t_children[i]->get_size(), t_children[i]->get_size());
                        // copy_to_dense(*Lki, ldense.data());
                        // std::cout << "local perm " << std::endl;
                        // std::cout << '(' << local_perm->get_target_cluster().get_size() << ',' << local_perm->get_source_cluster().get_size() << "),(" << local_perm->get_target_cluster().get_offset() << ',' << local_perm->get_source_cluster().get_offset() << ')' << std::endl;
                        // std::cout << "nb child" << local_perm->get_children().size() << std::endl;
                        // copy_to_dense(*local_perm, permdense.data());
                        // std::cout << "norme dense ? : " << normFrob(ldense) << ',' << normFrob(permdense) << std::endl;
                        local_perm->set_epsilon(H.m_tree_data->m_epsilon);
                        auto lperm = local_perm->hmatrix_product(*Lki);
                        Lki->assign(lperm, *t_children[i], *t_children[k]);
                        // copy_to_dense(*Lki, ldense2.data());
                        // std::cout << "ca a permuté ? :" << normFrob(ldense - ldense2) / normFrob(ldense) << std::endl;
                    }
                    //// Normalement la on permute le blocs en bas a gauche d'un bloc
                }
            }
        }
    }

    friend void HLU_noperm(HMatrix &H, const Cluster<CoordinatePrecision> &t, HMatrix &L, HMatrix &U) {
        auto Htt = H.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        // Si on est sur un feuille on fait le vrai LU
        if (Htt->is_dense()) {
            Matrix<CoefficientPrecision> Hdense = *Htt->get_dense_data();
            auto hdense                         = Hdense;
            std::vector<int> pivot(t.get_size(), 0.0);
            Matrix<CoefficientPrecision> l(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> u(t.get_size(), t.get_size());
            get_lu_factorisation(hdense, l, u, pivot);
            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
        } else {
            // On est sur un bloc diagonal qui a des fils, on peut descendre
            auto &t_children = t.get_children();
            for (int i = 0; i < t_children.size(); ++i) {
                HLU_noperm(H, *t_children[i], L, U);

                for (int j = i + 1; j < t_children.size(); ++j) {
                    //// On apopelle forward sur la matrice H permutée par la perm donné par l'appele de LU(ti)
                    FM(L, *t_children[i], *t_children[j], H, U);
                    /// ----------------> Ca nous donne U12
                    FM_T(U, *t_children[i], *t_children[j], H, L);
                    //// -----------------> Ca nous donne (PL)21 du coup il faut permuter dés qu'on a accés a la matrice de permutation

                    for (int r = i + 1; r < t_children.size(); ++r) {
                        auto Htemp = H.get_block(t_children[j]->get_size(), t_children[r]->get_size(), t_children[j]->get_offset(), t_children[r]->get_offset());
                        auto Ltemp = L.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset());
                        auto Utemp = U.get_block(t_children[i]->get_size(), t_children[r]->get_size(), t_children[i]->get_offset(), t_children[r]->get_offset());
                        // Htemp->Moins(Ltemp->hmatrix_product(*Utemp));
                        Htemp->Moins(Ltemp->hmatrix_product(*Utemp));
                    }
                }
            }
        }
    }

    friend void HLU(HMatrix &H, const Cluster<CoordinatePrecision> &t, HMatrix &L, HMatrix &U, HMatrix &permutation, HMatrix &unpermutation) {
        auto Htt = H.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        // Si on est sur un feuille on fait le vrai LU
        if (Htt->is_dense()) {
            Matrix<CoefficientPrecision> Hdense = *Htt->get_dense_data();
            auto hdense                         = Hdense;
            std::vector<int> pivot(t.get_size(), 0.0);
            Matrix<CoefficientPrecision> l(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> u(t.get_size(), t.get_size());
            get_lu_factorisation(hdense, l, u, pivot);
            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            Matrix<CoefficientPrecision> inv_cyclmat(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> cyclmat(t.get_size(), t.get_size());

            for (int k = 0; k < t.get_size(); ++k) {
                inv_cyclmat(pivot[k] - 1, k) = 1.0;
                cyclmat(k, pivot[k] - 1)     = 1.0;
            }
            permutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(inv_cyclmat);
            permutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            unpermutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(cyclmat);
            unpermutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            Matrix<CoefficientPrecision> ldense(t.get_size(), t.get_size());
            copy_to_dense(*L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), ldense.data());
            Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
            copy_to_dense(*U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), udense.data());
            std::cout << "  erreur sur la feuille " << t.get_size() << ',' << t.get_offset() << ':' << normFrob(hdense - inv_cyclmat * ldense * udense) / normFrob(hdense) << std::endl;
            std::cout << "  erreur sur la feuille unperm" << t.get_size() << ',' << t.get_offset() << ':' << normFrob(hdense - ldense * udense) / normFrob(hdense) << std::endl;
        } else {
            // On est sur un bloc diagonal qui a des fils, on peut descendre
            auto &t_children = t.get_children();
            if (permutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_children().size() == 0) {
                for (auto &ti : t_children) {
                    for (auto &tj : t_children) {
                        auto child = permutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->add_child(ti.get(), tj.get());
                    }
                }
            }
            if (unpermutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_children().size() == 0) {
                for (auto &ti : t_children) {
                    for (auto &tj : t_children) {
                        auto child = unpermutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->add_child(ti.get(), tj.get());
                    }
                }
            }
            for (int i = 0; i < t_children.size(); ++i) {
                HLU(H, *t_children[i], L, U, permutation, unpermutation);

                for (int j = i + 1; j < t_children.size(); ++j) {
                    //// On apopelle forward sur la matrice H permutée par la perm donné par l'appele de LU(ti)
                    Matrix<CoefficientPrecision> denseperm(t_children[i]->get_size(), t_children[i]->get_size());
                    auto unt = unpermutation.get_block(t_children[i]->get_size(), t_children[i]->get_size(), t_children[i]->get_offset(), t_children[i]->get_offset());
                    copy_to_dense(*unt, denseperm.data());
                    auto ht    = H.get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset());
                    auto Hperm = unt->hmatrix_product(*ht);
                    Matrix<CoefficientPrecision> hpermdense(t_children[i]->get_size(), t_children[j]->get_size());
                    copy_to_dense(Hperm, hpermdense.data());
                    Matrix<CoefficientPrecision>
                        ldense(t_children[i]->get_size(), t_children[i]->get_size());
                    copy_to_dense(*L.get_block(t_children[i]->get_size(), t_children[i]->get_size(), t_children[i]->get_offset(), t_children[i]->get_offset()), ldense.data());
                    Forward_Matrix(L, *t_children[i], *t_children[j], Hperm, U);
                    Matrix<CoefficientPrecision> udense(t_children[i]->get_size(), t_children[j]->get_size());
                    copy_to_dense(*U.get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset()), udense.data());
                    std::cout << "-----------------------------erreur forward : " << normFrob(hpermdense - ldense * udense) / normFrob(hpermdense) << std::endl;
                    // Forward_Matrix(L, *t_children[i], *t_children[j], H, U);
                    /// ----------------> Ca nous donne U12
                    ///// La on écrit pas la bonne valeur dans L mais c'est celle dont on a besoin pour le moins du coup il faut permuter aprés -> quand on fera LU dessus
                    Matrix<CoefficientPrecision> hdense(t_children[j]->get_size(), t_children[i]->get_size());
                    copy_to_dense(*H.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset()), hdense.data());
                    Matrix<CoefficientPrecision> uudense(t_children[i]->get_size(), t_children[i]->get_size());
                    // std::cout << "+++++++++++++++++++++++++++++++ avan backward le bloc test :"
                    //           << " dense ?" << test->is_dense() << ",low rank ? " << test->is_dense() << " de norme : " << normFrob(dense) << std::endl;
                    copy_to_dense(*U.get_block(t_children[i]->get_size(), t_children[i]->get_size(), t_children[i]->get_offset(), t_children[i]->get_offset()), uudense.data());
                    Forward_Matrix_T(U, *t_children[i], *t_children[j], H, L);
                    Matrix<CoefficientPrecision> lldense(t_children[j]->get_size(), t_children[i]->get_size());
                    copy_to_dense(*L.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset()), lldense.data());
                    std::cout << "--------------------------erreur backward : " << normFrob(lldense * uudense - hdense) / normFrob(hdense) << std::endl;
                    //// -----------------> Ca nous donne (PL)21 du coup il faut permuter dés qu'on a accés a la matrice de permutation

                    for (int r = i + 1; r < t_children.size(); ++r) {
                        auto Htemp = H.get_block(t_children[j]->get_size(), t_children[r]->get_size(), t_children[j]->get_offset(), t_children[r]->get_offset());
                        auto Ltemp = L.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset());
                        auto Utemp = U.get_block(t_children[i]->get_size(), t_children[r]->get_size(), t_children[i]->get_offset(), t_children[r]->get_offset());
                        Matrix<CoefficientPrecision> hh(t_children[j]->get_size(), t_children[r]->get_size());
                        Matrix<CoefficientPrecision> hmoins(t_children[j]->get_size(), t_children[r]->get_size());
                        // std::cout << "bonne taille pour le moins ? " << std::endl;

                        // std::cout << "H : " << Htemp->get_target_cluster().get_size() << ',' << Htemp->get_source_cluster().get_size() << "?= " << t_children[j]->get_size() << ',' << t_children[r]->get_size() << std::endl;
                        // std::cout << "L : " << Ltemp->get_target_cluster().get_size() << ',' << Ltemp->get_source_cluster().get_size() << "?=" << t_children[j]->get_size() << ',' << t_children[i]->get_size() << std::endl;
                        // std::cout << "U : " << Utemp->get_target_cluster().get_size() << ',' << Utemp->get_source_cluster().get_size() << "?= " << t_children[r]->get_size() << ',' << t_children[r]->get_size() << std::endl;
                        Matrix<CoefficientPrecision> ll(t_children[j]->get_size(), t_children[i]->get_size());
                        Matrix<CoefficientPrecision> uu(t_children[i]->get_size(), t_children[r]->get_size());
                        copy_to_dense(*Htemp, hh.data());
                        copy_to_dense(*Utemp, uu.data());
                        copy_to_dense(*Ltemp, ll.data());
                        // Htemp->Moins(Ltemp->hmatrix_product(*Utemp));
                        Htemp->Moins(Ltemp->hmatrix_product(*Utemp));

                        copy_to_dense(*Htemp, hmoins.data());
                        std::cout << "              erreur MOINS :" << normFrob(hmoins - (hh - ll * uu)) << std::endl;
                    }
                }
                if (i > 0) {
                    auto local_perm = permutation.get_block(t_children[i]->get_size(), t_children[i]->get_size(), t_children[i]->get_offset(), t_children[i]->get_offset());
                    /// si on a plus de 2 fils par étage ca risque de tout cassé du coup on doit prendre tout les blocs a gauche de la diagonale ( et pas juste L21)
                    for (int k = 0; k < i; ++k) {
                        auto Lki = L.get_block(t_children[i]->get_size(), t_children[k]->get_size(), t_children[i]->get_offset(), t_children[k]->get_offset());

                        Matrix<double> ldense(t_children[i]->get_size(), t_children[k]->get_size());
                        Matrix<double> ldense2(t_children[i]->get_size(), t_children[k]->get_size());
                        Matrix<double> permdense(t_children[i]->get_size(), t_children[i]->get_size());
                        copy_to_dense(*Lki, ldense.data());
                        std::cout << "local perm " << std::endl;
                        std::cout << '(' << local_perm->get_target_cluster().get_size() << ',' << local_perm->get_source_cluster().get_size() << "),(" << local_perm->get_target_cluster().get_offset() << ',' << local_perm->get_source_cluster().get_offset() << ')' << std::endl;
                        std::cout << "nb child" << local_perm->get_children().size() << std::endl;
                        copy_to_dense(*local_perm, permdense.data());
                        std::cout << "norme dense ? : " << normFrob(ldense) << ',' << normFrob(permdense) << std::endl;
                        auto lperm = local_perm->hmatrix_product(*Lki);
                        Lki->assign(lperm, *t_children[i], *t_children[k]);
                        copy_to_dense(*Lki, ldense2.data());
                        std::cout << "ca a permuté ? :" << normFrob(ldense - ldense2) / normFrob(ldense) << std::endl;
                    }
                    //// Normalement la on permute le blocs en bas a gauche d'un bloc
                }
            }
        }
    }

    friend void HMatrix_PLU_new(HMatrix &H, const Cluster<CoordinatePrecision> &t, HMatrix &L, HMatrix &U, HMatrix &permutation) {
        auto Htt = H.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (Htt->is_dense()) {
            // std::cout << "  on fait du dense : " << std::endl;
            Matrix<CoefficientPrecision> Hdense = *Htt->get_dense_data();
            auto hdense                         = Hdense;
            std::vector<int> pivot(t.get_size(), 0.0);
            Matrix<CoefficientPrecision> l(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> u(t.get_size(), t.get_size());
            get_lu_factorisation(hdense, l, u, pivot);
            // for (int k = 0; k < t.get_size(); ++k) {
            //     permutation[k + t.get_offset()] = pivot[k];
            // }
            // permutation.push_back(pivot);
            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            Matrix<CoefficientPrecision> cyclmat(t.get_size(), t.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                cyclmat(pivot[k] - 1, k) = 1.0;
            }
            permutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(cyclmat);
            permutation.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            auto lperm = cyclmat * l;
            // vrai_L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(lperm);
            // vrai_L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            Matrix<CoefficientPrecision> ldense(t.get_size(), t.get_size());
            copy_to_dense(*L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), ldense.data());
            Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
            copy_to_dense(*U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), udense.data());
            std::cout << "  erreur sur la feuille " << t.get_size() << ',' << t.get_offset() << ':' << normFrob(hdense - cyclmat * ldense * udense) / normFrob(hdense) << std::endl;
            //////// Si on est pas tout en haut il faut permuter le bloc L21 qui a été calculer juste avant avec Forward_M_T
            for (auto &leaf : L.get_leaves()) {
                if (leaf->get_target_cluster().get_offset() > leaf->get_source_cluster().get_offset()) {
                    Matrix<CoefficientPrecision> leafdense(t.get_size(), t.get_size());
                    Matrix<CoefficientPrecision> permdense(t.get_size(), t.get_size());
                    copy_to_dense(*leaf, leafdense.data());

                    auto perm_local = permutation.get_block(leaf->get_target_cluster().get_size(), leaf->get_target_cluster().get_size(), leaf->get_target_cluster().get_offset(), leaf->get_target_cluster().get_offset());
                    auto l_perm     = perm_local->hmatrix_product(*leaf);
                    Matrix<double> permat(t.get_size(), t.get_size());
                    copy_to_dense(l_perm, permat.data());
                    std::cout << "permutation l : " << normFrob(leafdense - l_perm) / normFrob(ldense) << std::endl;
                }
            }
        } else {
            // std::cout << "  appel récursif" << std::endl;
            auto &t_children = t.get_children();
            for (int i = 0; i < t_children.size(); ++i) {
                HMatrix_PLU_new(H, *t_children[i], L, U, permutation);
                for (int j = i + 1; j < t_children.size(); ++j) {
                    // Matrix<double> inv_perm_i(t_children[i]->get_size(), t_children[i]->get_size());
                    // Matrix<double> inv_perm_j(t_children[j]->get_size(), t_children[j]->get_size());
                    // for (int k = 0; k < t_children[i]->get_size(); ++k) {
                    //     inv_perm_i(k, permutation[k] - 1) = 1.0;
                    // }
                    //// On apopelle forward sur la pmatrice permutée par la perm donné par l'appele de LU(ti)
                    // auto Hperm = H.get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset());
                    // Hperm->permutation(inv_perm_i, *t_children[i]);
                    auto Hperm = permutation.get_block(t_children[i]->get_size(), t_children[i]->get_size(), t_children[i]->get_offset(), t_children[i]->get_offset())->hmatrix_product(*H.get_block(t_children[i]->get_size(), t_children[j]->get_size(), t_children[i]->get_offset(), t_children[j]->get_offset()));
                    Forward_Matrix(L, *t_children[i], *t_children[j], Hperm, U);

                    // Forward_Matrix(L, *t_children[i], *t_children[j], H, U);
                    /// ----------------> Ca nous donne U12
                    ///// La on écrit pas la bonne valeur dans L mais c'est celle dont on a besoin pour le moins du coup il faut permuter aprés -> quand on fera LU dessus
                    Forward_Matrix_T(U, *t_children[i], *t_children[j], H, L);
                    // vrai_L.assign(*L.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset()), *t_children[i], *t_children[j]);
                    //// -----------------> Ca nous donne (PL)21 du coup il faut permuter dés qu'on a accés a la matrice de permutation

                    for (int r = i + 1; r < t_children.size(); ++r) {
                        auto Htemp = H.get_block(t_children[j]->get_size(), t_children[r]->get_size(), t_children[j]->get_offset(), t_children[r]->get_offset());
                        auto Ltemp = L.get_block(t_children[j]->get_size(), t_children[i]->get_size(), t_children[j]->get_offset(), t_children[i]->get_offset());
                        auto Utemp = U.get_block(t_children[i]->get_size(), t_children[r]->get_size(), t_children[i]->get_offset(), t_children[r]->get_offset());
                        Htemp->Moins(Ltemp->hmatrix_product(*Utemp));
                    }
                }
            }
        }
    }
    /////////////////////////////////////////////////
    /////// HLU ----------------------> MOUAI , ca marche si il y a pas de permutation et qu'elles ont toutes le mêmes cluster tree (07/12)
    ///////////////////////////////////////////
    friend void
    hmatrix_LU(HMatrix &H, const Cluster<CoordinatePrecision> &t, HMatrix &L, HMatrix &U) {
        auto Htt = H.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        Matrix<double> hh(t.get_size(), t.get_size());
        copy_to_dense(*Htt, hh.data());
        std::cout << "appel de HLU que une matrice de norme : " << normFrob(hh) << std::endl;
        auto &H_children = Htt->get_children();
        if (Htt->is_dense()) {
            Matrix<CoefficientPrecision> Hdense = *Htt->get_dense_data();
            auto hdense                         = Hdense;
            std::vector<int> ipiv(t.get_size(), 0.0);
            int info = -1;
            int size = t.get_size();
            Lapack<CoefficientPrecision>::getrf(&size, &size, Hdense.data(), &size, ipiv.data(), &info);
            Matrix<CoefficientPrecision> l(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> u(t.get_size(), t.get_size());
            for (int i = 0; i < t.get_size(); ++i) {
                l(i, i) = 1;
                u(i, i) = Hdense(i, i);

                for (int j = 0; j < i; ++j) {
                    l(i, j) = Hdense(i, j);
                    u(j, i) = Hdense(j, i);
                }
            }
            std::vector<CoefficientPrecision> vrai_pivot(t.get_size());
            for (int k = 1; k < size + 1; ++k) {
                vrai_pivot[k - 1] = k;
            }
            for (int k = 0; k < size; ++k) {
                if (ipiv[k] - 1 != k) {
                    int temp                = vrai_pivot[k];
                    vrai_pivot[k]           = vrai_pivot[ipiv[k] - 1];
                    vrai_pivot[ipiv[k] - 1] = temp;
                }
            }

            Matrix<CoefficientPrecision> cyclmat(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> inv_cyclmat(t.get_size(), t.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                cyclmat(vrai_pivot[k] - 1, k)     = 1.0;
                inv_cyclmat(k, vrai_pivot[k] - 1) = 1.0;
            }
            Matrix<double> id(t.get_size(), t.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                id(k, k) = 1.0;
            }
            std::cout << "norm(permutaion-id)  " << normFrob(cyclmat - id) << std::endl;
            std::cout << "erreur invperm(M) -LU " << normFrob(inv_cyclmat * hdense - l * u) / normFrob(hdense);
            // l = cyclmat * l;

            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->delete_children();
            Matrix<CoefficientPrecision> ldense(t.get_size(), t.get_size());
            copy_to_dense(*L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), ldense.data());
            Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
            copy_to_dense(*U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), udense.data());
            std::cout << "erreur sur la feuille " << t.get_size() << ',' << t.get_offset() << ':' << normFrob(hdense - ldense * udense) / normFrob(hdense) << std::endl;

        } else {
            auto &t_children = t.get_children();
            for (int i = 0; i < t_children.size(); ++i) {
                auto &ti = t_children[i];
                hmatrix_LU(H, *ti, L, U);
                std::cout << "HLU ok avec ti = " << ti->get_size() << ',' << ti->get_offset() << std::endl;
                for (int j = i + 1; j < t_children.size(); ++j) {
                    auto &tj = t_children[j];
                    std::cout << "ti : " << ti->get_size() << ',' << ti->get_offset() << std::endl;
                    std::cout << "tj: " << tj->get_size() << ',' << tj->get_offset() << std::endl;
                    Matrix<CoefficientPrecision> h1(ti->get_size(), tj->get_size());
                    copy_to_dense(*H.get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset()), h1.data());
                    Matrix<CoefficientPrecision> l1(tj->get_size(), ti->get_size());
                    Matrix<CoefficientPrecision> u1(ti->get_size(), tj->get_size());

                    std::cout << "forward_M" << std::endl;
                    Forward_Matrix(L, *ti, *tj, H, U);
                    copy_to_dense(*U.get_block(ti->get_size(), tj->get_size(), ti->get_offset(), tj->get_offset()), u1.data());
                    std::cout
                        << "forward_M ok , U " << '(' << ti->get_size() << ',' << ti->get_offset() << ')' << ';' << '(' << tj->get_size() << ',' << tj->get_offset() << ')' << "de norme : " << normFrob(u1) << std::endl;
                    std::cout << "forward_M_T" << std::endl;

                    Forward_Matrix_T(U, *tj, *ti, *Htt, L);
                    copy_to_dense(*L.get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset()), l1.data());

                    std::cout << "forward_M_T ok , L" << '(' << tj->get_size() << ',' << tj->get_offset() << ')' << ';' << '(' << ti->get_size() << ',' << ti->get_offset() << ')' << "de norme : " << normFrob(l1) << std::endl;
                    for (int r = i + 1; r < t_children.size(); ++r) {
                        auto &tr   = t_children[r];
                        auto Htemp = H.get_block(tj->get_size(), tr->get_size(), tj->get_offset(), tr->get_offset());
                        auto Ltemp = L.get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
                        auto Utemp = U.get_block(ti->get_size(), tr->get_size(), ti->get_offset(), tr->get_offset());
                        Matrix<CoefficientPrecision> Hdense(tj->get_size(), tr->get_size());
                        copy_to_dense(*Htemp, Hdense.data());
                        Matrix<CoefficientPrecision> prod_dense(tj->get_size(), tr->get_size());
                        Matrix<CoefficientPrecision> ldense(tj->get_size(), ti->get_size());
                        Matrix<CoefficientPrecision> udense(ti->get_size(), tr->get_size());
                        copy_to_dense(*Utemp, udense.data());
                        copy_to_dense(*Ltemp, ldense.data());
                        std::cout << "nb chidren L et U : " << Ltemp->get_children().size() << ',' << Utemp->get_children().size() << std::endl;
                        std::cout << (Ltemp->is_dense()) << ',' << (Utemp->is_dense()) << std::endl;
                        std::cout << "facteur L et U de prod-----------------------------------------> : " << normFrob(ldense) << ',' << normFrob(udense) << std::endl;
                        if ((Ltemp->is_dense()) and (Utemp->is_dense())) {
                            Matrix<CoefficientPrecision> ldense(tj->get_size(), ti->get_size());
                            Matrix<CoefficientPrecision> udense(ti->get_size(), tr->get_size());
                            Matrix<CoefficientPrecision> hdense(tj->get_size(), tr->get_size());
                            copy_to_dense(*Htemp, hdense.data());
                            copy_to_dense(*Utemp, udense.data());
                            copy_to_dense(*Ltemp, ldense.data());
                            Htemp->set_dense_data(hdense - ldense * udense);
                        } else {
                            auto prod = Ltemp->hmatrix_product(*Utemp);
                            copy_to_dense(prod, prod_dense.data());

                            std::cout << "erreur produit " << normFrob(ldense * udense - prod_dense) << std::endl;
                            std::cout << "norme de H et prod : " << normFrob(Hdense) << '!' << normFrob(prod_dense) << std::endl;
                            auto test = Hdense - prod_dense;
                            Htemp->Moins(Ltemp->hmatrix_product(*Utemp));
                            copy_to_dense(*Htemp, Hdense.data());
                            std::cout << "aprés le moins : " << normFrob(Hdense) << std::endl;
                            std::cout << "erreur sur le moins ? " << normFrob(test - Hdense) / normFrob(test) << std::endl;
                        }
                    }
                }
            }
        }
    }
    ///////////////////////////////////////////
    /// HLU avec permutaion H =(PL)U
    /// on veut des mat trianguilaires du coup il faut permuter tout les L
    friend void
    hmatrix_PLU(HMatrix &H, const Cluster<CoordinatePrecision> &t, HMatrix &L, HMatrix &U, std::vector<int> &permutation) {
        auto Htt         = H.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        auto &H_children = Htt->get_children();
        if (H_children.size() == 0) {
            Matrix<CoefficientPrecision> Hdense = *Htt->get_dense_data();
            auto hdense                         = Hdense;
            std::vector<int> ipiv(t.get_size(), 0.0);
            int info = -1;
            int size = t.get_size();
            Lapack<CoefficientPrecision>::getrf(&size, &size, Hdense.data(), &size, ipiv.data(), &info);
            Matrix<CoefficientPrecision> l(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> u(t.get_size(), t.get_size());
            for (int i = 0; i < t.get_size(); ++i) {
                l(i, i) = 1;
                u(i, i) = Hdense(i, i);

                for (int j = 0; j < i; ++j) {
                    l(i, j) = Hdense(i, j);
                    u(j, i) = Hdense(j, i);
                }
            }
            std::vector<CoefficientPrecision> vrai_pivot(t.get_size());
            for (int k = 1; k < size + 1; ++k) {
                vrai_pivot[k - 1] = k;
            }
            for (int k = 0; k < size; ++k) {
                if (ipiv[k] - 1 != k) {
                    int temp                = vrai_pivot[k];
                    vrai_pivot[k]           = vrai_pivot[ipiv[k] - 1];
                    vrai_pivot[ipiv[k] - 1] = temp;
                }
            }
            for (int k = 0; k < t.get_size(); ++k) {
                permutation[k + t.get_offset()] = vrai_pivot[k];
            }
            // Matrix<CoefficientPrecision> cyclmat(t.get_size(), t.get_size());
            // for (int k = 0; k < t.get_size(); ++k) {
            //     cyclmat(vrai_pivot[k] - 1, k) = 1.0;
            // }
            // Matrix<double> id(t.get_size(), t.get_size());
            // for (int k = 0; k < t.get_size(); ++k) {
            //     id(k, k) = 1.0;
            // }
            // std::cout << "norm(permutaion-id)  " << normFrob(cyclmat - id) << std::endl;
            // l = cyclmat * l;
            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
            std::cout << "feuille ok" << std::endl;
            ////TEST SUR LA FEUILLE

            // std::cout << "erreu de la feuille permutée : " << normFrob(hdense - l * u) / normFrob(hdense) << std::endl;
        } else {
            auto &t_children = t.get_children();
            for (int i = 0; i < t_children.size(); ++i) {
                auto &ti = t_children[i];
                hmatrix_LU(H, *ti, L, U);
                for (int j = i + 1; j < t_children.size(); ++j) {
                    auto &tj = t_children[j];

                    std::cout << "M " << std::endl;

                    Forward_Matrix(L, *ti, *tj, H, U);
                    std::cout << "M ok " << std::endl;
                    std::cout << "M_T" << std::endl;
                    Forward_Matrix_T(U, *ti, *tj, H, L);
                    std::cout << "M_T ok" << std::endl;
                    for (int r = i + 1; r < t_children.size(); ++r) {
                        auto &tr   = t_children[r];
                        auto Htemp = H.get_block(tj->get_size(), tr->get_size(), tj->get_offset(), tr->get_offset());
                        auto Ltemp = L.get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
                        auto Utemp = U.get_block(ti->get_size(), tr->get_size(), ti->get_offset(), tr->get_offset());
                        Htemp->Moins(Ltemp->hmatrix_product(*Utemp));
                    }
                }
            }
        }
    }
    ///////////////////////////////////////
    friend void Forward_M(const HMatrix &L, HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &XX) {
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        if (Zts->get_children().size() == 0) {
            if (Zts->is_dense()) {
                auto Zdense = *Zts->get_dense_data();
                Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
                for (int j = 0; j < s.get_size(); ++j) {
                    std::vector<CoefficientPrecision> Xtj(L.get_source_cluster().get_size(), 0.0);
                    std::vector<CoefficientPrecision> Ztj(Z.get_target_cluster().get_size(), 0.0);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Ztj[k + t.get_offset()] = Zdense(k, j);
                    }
                    L.forward_substitution_extract(t, Xtj, Ztj);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Xupdate(k, j) = Xtj[k + t.get_offset()];
                    }
                }
                XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xupdate);
            } else {
                auto U = Zts->get_low_rank_data()->Get_U();
                auto V = Zts->get_low_rank_data()->Get_V();
                Matrix<CoefficientPrecision> Xu(t.get_size(), Zts->get_low_rank_data()->rank_of());
                for (int j = 0; j < Zts->get_low_rank_data()->rank_of(); ++j) {
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
                XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_low_rank_data(Xlr);
            }
        } else {
            for (int i = 0; i < t.get_children().size(); ++i) {
                auto &child_t = t.get_children()[i];
                for (auto &child_s : s.get_children()) {
                    Forward_M(L, Z, *child_t, *child_s, XX);
                    for (int j = i + 1; j < t.get_children().size(); ++j) {
                        auto &child_tt = t.get_children()[j];
                        auto Ztemp     = Z.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
                        auto Ltemp     = L.get_block(child_t->get_size(), child_tt->get_size(), child_t->get_offset(), child_tt->get_offset());
                        auto Xtemp     = XX.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
                        if (Ltemp == nullptr) {
                            auto ll = L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
                            Matrix<CoefficientPrecision> dense(t.get_size(), t.get_size());
                            copy_to_dense(*ll, dense.data());

                            Matrix<CoefficientPrecision> dense_l(child_t->get_size(), child_tt->get_size());
                            for (int k = 0; k < child_t->get_size(); ++k) {
                                for (int l = 0; l < child_tt->get_size(); ++l) {
                                    dense_l(k, l) = dense(k + child_t->get_offset(), l + child_tt->get_offset());
                                }
                            }
                            Matrix<CoefficientPrecision> dense_x(child_tt->get_size(), child_s->get_size());
                            copy_to_dense(*XX.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset()), dense_x.data());
                            auto z = *Ztemp->get_dense_data();
                            z      = z - dense_l * dense_x;
                            Ztemp->set_dense_data(z);
                        } else {
                            Ztemp->Moins(Ltemp->hmatrix_product(*Xtemp));
                        }
                    }
                }
            }
        }
    }
    /// END DELETE STAIT

    ////////////////////////////
    /// LX =Z
    /// TEST PASSED : YES

    friend void Forward_M_new(const HMatrix &L, HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &XX) {
        Matrix<CoefficientPrecision> zdense(Z.get_target_cluster().get_size(), Z.get_source_cluster().get_size());
        copy_to_dense(Z, zdense.data());
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        if (Zts->get_children().size() == 0) {
            if (Zts->is_dense()) {
                auto Xts = XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());

                auto Zdense = *Zts->get_dense_data();
                Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
                Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
                for (int j = 0; j < s.get_size(); ++j) {
                    std::vector<CoefficientPrecision> Xtj(L.get_source_cluster().get_size(), 0.0);
                    std::vector<CoefficientPrecision> Ztj(Z.get_target_cluster().get_size(), 0.0);
                    for (int k = 0; k < t.get_size(); ++k) {
                        Ztj[k + t.get_offset() - Z.get_target_cluster().get_offset()] = Zdense(k, j);
                    }
                    auto zz = Zdense.get_col(j);
                    // L.forward_substitution_extract_new(t, Xtj, Ztj);
                    L.forward_substitution(t, Xtj, Ztj, L.get_target_cluster().get_offset());

                    std::vector<CoefficientPrecision> xx;
                    for (int k = 0; k < t.get_size(); ++k) {
                        Xupdate(k, j) = Xtj[k + t.get_offset() - L.get_target_cluster().get_offset()];
                    }
                }
                XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xupdate);
            } else {
                auto U = Zts->get_low_rank_data()->Get_U();
                auto V = Zts->get_low_rank_data()->Get_V();
                Matrix<CoefficientPrecision> Xu(t.get_size(), Zts->get_low_rank_data()->rank_of());
                for (int j = 0; j < Zts->get_low_rank_data()->rank_of(); ++j) {
                    std::vector<CoefficientPrecision> Uj(Z.get_target_cluster().get_size());
                    std::vector<CoefficientPrecision> Aj(L.get_target_cluster().get_size());
                    for (int k = 0; k < t.get_size(); ++k) {
                        Uj[k + t.get_offset() - Z.get_target_cluster().get_offset()] = U(k, j);
                    }
                    // L.forward_substitution_extract_new(t, Aj, Uj);
                    L.forward_substitution(t, Aj, Uj, L.get_target_cluster().get_offset());

                    for (int k = 0; k < t.get_size(); ++k) {
                        Xu(k, j) = Aj[k + t.get_offset() - XX.get_target_cluster().get_offset()];
                    }
                }
                // Matrix<CoefficientPrecision> ldense(L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_target_cluster().get_size(), L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_source_cluster().get_size());
                // copy_to_dense(*L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), ldense.data());
                // Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
                // for (int k = 0; k < t.get_size(); ++k) {
                //     for (int l = 0; l < t.get_size(); ++l) {
                //         ll(k, l) = ldense(k + t.get_offset() - L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_target_cluster().get_offset(), l + t.get_offset() - L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_source_cluster().get_offset());
                //     }
                // }

                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Xu, V);
                XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_low_rank_data(Xlr);
                // std::cout << "on va push sur : " << t.get_size() << ',' << t.get_offset() << '/' << s.get_size() << ',' << s.get_offset() << std::endl;
                // std::cout << "lo<w rank : erreur sur ce qu'on a push : " << normFrob(ll * Xu * V - U * V) / normFrob(U * V) << std::endl;
            }
        } else {
            for (int i = 0; i < t.get_children().size(); ++i) {
                auto &child_t = t.get_children()[i];
                for (auto &child_s : s.get_children()) {
                    Forward_M_new(L, Z, *child_t, *child_s, XX);
                    for (int j = i + 1; j < t.get_children().size(); ++j) {
                        auto &child_tt = t.get_children()[j];
                        auto Ztemp     = Z.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
                        auto Ltemp     = L.get_block(child_tt->get_size(), child_t->get_size(), child_tt->get_offset(), child_t->get_offset());
                        auto Xtemp     = XX.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
                        if (!(Ltemp->get_target_cluster() == *child_tt) or !(Ltemp->get_source_cluster() == *child_t)) {
                            Matrix<CoefficientPrecision> dense(Ltemp->get_target_cluster().get_size(), Ltemp->get_source_cluster().get_size());
                            copy_to_dense(*Ltemp, dense.data());
                            // std::cout << "norme de dense : " << normFrob(dense) << std::endl;

                            Matrix<CoefficientPrecision> dense_l(child_tt->get_size(), child_t->get_size());
                            for (int k = 0; k < child_tt->get_size(); ++k) {
                                for (int l = 0; l < child_t->get_size(); ++l) {
                                    int ref       = dense(k + child_tt->get_offset() - Ltemp->get_target_cluster().get_offset(), l + child_t->get_offset() - Ltemp->get_source_cluster().get_offset());
                                    dense_l(k, l) = ref;
                                    // std::cout << dense(k + child_tt->get_offset() - Ltemp->get_target_cluster().get_offset(), l + child_t->get_offset() - Ltemp->get_source_cluster().get_offset()) << ',' << ref << ';';
                                    // std::cout << dense(k + child_tt->get_offset() - Ltemp->get_target_cluster().get_offset(), l + child_t->get_offset() - Ltemp->get_source_cluster().get_offset())<<',' <<    << std::endl;
                                }
                            }
                            // std::cout << std::endl;
                            // for (int k = 0; k < dense.nb_rows(); ++k) {
                            //     for (int l = 0; l < dense.nb_cols(); ++l) {
                            //         std::cout << dense(k, l) << ',';
                            //     }
                            //     std::cout << std::endl;
                            // }
                            Matrix<CoefficientPrecision> dense_x(child_t->get_size(), child_s->get_size());
                            auto temp = XX.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
                            copy_to_dense(*XX.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset()), dense_x.data());
                            Matrix<CoefficientPrecision> z(child_tt->get_size(), child_s->get_size());
                            copy_to_dense(*Ztemp, z.data());
                            // std::cout << "le z! " << normFrob(z) << std::endl;
                            // std::cout << normFrob(dense_l) << ',' << normFrob(dense_x) << std::endl;
                            z = z - dense_l * dense_x;
                            // std::cout << " le z? " << normFrob(z) << std::endl;
                            Ztemp->set_dense_data(z);
                        } else {
                            Ztemp->Moins(Ltemp->hmatrix_product(*Xtemp));
                        }
                    }
                }
            }
        }
    }
    //////////////////////
    // Pour résoudre  X U = Z
    // friend void Forward_M_T_new_new(const HMatrix &U, HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &XX) {
    //     // Matrix<CoefficientPrecision> zdense(Z.get_target_cluster().get_size(), Z.get_source_cluster().get_size());
    //     // copy_to_dense(Z, zdense.data());
    //     auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    //     if (Zts->get_children().size() == 0) {
    //         if (Zts->is_dense()) {
    //             auto Xts = XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());

    //             auto Zdense = *Zts->get_dense_data();
    //             Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
    //             Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
    //             for (int j = 0; j < s.get_size(); ++j) {
    //                 std::vector<CoefficientPrecision> Xtj(L.get_source_cluster().get_size(), 0.0);
    //                 // std::vector<CoefficientPrecision> Ztj(Z.get_target_cluster().get_size(), 0.0);
    //                 // for (int k = 0; k < t.get_size(); ++k) {
    //                 //     Ztj[k + t.get_offset() - Z.get_target_cluster().get_offset()] = Zdense(k, j);
    //                 // }
    //                 auto Ztj = Zdense.get_row(j);
    //                 // L.forward_substitution_extract_new(t, Xtj, Ztj);
    //                 U.forward_substitution_T(t, Xtj, Ztj, U.get_source_cluster().get_offset());

    //                 std::vector<CoefficientPrecision> xx;
    //                 for (int k = 0; k < t.get_size(); ++k) {
    //                     Xupdate(k, j) = Xtj[k + t.get_offset() - L.get_target_cluster().get_offset()];
    //                 }
    //             }
    //             XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xupdate);
    //         } else {
    //             auto U = Zts->get_low_rank_data()->Get_U();
    //             auto V = Zts->get_low_rank_data()->Get_V();
    //             Matrix<CoefficientPrecision> Xu(t.get_size(), Zts->get_low_rank_data()->rank_of());
    //             for (int j = 0; j < Zts->get_low_rank_data()->rank_of(); ++j) {
    //                 std::vector<CoefficientPrecision> Uj(Z.get_target_cluster().get_size());
    //                 std::vector<CoefficientPrecision> Aj(L.get_target_cluster().get_size());
    //                 for (int k = 0; k < t.get_size(); ++k) {
    //                     Uj[k + t.get_offset() - Z.get_target_cluster().get_offset()] = U(k, j);
    //                 }
    //                 // L.forward_substitution_extract_new(t, Aj, Uj);
    //                 L.forward_substitution(t, Aj, Uj, L.get_target_cluster().get_offset());

    //                 for (int k = 0; k < t.get_size(); ++k) {
    //                     Xu(k, j) = Aj[k + t.get_offset() - XX.get_target_cluster().get_offset()];
    //                 }
    //             }
    //             // Matrix<CoefficientPrecision> ldense(L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_target_cluster().get_size(), L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_source_cluster().get_size());
    //             // copy_to_dense(*L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), ldense.data());
    //             // Matrix<CoefficientPrecision> ll(t.get_size(), t.get_size());
    //             // for (int k = 0; k < t.get_size(); ++k) {
    //             //     for (int l = 0; l < t.get_size(); ++l) {
    //             //         ll(k, l) = ldense(k + t.get_offset() - L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_target_cluster().get_offset(), l + t.get_offset() - L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_source_cluster().get_offset());
    //             //     }
    //             // }

    //             LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Xu, V);
    //             XX.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_low_rank_data(Xlr);
    //             // std::cout << "on va push sur : " << t.get_size() << ',' << t.get_offset() << '/' << s.get_size() << ',' << s.get_offset() << std::endl;
    //             // std::cout << "lo<w rank : erreur sur ce qu'on a push : " << normFrob(ll * Xu * V - U * V) / normFrob(U * V) << std::endl;
    //         }
    //     } else {
    //         for (int i = 0; i < t.get_children().size(); ++i) {
    //             auto &child_t = t.get_children()[i];
    //             for (auto &child_s : s.get_children()) {
    //                 Forward_M_new(L, Z, *child_t, *child_s, XX);
    //                 for (int j = i + 1; j < t.get_children().size(); ++j) {
    //                     auto &child_tt = t.get_children()[j];
    //                     auto Ztemp     = Z.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
    //                     auto Ltemp     = L.get_block(child_tt->get_size(), child_t->get_size(), child_tt->get_offset(), child_t->get_offset());
    //                     auto Xtemp     = XX.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
    //                     if (!(Ltemp->get_target_cluster() == *child_tt) or !(Ltemp->get_source_cluster() == *child_t)) {
    //                         Matrix<CoefficientPrecision> dense(Ltemp->get_target_cluster().get_size(), Ltemp->get_source_cluster().get_size());
    //                         copy_to_dense(*Ltemp, dense.data());
    //                         // std::cout << "norme de dense : " << normFrob(dense) << std::endl;

    //                         Matrix<CoefficientPrecision> dense_l(child_tt->get_size(), child_t->get_size());
    //                         for (int k = 0; k < child_tt->get_size(); ++k) {
    //                             for (int l = 0; l < child_t->get_size(); ++l) {
    //                                 int ref       = dense(k + child_tt->get_offset() - Ltemp->get_target_cluster().get_offset(), l + child_t->get_offset() - Ltemp->get_source_cluster().get_offset());
    //                                 dense_l(k, l) = ref;
    //                                 // std::cout << dense(k + child_tt->get_offset() - Ltemp->get_target_cluster().get_offset(), l + child_t->get_offset() - Ltemp->get_source_cluster().get_offset()) << ',' << ref << ';';
    //                                 // std::cout << dense(k + child_tt->get_offset() - Ltemp->get_target_cluster().get_offset(), l + child_t->get_offset() - Ltemp->get_source_cluster().get_offset())<<',' <<    << std::endl;
    //                             }
    //                         }
    //                         // std::cout << std::endl;
    //                         // for (int k = 0; k < dense.nb_rows(); ++k) {
    //                         //     for (int l = 0; l < dense.nb_cols(); ++l) {
    //                         //         std::cout << dense(k, l) << ',';
    //                         //     }
    //                         //     std::cout << std::endl;
    //                         // }
    //                         Matrix<CoefficientPrecision> dense_x(child_t->get_size(), child_s->get_size());
    //                         auto temp = XX.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
    //                         copy_to_dense(*XX.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset()), dense_x.data());
    //                         Matrix<CoefficientPrecision> z(child_tt->get_size(), child_s->get_size());
    //                         copy_to_dense(*Ztemp, z.data());
    //                         // std::cout << "le z! " << normFrob(z) << std::endl;
    //                         // std::cout << normFrob(dense_l) << ',' << normFrob(dense_x) << std::endl;
    //                         z = z - dense_l * dense_x;
    //                         // std::cout << " le z? " << normFrob(z) << std::endl;
    //                         Ztemp->set_dense_data(z);
    //                     } else {
    //                         Ztemp->Moins(Ltemp->hmatrix_product(*Xtemp));
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    // friend void Forward_M_T_new(const HMatrix &U, HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &X) {
    //     auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offste());
    //     if ( Zts->get_children().size() >0){
    //         for (auto & child_t : t.get_children())
    //     }
    // }
    // RESOUDRE XU =Z ( on connait U et Z)
    // friend my_forward_M_T (const HMatrix &U , HMatrix &Z , const Cluster<CoordinatePrecision> &t , const Cluster<CoordinatePrecision> &s, HMatrix &X){
    //     // On prend le bloc Zts
    //     auto Zts= Z.get_block(t.get_size() , s.get_size() , t.get_offset() , s.get_offset()) ;
    //     // On regarde si c'est un feuille
    //     auto children= Zts->get_children() ;
    //     // SI C EST UNE FEUILLE
    //     if (children.size() == 0){
    //         if (Zts->is_dense()){
    //             auto zdense = Zts->get_dense_data() ;
    //             auto utemp  = U.extract(s.get_size(), s.get_size(), s.get_offset(), s.get_offset());

    //             //On appele froward_T sur les lignes
    //             for (int k = s.get)
    //         }

    //     }

    // }
    friend void Forward_M_T_new(const HMatrix &U, HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &X) {
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        if (Zts->get_children().size() > 0) {
            for (int i = 0; i < t.get_children().size(); ++i) {
                auto &child_t = t.get_children()[i];
                for (auto &child_s : s.get_children()) {
                    Forward_M_T_new(U, Z, *child_t, *child_s, X);
                    for (auto &child_tt : s.get_children()) {
                        auto Ztemp = Z.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
                        auto Utemp = U.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
                        auto Xtemp = X.get_block(child_t->get_size(), child_tt->get_size(), child_t->get_offset(), child_tt->get_offset());
                        if (child_tt->get_offset() < child_s->get_offset()) {
                            // if (!((Utemp->get_target_cluster() == *child_tt) and (Utemp->get_source_cluster() == *child_s)) or !((Ztemp->get_target_cluster() == *child_t) and (Ztemp->get_source_cluster() == *child_s))
                            //     or !((Xtemp->get_target_cluster() == *child_t) and (Xtemp->get_source_cluster() == *child_tt))) {
                            //     Matrix<CoefficientPrecision> dense_u(child_tt->get_size(), child_s->get_size());
                            //     Matrix<CoefficientPrecision> dense_x(child_t->get_size(), child_tt->get_size());
                            //     Matrix<CoefficientPrecision> z(child_t->get_size(), child_s->get_size());
                            //     copy_to_dense(*Utemp, dense_u.data());
                            //     copy_to_dense(*Ztemp, z.data());
                            //     copy_to_dense(*Xtemp, dense_x.data());
                            //     // auto dense_u = Utemp->extract(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
                            //     // auto dense_x = Xtemp->extract(child_t->get_size(), child_tt->get_size(), child_t->get_offset(), child_s->get_offset());
                            //     // auto z       = Ztemp->extract(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
                            //     z = z - dense_x * dense_u;
                            //     Ztemp->set_dense_data(z);
                            // } else {
                            Matrix<CoefficientPrecision> dense_u = Utemp->extract(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
                            Matrix<CoefficientPrecision> dense_x = Xtemp->extract(child_t->get_size(), child_tt->get_size(), child_t->get_offset(), child_tt->get_offset());
                            Matrix<CoefficientPrecision> z       = Ztemp->extract(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
                            // std::cout << "!!!!!" << std::endl;
                            // std::cout << "taille : " << z.nb_rows() << ',' << z.nb_cols() << "=" << dense_x.nb_rows() << ',' << dense_x.nb_cols() << "x" << dense_u.nb_rows() << ',' << dense_u.nb_cols() << std::endl;
                            // std::cout << "!!!!!!!!!!!!!" << std::endl;
                            auto tempp = dense_x * dense_u;
                            z          = z - dense_x * dense_u;
                            std::cout << "z ok :" << normFrob(z) << std::endl;
                            // if (Ztemp->is_dense()) {
                            Ztemp->set_dense_data(z);
                            // } else {
                            //     Matrix<CoefficientPrecision> Ztest(child_t->get_size(), child_s->get_size());
                            //     Ztemp->Moins(Xtemp->hmatrix_product(*Utemp));
                            // }
                            // copy_to_dense(*Ztemp, Ztest.data());
                            // std::cout << "erreur recursion " << normFrob(z - Ztest) / normFrob(Ztest) << std::endl;
                            // /Ztemp->set_dense_data(z);
                            // }
                        }
                    }
                }
            }
        } else {
            if (Zts->is_dense()) {
                auto Zdense = *Zts->get_dense_data();
                auto utemp  = U.extract(s.get_size(), s.get_size(), s.get_offset(), s.get_offset());
                Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
                for (int j = 0; j < t.get_size(); ++j) {
                    std::vector<CoefficientPrecision> Xtj(U.get_target_cluster().get_size(), 0.0);
                    std::vector<CoefficientPrecision> Ztj(Z.get_source_cluster().get_size(), 0.0);
                    for (int k = 0; k < s.get_size(); ++k) {
                        Ztj[k + s.get_offset() - Z.get_source_cluster().get_offset()] = Zdense(j, k);
                    }
                    auto temp = Zdense.get_row(j);
                    U.forward_substitution_T(s, Xtj, Ztj, U.get_target_cluster().get_offset());
                    std::vector<CoefficientPrecision> xx(s.get_size());
                    for (int k = 0; k < s.get_size(); ++k) {
                        xx[k] = Xtj[k + s.get_offset()];
                    }

                    for (int k = 0; k < s.get_size(); ++k) {
                        Xupdate(j, k) = xx[k];
                    }
                    // std::cout << "dans la boucle erreur " << norm2(utemp.transp(utemp) * Xupdate.get_row(j) - temp) / norm2(temp) << std::endl;
                }
                // std::cout << t.get_size() << ',' << t.get_offset() << '|' << s.get_size() << ',' << s.get_offset() << std::endl;
                // std::cout << "erreur toale sur le bloc calculé " << std::endl;

                // std::cout << normFrob(Xupdate * utemp - Zdense) / normFrob(Zdense) << std::endl;
                /// il faudrait caractére pour chrcher en normal ou en tranposé
                X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xupdate);
                // std::cout << "erreur toale sur le bloc push " << std::endl;
                // Matrix<CoefficientPrecision> xtest(t.get_size(), s.get_size());
                // copy_to_dense(*X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset()), xtest.data());
                // std::cout << normFrob(xtest * utemp - Zdense) / normFrob(Zdense) << std::endl;
            } else {

                auto u = Zts->get_low_rank_data()->Get_U();
                auto v = Zts->get_low_rank_data()->Get_V();
                Matrix<CoefficientPrecision> vX(Zts->get_low_rank_data()->rank_of(), s.get_size());
                for (int j = 0; j < Zts->get_low_rank_data()->rank_of(); ++j) {
                    std::vector<CoefficientPrecision> vj(Z.get_source_cluster().get_size());
                    std::vector<CoefficientPrecision> Aj(U.get_source_cluster().get_size());
                    for (int k = 0; k < s.get_size(); ++k) {
                        vj[k + s.get_offset() - Z.get_source_cluster().get_offset()] = v(j, k);
                    }
                    U.forward_substitution_T(s, Aj, vj, U.get_source_cluster().get_offset());
                    for (int k = 0; k < s.get_size(); ++k) {
                        vX(j, k) = Aj[k + s.get_offset() - X.get_source_cluster().get_offset()];
                    }
                }
                auto uu = U.extract(s.get_size(), s.get_size(), s.get_offset(), s.get_offset());

                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(u, vX);
                X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_low_rank_data(Xlr);
                // std::cout << " erreur avec low rank" << std::endl;
                // std::cout << normFrob(xtest * uu - Zdense) / normFrob(Zdense) << std::endl;
            }
        }
        // auto prod = X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->hmatrix_product(*U.get_block(s.get_size(), s.get_size(), s.get_offset(), s.get_offset()));
        // Matrix<CoefficientPrecision> test(t.get_size(), s.get_size());
        // copy_to_dense(prod, test.data());
        // std::cout << " alors forward T M ? -> erreur -> " << normFrob(test - ref) << std::endl;
    }
    /////////////////////////////
    //// DELETE STAIT : WAIT
    friend void
    Forward_M_T(const HMatrix &U, HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &X) {
        auto Zts = Z.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
        // std::cout << "forward_M_T avec t,s =" << t.get_size() << ',' << s.get_size() << ',' << t.get_offset() << ',' << s.get_offset() << std::endl;

        // if (Zts->get_children().size() > 0) {
        // if (!Zts->is_dense() and !Zts->is_low_rank()) {
        if (!Zts->is_leaf()) {
            // std::cout << " askip Zts hiérarchique" << std::endl;
            // std::cout << "Zts " << (Zts->get_dense_data() == nullptr) << ',' << (Zts->get_low_rank_data() == nullptr) << std::endl;
            for (int i = 0; i < t.get_children().size(); ++i) {
                auto &child_t = t.get_children()[i];
                for (auto &child_s : s.get_children()) {
                    Forward_M_T(U, Z, *child_t, *child_s, X);
                    for (auto &child_tt : s.get_children()) {
                        auto Ztemp = Z.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
                        auto Utemp = U.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
                        auto Xtemp = X.get_block(child_t->get_size(), child_tt->get_size(), child_t->get_offset(), child_tt->get_offset());
                        // std::cout << "Ztemp leaf ?" << Ztemp->is_leaf() << std::endl;
                        if (child_tt->get_offset() > child_s->get_offset()) {
                            // && Ztemp!=nullptr && Utemp != nullptr && Xtemp != nullptr
                            // std::cout << "est ce qu'il y a des nullptr " << (Ztemp == nullptr) << ',' << (Utemp == nullptr) << ',' << (Xtemp == nullptr) << std::endl;
                            if (Utemp == nullptr) {
                                auto uu = U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
                                Matrix<CoefficientPrecision> dense(t.get_size(), t.get_size());
                                copy_to_dense(*uu, dense.data());

                                Matrix<CoefficientPrecision> dense_u(child_tt->get_size(), child_s->get_size());
                                for (int k = 0; k < child_tt->get_size(); ++k) {
                                    for (int l = 0; l < child_s->get_size(); ++l) {
                                        dense_u(k, l) = dense(k + child_tt->get_offset(), l + child_s->get_offset());
                                    }
                                }
                                Matrix<CoefficientPrecision> dense_x(child_t->get_size(), child_tt->get_size());
                                copy_to_dense(*X.get_block(child_t->get_size(), child_tt->get_size(), child_t->get_offset(), child_tt->get_offset()), dense_x.data());
                                auto z = *Ztemp->get_dense_data();
                                z      = z - dense_x * dense_u;
                                Ztemp->set_dense_data(z);

                            } else {

                                Ztemp->Moins(Xtemp->hmatrix_product(*Utemp));
                            }
                        }
                    }
                }
            }
        } else {
            if (Zts->is_dense()) {
                auto Zdense = *Zts->get_dense_data();
                Matrix<CoefficientPrecision> Xupdate(t.get_size(), s.get_size());
                for (int j = 0; j < t.get_size(); ++j) {
                    std::vector<CoefficientPrecision> Xtj(X.get_source_cluster().get_size(), 0.0);
                    std::vector<CoefficientPrecision> Ztj(Z.get_source_cluster().get_size(), 0.0);
                    for (int k = 0; k < s.get_size(); ++k) {
                        Ztj[k + s.get_offset()] = Zdense(j, k);
                    }
                    // auto Ztj = Zdense.get_col(j);
                    U.forward_substitution_extract_T(s, Xtj, Ztj);
                    for (int k = 0; k < s.get_size(); ++k) {
                        Xupdate(j, k) = Xtj[k + s.get_offset()];
                    }
                }

                X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_dense_data(Xupdate);
            } else {
                auto u = Zts->get_low_rank_data()->Get_U();
                auto v = Zts->get_low_rank_data()->Get_V();
                Matrix<CoefficientPrecision> vX(Zts->get_low_rank_data()->rank_of(), s.get_size());
                for (int j = 0; j < Zts->get_low_rank_data()->rank_of(); ++j) {
                    std::vector<CoefficientPrecision> vj(Z.get_source_cluster().get_size());
                    std::vector<CoefficientPrecision> Aj(U.get_source_cluster().get_size());
                    for (int k = 0; k < s.get_size(); ++k) {
                        vj[k + s.get_offset()] = v(j, k);
                    }
                    U.forward_substitution_extract_T(s, Aj, vj);
                    for (int k = 0; k < s.get_size(); ++k) {
                        vX(j, k) = Aj[k + s.get_offset()];
                    }
                }

                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(u, vX);
                X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->set_low_rank_data(Xlr);
            }
        }
    }

    /// END DELETE STAIT

    /////////////////////////////
    // HLU
    //////////////////////

    // pas trés claire de quelle structure eswt sencé avoir L et U mais bon

    // void friend H_LU(HMatrix &L, HMatrix &U, HMatrix &A, const Cluster<CoordinatePrecision> &t) {
    //     std::cout << "HLU avec t = " << t.get_size() << ',' << t.get_offset() << std::endl;
    //     // Si on est sur une feuille -> vrai LU
    //     if (A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->is_dense()) {
    //         Matrix<CoefficientPrecision> LU = *A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_dense_data();
    //         auto lu                         = LU;
    //         std::vector<int> ipiv(t.get_size(), 0.0);
    //         int info = -1;
    //         int size = t.get_size();
    //         Lapack<CoefficientPrecision>::getrf(&size, &size, LU.data(), &size, ipiv.data(), &info);
    //         Matrix<CoefficientPrecision> l(t.get_size(), t.get_size());
    //         Matrix<CoefficientPrecision> u(t.get_size(), t.get_size());
    //         for (int i = 0; i < t.get_size(); ++i) {
    //             l(i, i) = 1;
    //             u(i, i) = LU(i, i);

    //             for (int j = 0; j < i; ++j) {
    //                 l(i, j) = LU(i, j);
    //                 u(j, i) = LU(j, i);
    //             }
    //         }
    //         std::vector<CoefficientPrecision> vrai_pivot(t.get_size());
    //         for (int k = 1; k < size + 1; ++k) {
    //             vrai_pivot[k - 1] = k;
    //         }
    //         for (int k = 0; k < size; ++k) {
    //             if (ipiv[k] - 1 != k) {
    //                 int temp                = vrai_pivot[k];
    //                 vrai_pivot[k]           = vrai_pivot[ipiv[k] - 1];
    //                 vrai_pivot[ipiv[k] - 1] = temp;
    //             }
    //         }
    //         Matrix<CoefficientPrecision> cyclmat(t.get_size(), t.get_size());
    //         for (int k = 0; k < t.get_size(); ++k) {
    //             cyclmat(vrai_pivot[k] - 1, k) = 1.0;
    //         }
    //         l = cyclmat * l;
    //         L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
    //         U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
    //         std::cout << "feuille ok" << std::endl;
    //         ////TEST SUR LA FEUILLE
    //         std::cout << "erreu de la feuille " << normFrob(lu - l * u) / normFrob(lu);
    //         std::cout << normFrob(LU - l * u) / normFrob(LU) << std::endl;
    //     } else {
    //         for (auto &child_t : t.get_children()) {
    //             H_LU(L, U, A, *child_t);
    //             for (auto &child_s : t.get_children()) {
    //                 if (child_s->get_offset() > child_t->get_offset()) {
    //                     std::cout << "on appel forward_M_T avec t,s = " << child_s->get_size() << ',' << child_t->get_size() << ',' << child_s->get_offset() << ',' << child_t->get_offset() << std::endl;
    //                     Forward_M_T(U, A, *child_s, *child_t, L);
    //                     std::cout << "forward_M_T ok " << std::endl;
    //                     Forward_M(L, A, *child_t, *child_s, U);
    //                     for (auto &child_r : t.get_children()) {
    //                         if (child_r->get_offset() > child_t->get_offset()) {

    //                             auto Atemp = A.get_block(child_s->get_size(), child_r->get_size(), child_s->get_offset(), child_r->get_offset());
    //                             auto Ltemp = L.get_block(child_s->get_size(), child_t->get_size(), child_s->get_offset(), child_t->get_offset());
    //                             auto Utemp = U.get_block(child_t->get_size(), child_r->get_size(), child_t->get_offset(), child_r->get_offset());
    //                             std::cout << "est ce qu'il y a des nullptr ?" << (Atemp == nullptr) << ',' << (Utemp == nullptr) << ',' << (Ltemp == nullptr) << std::endl;
    //                             Atemp->Moins(Ltemp->hmatrix_product(*Utemp));
    //                             std::cout << "moins ok " << std::endl;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    void friend H_LU(HMatrix &A, const Cluster<CoordinatePrecision> &t, HMatrix &Lh, HMatrix &Uh) {
        auto &sons = A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_children();
        // SI LE BLOC EST UNE FEUILLE -> VRAI LU
        if (sons.size() == 0) {
            // std::cout << "begin dgtrf" << std::endl;
            int size = t.get_size();
            // Matrix<CoefficientPrecision> a(size, size);
            // Matrix<CoefficientPrecision> at(size, size);
            // copy_to_dense(*A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), a.data());
            // copy_to_dense(*A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset()), at.data());
            auto a  = *A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_dense_data();
            auto at = *A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_dense_data();
            std::vector<int> ipiv(size, 0.0);
            int info = -1;
            Lapack<CoefficientPrecision>::getrf(&size, &size, a.data(), &size, ipiv.data(), &info);
            // std::cout << "info : " << info << std::endl;
            // int lwork = size * size;
            // std::vector<CoefficientPrecision> work(lwork);
            // // Effectue l'inversion
            // Matrix<CoefficientPrecision> id(size, size);
            // for (int k = 0; k < size; ++k) {
            //     id(k, k) = 1.0;
            // }
            // info = -1;
            // Lapack<double>::getri(&size, a.data(), &size, ipiv.data(), work.data(), &lwork, &info);
            // std::cout << "inversion ok ? info :" << info << "erreur " << normFrob(a * at - id) / size << std::endl;

            // std::cout << "degetrf ok" << std::endl;
            Matrix<CoefficientPrecision> L(size, size);
            Matrix<CoefficientPrecision> U(size, size);
            for (int k = 0; k < size; ++k) {
                for (int l = 0; l < size; ++l) {
                    // std::cout << a(k, l) << "   ,    ";
                    if (k == l) {
                        L(k, l) = 1;
                        U(k, l) = a(k, l);
                    } else if (k < l) {
                        U(k, l) = a(k, l);
                    } else {
                        L(k, l) = a(k, l);
                    }
                }
            }
            std::vector<CoefficientPrecision> vrai_pivot(t.get_size());
            for (int k = 1; k < size + 1; ++k) {
                vrai_pivot[k - 1] = k;
            }
            for (int k = 0; k < size; ++k) {
                if (ipiv[k] - 1 != k) {
                    int temp                = vrai_pivot[k];
                    vrai_pivot[k]           = vrai_pivot[ipiv[k] - 1];
                    vrai_pivot[ipiv[k] - 1] = temp;
                }
            }
            Matrix<CoefficientPrecision> cyclmat(t.get_size(), t.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                cyclmat(vrai_pivot[k] - 1, k) = 1.0;
            }
            L = cyclmat * L;
            Lh.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(L);
            Uh.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(U);
            std::cout << "feuille ok" << std::endl;
            ////TEST SUR LA FEUILLE
            std::cout << "erreur de la feuille " << normFrob(at - L * U) / normFrob(at) << std::endl;
            ;
            // SINON ON DECEND
        } else {
            auto &t_child = t.get_children();
            for (int i = 0; i < t_child.size(); ++i) {
                auto &ti = t_child[i];
                // std::cout << "appel recursif " << std::endl;
                H_LU(A, *ti, Lh, Uh);
                for (int j = i + 1; j < t_child.size(); ++j) {
                    auto &tj = t_child[j];
                    // std::cout << "forward_M_T ?" << std::endl;
                    //// TEST POUR FOWARD M8T
                    Matrix<CoefficientPrecision> aa(tj->get_size(), ti->get_size());
                    Matrix<CoefficientPrecision> ll(ti->get_size(), tj->get_size());
                    Matrix<CoefficientPrecision> uu(ti->get_size(), ti->get_size());

                    copy_to_dense(*A.get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset()), aa.data());
                    copy_to_dense(*Uh.get_block(ti->get_size(), ti->get_size(), ti->get_offset(), ti->get_offset()), aa.data());

                    // auto utemp = Uh.get_block(ti->get_size(), ti->get_size(), ti->get_offset(), ti->get_offset());
                    // auto atemp = A.get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
                    // auto ltemp = Lh.get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
                    Forward_M_T_new(Uh, A, *tj, *ti, Lh);
                    copy_to_dense(*Lh.get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset()), ll.data());

                    std::cout << "error forward_mt:" << std::endl;
                    std::cout << normFrob(ll * uu - aa) / normFrob(aa) << std::endl;
                    // std::cout << "forward_M_T ok" << std::endl;
                    // std::cout << "forward_M ?" << std::endl;
                    Forward_M_new(Lh, A, *ti, *tj, Uh);
                    // std::cout << "forward_M ok  ?" << std::endl;
                    for (int r = i + 1; r < t_child.size(); ++r) {
                        auto &tr   = t_child[r];
                        auto Atemp = A.get_block(tj->get_size(), tr->get_size(), tj->get_offset(), tr->get_offset());
                        auto Utemp = Uh.get_block(ti->get_size(), tr->get_size(), ti->get_offset(), tr->get_offset());
                        auto Ltemp = Lh.get_block(tj->get_size(), ti->get_size(), tj->get_offset(), ti->get_offset());
                        // Matrix<CoefficientPrecision> atemp(tj->get_size(), tr->get_size());
                        // Matrix<CoefficientPrecision> utemp(ti->get_size(), tr->get_size());
                        // Matrix<CoefficientPrecision> ltemp(tj->get_size(), ti->get_size());
                        Atemp->Moins(Ltemp->hmatrix_product(*Utemp));
                        // std::cout << "moins ok" << std::endl;
                    }
                }
            }
        }
    }
    void friend H_LU_new(HMatrix &L, HMatrix &U, HMatrix &A, const Cluster<CoordinatePrecision> &t) {
        std::cout << "HLU avec t = " << t.get_size() << ',' << t.get_offset() << std::endl;
        auto Atemp = A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        if (!((Atemp->get_target_cluster() == t) and (Atemp->get_source_cluster() == t))) {
            Matrix<CoefficientPrecision> adense(Atemp->get_target_cluster().get_size(), Atemp->get_source_cluster().get_size());
            Matrix<CoefficientPrecision> arestr(t.get_size(), t.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                for (int l = 0; l < t.get_size(); ++l) {
                    arestr(k, l) = adense(k + t.get_offset() - Atemp->get_target_cluster().get_offset(), l + t.get_offset() - Atemp->get_source_cluster().get_offset());
                }
            }
            std::vector<int> ipiv(t.get_size(), 0.0);
            int info = -1;
            int size = t.get_size();
            Lapack<CoefficientPrecision>::getrf(&size, &size, arestr.data(), &size, ipiv.data(), &info);
            Matrix<CoefficientPrecision> l(t.get_size(), t.get_size());
            Matrix<CoefficientPrecision> u(t.get_size(), t.get_size());
            for (int i = 0; i < t.get_size(); ++i) {
                l(i, i) = 1;
                u(i, i) = arestr(i, i);

                for (int j = 0; j < i; ++j) {
                    l(i, j) = arestr(i, j);
                    u(j, i) = arestr(j, i);
                }
            }
            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
        } else {
            if (A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_children().size() == 0) {
                // normalement t'es forcemment dense
                if (A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->is_dense()) {
                    Matrix<CoefficientPrecision> LU = *A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_dense_data();
                    std::vector<int> ipiv(t.get_size(), 0.0);
                    int info = -1;
                    int size = t.get_size();
                    Lapack<CoefficientPrecision>::getrf(&size, &size, LU.data(), &size, ipiv.data(), &info);
                    Matrix<CoefficientPrecision> l(t.get_size(), t.get_size());
                    Matrix<CoefficientPrecision> u(t.get_size(), t.get_size());
                    for (int i = 0; i < t.get_size(); ++i) {
                        l(i, i) = 1;
                        u(i, i) = LU(i, i);

                        for (int j = 0; j < i; ++j) {
                            l(i, j) = LU(i, j);
                            u(j, i) = LU(j, i);
                        }
                    }
                    L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
                    U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
                    std::cout << "feuille ok" << std::endl;
                } else {
                    std::cout << "?! la c'est pas normal " << std::endl;
                }
            }

            else {
                for (auto &child_t : t.get_children()) {
                    H_LU_new(L, U, A, *child_t);
                    for (auto &child_s : t.get_children()) {
                        if (child_s->get_offset() < child_t->get_offset()) {
                            Forward_M_new(L, A, *child_t, *child_s, U);
                            Forward_M_T_new(U, A, *child_s, *child_t, L);

                            for (auto &child_r : t.get_children()) {
                                if (child_r->get_offset() < child_t->get_offset()) {

                                    auto Atemp = A.get_block(child_s->get_size(), child_r->get_size(), child_s->get_offset(), child_r->get_offset());
                                    auto Ltemp = L.get_block(child_s->get_size(), child_t->get_size(), child_s->get_offset(), child_t->get_offset());
                                    auto Utemp = U.get_block(child_t->get_size(), child_r->get_size(), child_t->get_offset(), child_r->get_offset());
                                    std::cout << "est ce qu'on a essayé de descendre trop bas ? " << std::endl;
                                    std::cout << "Atemp :  " << (Atemp->get_target_cluster() == *child_s) << ',' << (Atemp->get_source_cluster() == *child_r) << std::endl;
                                    std::cout << "Ltemp : " << (Ltemp->get_target_cluster() == *child_s) << ',' << (Ltemp->get_source_cluster() == *child_t) << std::endl;
                                    std::cout << "Utemp : " << (Utemp->get_target_cluster() == *child_t) << ',' << (Utemp->get_source_cluster() == *child_r) << std::endl;
                                    Atemp->Moins(Ltemp->hmatrix_product(*Utemp));
                                    std::cout << "moins ok " << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    void friend my_hlu(HMatrix &A, HMatrix &L, HMatrix &U, const Cluster<CoordinatePrecision> &t) {
        auto At = A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset());
        std::cout << " hlu sur :  " << t.get_size() << ',' << t.get_offset() << std::endl;
        if (At->get_children().size() == 0) {
            std::cout << "yo " << std::endl;
            Matrix<CoefficientPrecision> mat(t.get_size(), t.get_size());
            copy_to_dense(*At, mat.data());
            int info = -1;
            int size = mat.nb_rows();
            std::vector<int> ipiv(size, 0.0);

            Lapack<CoefficientPrecision>::getrf(&size, &size, mat.data(), &size, ipiv.data(), &info);
            Matrix<CoefficientPrecision> l(size, size);
            Matrix<CoefficientPrecision> u(size, size);
            std::cout << "info : " << info << std::endl;

            for (int i = 0; i < size; ++i) {
                l(i, i) = 1;
                u(i, i) = mat(i, i);

                for (int j = 0; j < i; ++j) {
                    l(i, j) = mat(i, j);
                    u(j, i) = mat(j, i);
                }
            }
            L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
            U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
            std::cout << "on a push surla feuille " << t.get_size() << ',' << t.get_offset() << std::endl;
        } else {
            std::cout << "yu" << std::endl;
            auto &t1 = *t.get_children()[0];
            auto &t2 = *t.get_children()[1];
            std::cout << "t1/t2 :" << t1.get_size() << ',' << t1.get_offset() << '/' << t2.get_size() << ',' << t2.get_offset() << std::endl;
            std::cout << "t" << t.get_size() << ',' << t.get_offset() << std::endl;

            // on vérifie que L et U ont bien des fils sinon on le fait en dense
            auto L21 = L.get_block(t2.get_size(), t1.get_size(), t2.get_offset(), t1.get_offset());
            auto U12 = U.get_block(t1.get_size(), t2.get_size(), t1.get_offset(), t2.get_offset());
            if ((U12->get_target_cluster() == t1) and (U12->get_source_cluster() == t2) and (L21->get_target_cluster() == t2) and (L21->get_source_cluster() == t1)) {
                my_hlu(A, L, U, t1); // l11 et u11

                std::cout << "hlu sur " << t1.get_size() << ',' << t1.get_offset() << "ok" << std::endl;
                std::cout << "on est sur le bloc " << t.get_size() << ',' << t.get_offset() << std::endl;
                auto A21 = A.get_block(t2.get_size(), t1.get_size(), t2.get_offset(), t1.get_offset());
                auto L11 = L.get_block(t1.get_size(), t1.get_size(), t1.get_offset(), t1.get_offset());
                auto U11 = U.get_block(t1.get_size(), t1.get_size(), t1.get_offset(), t1.get_offset());
                auto A12 = A.get_block(t1.get_size(), t2.get_size(), t1.get_offset(), t2.get_offset());
                Matrix<CoefficientPrecision> a12(t1.get_size(), t2.get_size());
                copy_to_dense(*A12, a12.data());
                Matrix<CoefficientPrecision> a21(t2.get_size(), t1.get_size());
                copy_to_dense(*A21, a21.data());
                std::cout << "appel de forward_M sur" << t1.get_size() << ',' << t1.get_offset() << '/' << t2.get_size() << ',' << t2.get_offset() << std::endl;
                std::cout << "_____________________________" << std::endl;
                Forward_M_new(*L11, *A12, t1, t2, *U12); // U12 grace a l11
                std::cout << "_____________________________" << std::endl;

                std::cout << "forward ok " << std::endl;
                // Forward_M_new(L, A, t1, t2, U);
                //  auto lu12 = L11->hmatrix_product(*U12);
                //   Matrix<CoefficientPrecision> lu12dense(t1.get_size(), t2.get_size());
                //   copy_to_dense(lu12, lu12dense.data());
                Matrix<CoefficientPrecision> ltemp(t1.get_size(), t1.get_size());
                copy_to_dense(*L11, ltemp.data());
                Matrix<CoefficientPrecision> utemp(t1.get_size(), t2.get_size());
                copy_to_dense(*U12, utemp.data());
                std::cout << "test sur L1 et U12" << std::endl;
                std::cout << L11->get_children().size() << ',' << L11->is_dense() << ',' << L11->is_low_rank() << ',' << L11->get_leaves().size() << std::endl;
                std::cout << U12->get_children().size() << ',' << U12->is_dense() << ',' << U12->is_low_rank() << ',' << U12->get_leaves().size() << std::endl;
                std::cout << "norme de X : " << normFrob(utemp) << std::endl;
                std::cout << "erreur forward_M :" << normFrob(ltemp * utemp - a12) / normFrob(a12) << std::endl;
                Forward_M_T_new(*U11, *A21, t2, t1, *L21); // L21 grace a u11
                auto A22 = A.get_block(t2.get_size(), t2.get_size(), t2.get_offset(), t2.get_offset());
                A22->Moins(L21->hmatrix_product(*U12));
                my_hlu(A, L, U, t2);
            } else {
                std::cout << "whaaaaaaaaaaaaaaaaaaaaaaaaaaaat" << std::endl;
                Matrix<CoefficientPrecision> mat(t.get_size(), t.get_size());
                copy_to_dense(*At, mat.data());
                int info = -1;
                int size = mat.nb_rows();
                std::vector<int> ipiv(size, 0.0);

                Lapack<CoefficientPrecision>::getrf(&size, &size, mat.data(), &size, ipiv.data(), &info);
                std::cout << "info : " << info << std::endl;
                Matrix<CoefficientPrecision> l(size, size);
                Matrix<CoefficientPrecision> u(size, size);
                for (int i = 0; i < size; ++i) {
                    l(i, i) = 1;
                    u(i, i) = mat(i, i);

                    for (int j = 0; j < i; ++j) {
                        l(i, j) = mat(i, j);
                        u(j, i) = mat(j, i);
                    }
                }
                L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
                U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
            }
        }
    }
    // void friend my_hlu(HMatrix *A, HMatrix *L, HMatrix *U) {
    //     if ((A->get_children().size() > 0) and (L->get_children().size() > 0) and (U->get_children().size() > 0)) {
    //         auto &t1 = *A->get_target_cluster().get_children()[0];
    //         auto &t2 = *A->get_target_cluster().get_children()[1];

    //         auto A11 = A->get_block(t1.get_size(), t1.get_size(), t1.get_offset(), t1.get_offset());
    //         auto A12 = A->get_block(t1.get_size(), t2.get_size(), t1.get_offset(), t2.get_offset());
    //         auto A21 = A->get_block(t2.get_size(), t1.get_size(), t2.get_offset(), t1.get_offset());
    //         auto A22 = A->get_block(t2.get_size(), t2.get_size(), t2.get_offset(), t2.get_offset());

    //         auto L11 = L->get_block(t1.get_size(), t1.get_size(), t1.get_offset(), t1.get_offset());
    //         auto L12 = L->get_block(t1.get_size(), t2.get_size(), t1.get_offset(), t2.get_offset());
    //         auto L21 = L->get_block(t2.get_size(), t1.get_size(), t2.get_offset(), t1.get_offset());
    //         auto L22 = L->get_block(t2.get_size(), t2.get_size(), t2.get_offset(), t2.get_offset());

    //         auto U11 = U->get_block(t1.get_size(), t1.get_size(), t1.get_offset(), t1.get_offset());
    //         auto U12 = U->get_block(t1.get_size(), t2.get_size(), t1.get_offset(), t2.get_offset());
    //         auto U21 = U->get_block(t2.get_size(), t1.get_size(), t2.get_offset(), t1.get_offset());
    //         auto U22 = U->get_block(t2.get_size(), t2.get_size(), t2.get_offset(), t2.get_offset());

    //         my_hlu(A11, L11, U11);
    //         Matrix<CoefficientPrecision> test(A11->get_target_cluster().get_size(), A11->get_source_cluster().get_size());
    //         copy_to_dense(*L11, test.data());
    //         std::cout << normFrob(test) << std::endl;  /// U11 ET L11
    //         Forward_M_new(*L11, *A12, t1, t2, *U12);   // U12 GRACE A L11
    //         Forward_M_T_new(*U11, *A21, t2, t1, *L21); // L21 GRACE A U11
    //         A22->Moins(L21->hmatrix_product(*U12));
    //         my_hlu(A22, L22, U22);
    //     } else {
    //         Matrix<CoefficientPrecision> mat(A->get_target_cluster().get_size(), A->get_source_cluster().get_size());
    //         copy_to_dense(*A, mat.data());
    //         int info = -1;
    //         int size = mat.nb_rows();
    //         std::vector<int> ipiv(size, 0.0);

    //         Lapack<CoefficientPrecision>::getrf(&size, &size, mat.data(), &size, ipiv.data(), &info);
    //         Matrix<CoefficientPrecision> l(size, size);
    //         Matrix<CoefficientPrecision> u(size, size);
    //         for (int i = 0; i < size; ++i) {
    //             l(i, i) = 1;
    //             u(i, i) = mat(i, i);

    //             for (int j = 0; j < i; ++j) {
    //                 l(i, j) = mat(i, j);
    //                 u(j, i) = mat(j, i);
    //             }
    //         }
    //         L->set_dense_data(l);
    //         U->set_dense_data(u);
    //     }
    // }

    void friend H_LU_daquin(HMatrix *A, HMatrix *L, HMatrix *U) {
        if ((A->get_children().size() > 0) and (L->get_children().size() > 0) and (U->get_children().size() > 0)) {
            auto A11 = A->get_children()[0].get();
            auto A12 = A->get_children()[1].get();
            auto A21 = A->get_children()[2].get();
            auto A22 = A->get_children()[3].get();
            if (A12->get_source_cluster().get_offset() < A21->get_source_cluster().get_offset()) {
                auto temp = A12;
                A12       = A21;
                A21       = temp;
            }
            std::cout << "l children" << L->get_children().size() << std::endl;
            auto L11 = L->get_children()[0].get();
            auto L12 = L->get_children()[1].get();
            auto L21 = L->get_children()[2].get();
            auto L22 = L->get_children()[3].get();
            if (L12->get_source_cluster().get_offset() < L21->get_source_cluster().get_offset()) {
                auto temp = L12;
                L12       = L21;
                L21       = temp;
            }
            auto U11 = U->get_children()[0].get();
            auto U12 = U->get_children()[1].get();
            auto U21 = U->get_children()[2].get();
            auto U22 = U->get_children()[3].get();
            if (U12->get_source_cluster().get_offset() < U21->get_source_cluster().get_offset()) {
                auto temp = U12;
                U12       = U21;
                U21       = temp;
            }

            H_LU_daquin(A11, L11, U11);
            H_LU_daquin(A12, L11, U12);
            A21->Moins(L21->hmatrix_product(*U11));
            H_LU_daquin(A21, L22, U21);
            A22->Moins(L21->hmatrix_product(*U12));
            H_LU_daquin(A22, L22, U22);
        } else {
            Matrix<CoefficientPrecision> mat(A->get_target_cluster().get_size(), A->get_source_cluster().get_size());
            copy_to_dense(*A, mat.data());
            int info = -1;
            int size = mat.nb_rows();
            std::vector<int> ipiv(size, 0.0);

            Lapack<CoefficientPrecision>::getrf(&size, &size, mat.data(), &size, ipiv.data(), &info);
            Matrix<CoefficientPrecision> l(size, size);
            Matrix<CoefficientPrecision> u(size, size);
            for (int i = 0; i < size; ++i) {
                l(i, i) = 1;
                u(i, i) = mat(i, i);

                for (int j = 0; j < i; ++j) {
                    l(i, j) = mat(i, j);
                    u(j, i) = mat(j, i);
                }
            }
            L->set_dense_data(l);
            U->set_dense_data(u);
        }
    }

    void friend H_LU_dense(HMatrix &L, HMatrix &U, HMatrix &A, const Cluster<CoordinatePrecision> &t, Matrix<CoefficientPrecision> Ldense, Matrix<CoefficientPrecision> Udense, Matrix<CoefficientPrecision> Adense) {
        std::cout << "HLU avec t = " << t.get_size() << ',' << t.get_offset() << std::endl;
        if (A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_children().size() == 0) {
            // normalement t'es forcemment dense
            if (A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->is_dense()) {
                // partie hiérarchique
                Matrix<CoefficientPrecision> LU = *A.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->get_dense_data();
                std::vector<int> ipiv(t.get_size(), 0.0);
                int info = -1;
                int size = t.get_size();
                Lapack<CoefficientPrecision>::getrf(&size, &size, LU.data(), &size, ipiv.data(), &info);
                Matrix<CoefficientPrecision> l(t.get_size(), t.get_size());
                Matrix<CoefficientPrecision> u(t.get_size(), t.get_size());
                for (int i = 0; i < t.get_size(); ++i) {
                    l(i, i) = 1;
                    u(i, i) = LU(i, i);

                    for (int j = 0; j < i; ++j) {
                        l(i, j) = LU(i, j);
                        u(j, i) = LU(j, i);
                    }
                }
                L.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(l);
                U.get_block(t.get_size(), t.get_size(), t.get_offset(), t.get_offset())->set_dense_data(u);
                // partie dense

                Matrix<CoefficientPrecision> a_restr(t.get_size(), t.get_size());
                for (int k = 0; k < t.get_size(); ++k) {
                    for (int l = 0; l < t.get_size(); ++l) {
                        a_restr(k, l) = Adense(k + t.get_offset(), l + t.get_offset());
                    }
                }
                int infodense = -1;
                std::vector<int> ipivdense(t.get_size(), 0.0);

                Lapack<CoefficientPrecision>::getrf(&size, &size, a_restr.data(), &size, ipivdense.data(), &infodense);
                Matrix<CoefficientPrecision> ldense(t.get_size(), t.get_size());
                Matrix<CoefficientPrecision> udense(t.get_size(), t.get_size());
                for (int i = 0; i < t.get_size(); ++i) {
                    ldense(i, i) = 1;
                    udense(i, i) = a_restr(i, i);
                    for (int j = 0; j < i; ++j) {
                        ldense(i, j) = a_restr(i, j);
                        udense(j, i) = a_restr(j, i);
                    }
                }
                std::cout << " comparaison" << std::endl;
                Matrix<CoefficientPrecision> atest(A.get_target_cluster().get_size(), A.get_source_cluster().get_size());
                Matrix<CoefficientPrecision> ltest(L.get_target_cluster().get_size(), L.get_source_cluster().get_size());
                Matrix<CoefficientPrecision> utest(U.get_target_cluster().get_size(), U.get_source_cluster().get_size());
                copy_to_dense(A, atest.data());
                copy_to_dense(L, ltest.data());
                copy_to_dense(U, utest.data());
                std::cout << "A : " << normFrob(atest - Adense) << ", L :" << normFrob(ltest - Ldense) << ", U: " << normFrob(utest - Udense) << std::endl;
                std::cout << "l : " << normFrob(ldense - l) << ", u : " << normFrob(udense - u) << std::endl;

            } else {
                std::cout << "?! la c'est pas normal " << std::endl;
            }
        }

        else {
            for (auto &child_t : t.get_children()) {
                H_LU_dense(L, U, A, *child_t, Adense, Ldense, Udense);
                for (auto &child_s : t.get_children()) {
                    if (child_s->get_offset() > child_t->get_offset()) {
                        Forward_M_T_new(U, A, *child_s, *child_t, L);
                        Forward_M_new(L, A, *child_t, *child_s, U);
                        for (auto &child_r : t.get_children()) {
                            if (child_r->get_offset() > child_t->get_offset()) {
                                // PARTIE HMAT
                                auto Atemp = A.get_block(child_s->get_size(), child_r->get_size(), child_s->get_offset(), child_r->get_offset());
                                auto Ltemp = L.get_block(child_s->get_size(), child_t->get_size(), child_s->get_offset(), child_t->get_offset());
                                auto Utemp = U.get_block(child_t->get_size(), child_r->get_size(), child_t->get_offset(), child_r->get_offset());
                                // std::cout << "est ce qu'on a essayé de descendre trop bas ? " << std::endl;
                                // std::cout << "Atemp :  " << (Atemp->get_target_cluster() == *child_s) << ',' << (Atemp->get_source_cluster() == *child_r) << std::endl;
                                // std::cout << "Ltemp : " << (Ltemp->get_target_cluster() == *child_s) << ',' << (Ltemp->get_source_cluster() == *child_t) << std::endl;
                                // std::cout << "Utemp : " << (Utemp->get_target_cluster() == *child_t) << ',' << (Utemp->get_source_cluster() == *child_r) << std::endl;
                                Atemp->Moins(Ltemp->hmatrix_product(*Utemp));
                                std::cout << "moins ok " << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }

    friend void
    Forward_M_extract(const HMatrix &L, HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &X) {
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

            // std::cout << t.get_size() << ',' << t.get_offset() << ',' << s.get_size() << ',' << s.get_offset() << std::endl;
            // std::cout << "0 = ? " << normFrob(ll * Xupdate - Zdense) / normFrob(Zdense) << std::endl;
            // std::cout << "0 = ! " << normFrob(ll * *X.get_block(t.get_size(), s.get_size(), t.get_offset(), s.get_offset())->get_dense_data() - Zdense) / normFrob(Zdense) << std::endl;

        } else if (Zts->is_low_rank()) {
            // std::cout << "héhé " << std::endl;
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
                        // copy_to_dense(*Ltemp, ll.data());
                        // copy_to_dense(*Ztemp, zz.data());
                        // std::cout << "____________________" << std::endl;
                        // std::cout << normFrob(zz - ll * zz) << std::endl;
                        Ztemp->Moins(temp);
                        copy_to_dense(*Z.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset()), zz.data());
                        // std::cout << normFrob(zz) << std::endl;
                        // std::cout << "____________________" << std::endl;
                    }
                }
            }
        }
    }

    // friend void Forward_M(const HMatrix &L, HMatrix &Z, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, HMatrix &X) {
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
    //             L.forward_substitution(t, Xtj, Ztj);
    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 Xupdate(k, j) = Xtj[k + t.get_offset()];
    //             }
    //             std::vector<CoefficientPrecision> xrestr(s.get_size());
    //             auto zrestr = Zdense.get_col(j);

    //             for (int k = 0; k < s.get_size(); ++k) {
    //                 xrestr[k] = Xtj[k + s.get_offset()];
    //             }
    //         }

    //         MatrixGenerator<CoefficientPrecision> mat(Xupdate);
    //         Xts->compute_dense_data(mat);

    //         std::cout << t.get_size() << ',' << t.get_offset() << ',' << s.get_size() << ',' << s.get_offset() << std::endl;
    //         std::cout << "0 = ? " << normFrob(ll * Xupdate - Zdense) / normFrob(Zdense) << std::endl;

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
    //             L.forward_substitution(t, Aj, Uj);
    //             for (int k = 0; k < t.get_size(); ++k) {
    //                 Xu(k, j) = Aj[k + t.get_offset()];
    //             }
    //         }
    //         LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Xlr(Xu, V);
    //         Xts->set_low_rank_data(Xlr);
    //     } else {
    //         for (auto &child_t : t.get_children()) {
    //             for (auto &child_s : s.get_children()) {
    //                 Forward_M(L, Z, *child_t, *child_s, X);

    //                 for (auto &child_tt : t.get_children()) {
    //                     if (child_tt->get_offset() > child_t->get_offset()) {
    //                         auto Ztemp = Z.get_block(child_tt->get_size(), child_s->get_size(), child_tt->get_offset(), child_s->get_offset());
    //                         auto Ltemp = L.get_block(child_tt->get_size(), child_t->get_size(), child_tt->get_offset(), child_t->get_offset());

    //                         auto Xtemp = X.get_block(child_t->get_size(), child_s->get_size(), child_t->get_offset(), child_s->get_offset());
    //                         Matrix<CoefficientPrecision> ref(child_tt->get_size(), child_s->get_size());
    //                         auto temp = Ltemp->hmatrix_product(*Xtemp);

    //                         copy_to_dense(temp, ref.data());
    //                         Matrix<CoefficientPrecision> zz(Ztemp->get_target_cluster().get_size(), Ztemp->get_source_cluster().get_size());
    //                         copy_to_dense(*Ztemp, zz.data());
    //                         std::cout << "____________________" << std::endl;
    //                         std::cout << normFrob(zz - ref) << std::endl;
    //                         Ztemp->Moins(temp);
    //                         copy_to_dense(*Ztemp, zz.data());
    //                         std::cout << normFrob(zz) << std::endl;
    //                         std::cout << "____________________" << std::endl;
    //                         // *Ztemp = *Ztemp - Ltemp->hmatrix_product(*Xtemp);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

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

    // void friend H_LU(const HMatrix &M, HMatrix &L, HMatrix &U, const Cluster<CoordinatePrecision> &t) {
    //     auto Mt = M.get_block(t, t);
    //     if (Mt->is_leaf()) {
    //         auto m = Mt->get_dense_data();
    //         // l, u = lu(m) , L.get_block(t,t).set_danse_data(l)
    //     } else {
    //         for (auto &child_t : t.get_children()) {
    //             H_LU(M, L, U, child_t);
    //             for (auto &child_tt : t.get_children()) {
    //                 if (child_tt->get_offset() > child_t.get_offset()) {
    //                     ForwardT_M_extract(U, L, M, child_tt, child_t);
    //                     Forward_M_extract(L, U, M, child_t, child_tt);
    //                     for (auto &child : t.get_children()) {
    //                         if (child->get_offset() > child_t.get_offset()) {
    //                             auto Mts = M.get_block(child_tt, child);
    //                             auto Lts = L.get_block(child_tt, child_t);
    //                             auto Uts = U.get_block(child_t, child);
    //                             *Mts     = *Mts - Lts.hmatrix_product(Uts);
    //                         }
    //                     }
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
    /////////////////////////////
    ///// Hmatrix matrix
    Matrix<CoefficientPrecision> hmatrix_matrix(const Matrix<CoefficientPrecision> &U) const {
        // std::cout << "1" << '\n';
        Matrix<CoefficientPrecision> res(this->get_target_cluster().get_size(), U.nb_cols());

        if (this->get_source_cluster().get_size() == U.nb_rows()) {
            for (int k = 0; k < U.nb_cols(); ++k) {
                auto col_k = U.get_col(k);
                std::vector<CoefficientPrecision> res_col(this->get_target_cluster().get_size());
                this->add_vector_product('N', 1.0, col_k.data(), 1.0, res_col.data());
                res.set_col(k, res_col);
            }
        } else {
            std::cout << "woooooooooooooooo : problème  hmatrix matrix" << std::endl;
            std::cout << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << ',' << U.nb_rows() << ',' << U.nb_cols() << std::endl;
        }
        // std::cout << "1 ok" << '\n';

        return res;
    }

    Matrix<CoefficientPrecision> matrix_hmatrix(const Matrix<CoefficientPrecision> &U) const {
        Matrix<CoefficientPrecision> res(U.nb_rows(), this->get_source_cluster().get_size());

        if (this->get_target_cluster().get_size() == U.nb_cols()) {
            for (int k = 0; k < U.nb_rows(); ++k) {
                auto row_k = U.get_row(k);
                std::vector<CoefficientPrecision> res_row(this->get_source_cluster().get_size());
                this->add_vector_product('T', 1.0, row_k.data(), 1.0, res_row.data());
                res.set_row(k, res_row);
            }
        } else {
            std::cout << "woooooooooooooooo : problème  matrix hmatrix" << std::endl;
            std::cout << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << ',' << U.nb_rows() << ',' << U.nb_cols() << std::endl;
        }
        // std::cout << "1 ok" << '\n';

        return res;
    }

    // Matrix<CoefficientPrecision> matrix_hmatrix(const Matrix<CoefficientPrecision> &V) const {
    //     std::cout << "2 " << '\n';
    //     Matrix<CoefficientPrecision> res(V.nb_rows(), this->get_source_cluster().get_size());
    //     for (int k = 0; k < V.nb_rows(); ++k) {
    //         auto row_k = V.get_row(k);
    //         std::vector<CoefficientPrecision> res_row(this->get_source_cluster().get_size(), 0.0);
    //         std::cout << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << " fois transp  " << row_k.size() << " = " << res_row.size() << '\n';
    //         this->add_vector_product('T', 1.0, row_k.data(), 1.0, res_row.data());
    //         // for (int l = 0; l < this->get_source_cluster().get_size(); ++l) {
    //         //     res(k, l) = res_row[l];
    //         // }
    //         for (int l = 0; l < this->get_target_cluster().get_size(); ++l) {
    //             res(l, k) = res_row[l];
    //         }
    //     }
    //     std::cout << "2 ok " << '\n';
    //     return res;
    // }

    void dumb_vector_prod(const char *T, const std::vector<CoefficientPrecision> &x, std::vector<CoefficientPrecision> &y) {
        this->set_leaves_in_cache();
        for (auto &l : this->get_leaves()) {
            std::vector<CoefficientPrecision> xtemp(l->get_source_cluster().get_size());
            std::vector<CoefficientPrecision> ytemp(l->get_target_cluster().get_size());
            std::copy(x.begin() + l->get_source_cluster().get_offset(), x.begin() + l->get_source_cluster().get_size() + l->get_source_cluster().get_offset(), xtemp.begin());
            if (l->is_dense()) {
                auto m = *l->get_dense_data();
                ytemp  = m * xtemp;
            } else {
                ytemp = l->get_low_rank_data()->Get_U() * l->get_low_rank_data()->Get_V() * xtemp;
            }
            for (int k = 0; k < l->get_target_cluster().get_size(); ++k) {
                y[k + l->get_target_cluster().get_offset()] += ytemp[k];
            }
        }
    }
    // Matrix<CoefficientPrecision> matrix_hmatrix(const Matrix<CoefficientPrecision> &V) const {
    //     Matrix<CoefficientPrecision> res(V.nb_rows(), this->get_source_cluster().get_size());
    //     for (int k = 0; k < V.nb_rows(); ++k) {
    //         std::vector<CoefficientPrecision> row_k(V.nb_cols());
    //         for (int l = 0; l < V.nb_cols(); ++l) {
    //             row_k[l] = V(k, l);
    //         }
    //         std::vector<CoefficientPrecision> res_row(this->get_source_cluster().get_size(), 0.0);
    //         // std::cout << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << " fois transp  " << row_k.size() << " = " << res_row.size() << '\n';
    //         this->add_vector_product('T', 1.0, row_k.data(), 1.0, res_row.data());
    //         // for (int l = 0; l < this->get_source_cluster().get_size(); ++l) {
    //         //     res(k, l) = res_row[l];
    //         // }
    //         for (int l = 0; l < this->get_source_cluster().get_size(); ++l) {
    //             res(l, k) = res_row[l];
    //         }
    //     }
    //     // Matrix<CoefficientPrecision> hdense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
    //     // copy_to_dense(*this, hdense.data());
    //     // auto res = V * hdense;
    //     std::cout << "2 ok " << '\n';
    //     return res;
    // }
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

//////////////
// HMATRIX HMATRIX avec un chek de compression en plus
//////////////
template <typename CoefficientPrecision, typename CoordinatePrecision>
HMatrix<CoefficientPrecision, CoordinatePrecision> HMatrix<CoefficientPrecision, CoordinatePrecision>::hmatrix_product_compression(const HMatrix &B) const {
    HMatrix root_hmatrix(this->m_tree_data->m_target_cluster_tree, B.m_tree_data->m_source_cluster_tree, this->m_target_cluster, &B.get_source_cluster());
    root_hmatrix.set_admissibility_condition(this->m_tree_data->m_admissibility_condition);
    root_hmatrix.set_low_rank_generator(this->m_tree_data->m_low_rank_generator);
    root_hmatrix.set_eta(this->m_tree_data->m_eta);
    root_hmatrix.set_epsilon(this->m_tree_data->m_epsilon);
    root_hmatrix.set_maximal_block_size(this->m_tree_data->m_maxblocksize);
    root_hmatrix.set_minimal_target_depth(this->m_tree_data->m_minimal_target_depth);
    root_hmatrix.set_minimal_source_depth(this->m_tree_data->m_minimal_source_depth);

    SumExpression<CoefficientPrecision, CoordinatePrecision> root_sum_expression(this, &B);

    root_hmatrix.recursive_build_hmatrix_product_compression(root_sum_expression);

    return root_hmatrix;
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product_compression(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr) {
    auto &target_cluster  = this->get_target_cluster();
    auto &source_cluster  = this->get_source_cluster();
    auto &target_children = target_cluster.get_children();
    auto &source_children = source_cluster.get_children();
    bool admissible       = this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, this->m_tree_data->m_eta);
    auto test_restrict    = sum_expr.is_restrictible();
    bool is_restr         = sum_expr.is_restr();
    if (admissible) {
        // compute low rank approximation
        // std::cout << "________________________________" << std::endl;
        LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(sum_expr, *this->m_tree_data->m_low_rank_generator, this->get_target_cluster(), this->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
        // std::cout << "+++++++++++++++++++++++++++++++++++" << std::endl;
        if ((lr.Get_U().nb_rows() == 0) or (lr.Get_U().nb_cols() == 0) or (lr.Get_V().nb_rows() == 0) or (lr.Get_V().nb_cols() == 0)) {
            // this->compute_dense_data(sum_expr);
            if ((target_children.size() > 0) and (source_children.size() > 0)) {
                if ((test_restrict[0] == 0) and (test_restrict[1] == 0)) {
                    for (const auto &target_child : target_children) {
                        for (const auto &source_child : source_children) {
                            HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(target_child.get(), source_child.get());
                            // SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                            SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                            hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                        }
                    }
                } else {
                    this->compute_dense_data(sum_expr);
                }
            }

            else if ((target_children.size() == 0) and (source_children.size() > 0)) {
                if (test_restrict[1] == 0) {
                    for (const auto &source_child : source_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(&target_cluster, source_child.get());
                        SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_cluster.get_size(), target_cluster.get_offset(), source_child->get_size(), source_child->get_offset());
                        hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                    }
                } else {
                    // std::cout << "1" << std::endl;

                    this->compute_dense_data(sum_expr);
                }
            } else if ((source_children.size() == 0) and (target_children.size() > 0)) {
                if (test_restrict[0] == 0) {
                    for (const auto &target_child : target_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), &source_cluster);
                        SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_cluster.get_size(), source_cluster.get_offset());
                        hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                    }
                } else {

                    this->compute_dense_data(sum_expr);
                }
            } else {

                this->compute_dense_data(sum_expr);
            }

        } else {
            this->set_low_rank_data(lr);
        }
        // auto UV = sum_expr.ACA(-1,this->m_tree_data->m_epsilon);
        // if (UV[0].nb_cols() == 1 && UV[0].nb_rows() == 1) {
        //     std::cout << "ACA n'a pas trouvé d'approx du bloc admissible" << std::endl;
        //     this->compute_dense_data(sum_expr);

        // } else {
        //     LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(UV[0], UV[1]);
        //     this->set_low_rank_data(lr);
        // }

    } else if ((target_children.size() == 0) and (source_children.size() == 0)) {
        this->compute_dense_data(sum_expr);
    } else {
        if ((target_children.size() > 0) and (source_children.size() > 0)) {
            if ((test_restrict[0] == 0) and (test_restrict[1] == 0)) {
                for (const auto &target_child : target_children) {
                    for (const auto &source_child : source_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), source_child.get());
                        SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                        // SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_s(target_child->get_size(), source_child->get_size(), target_child->get_offset(), source_child->get_offset());
                        hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                    }
                }
            } else {
                // Il y a des trop gros blocs denses , normalemtn hmatrices =: minblocksize....... c'est a cause de ca qu'on est obligé de cjheker la restrictibilité a chaque fois....

                this->compute_dense_data(sum_expr);
            }
        }

        else if ((target_children.size() == 0) and (source_children.size() > 0)) {
            if (test_restrict[1] == 0) {
                for (const auto &source_child : source_children) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(&target_cluster, source_child.get());
                    SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_cluster.get_size(), target_cluster.get_offset(), source_child->get_size(), source_child->get_offset());
                    hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                }
            } else {
                this->compute_dense_data(sum_expr);
            }
        } else if ((source_children.size() == 0) and (target_children.size() > 0)) {
            if (test_restrict[0] == 0) {
                for (const auto &target_child : target_children) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), &source_cluster);
                    SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_cluster.get_size(), source_cluster.get_offset());
                    hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                }
            } else {
                this->compute_dense_data(sum_expr);
            }
        }
    }
}

////////////////////////
// HMATRIX HMATRIX
template <typename CoefficientPrecision, typename CoordinatePrecision>
HMatrix<CoefficientPrecision, CoordinatePrecision> HMatrix<CoefficientPrecision, CoordinatePrecision>::hmatrix_product(const HMatrix &B) const {
    HMatrix root_hmatrix(this->m_tree_data->m_target_cluster_tree, B.m_tree_data->m_source_cluster_tree, this->m_target_cluster, &B.get_source_cluster());
    root_hmatrix.set_admissibility_condition(this->m_tree_data->m_admissibility_condition);
    root_hmatrix.set_low_rank_generator(this->m_tree_data->m_low_rank_generator);
    root_hmatrix.set_eta(this->m_tree_data->m_eta);
    root_hmatrix.set_epsilon(this->m_tree_data->m_epsilon);
    root_hmatrix.set_maximal_block_size(this->m_tree_data->m_maxblocksize);
    root_hmatrix.set_minimal_target_depth(this->m_tree_data->m_minimal_target_depth);
    root_hmatrix.set_minimal_source_depth(this->m_tree_data->m_minimal_source_depth);

    SumExpression<CoefficientPrecision, CoordinatePrecision> root_sum_expression(this, &B);

    root_hmatrix.recursive_build_hmatrix_product(root_sum_expression);

    return root_hmatrix;
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr) {
    auto &target_cluster  = this->get_target_cluster();
    auto &source_cluster  = this->get_source_cluster();
    auto &target_children = target_cluster.get_children();
    auto &source_children = source_cluster.get_children();
    // if (sum_expr.get_sr().size() > 0) {
    //     std::cout << "ici   !!!!   " << sum_expr.get_sr()[0].nb_rows() << ',' << sum_expr.get_sr()[0].nb_cols() << ',' << sum_expr.get_sr()[1].nb_rows() << ',' << sum_expr.get_sr()[1].nb_cols() << std::endl;
    // }
    // critère pour descendre : on est sur une feuille ou pas:
    bool admissible    = this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, this->m_tree_data->m_eta);
    auto test_restrict = sum_expr.is_restrictible();
    if (admissible) {
        // this->compute_dense_data(sum_expr);
        //  this->compute_low_rank_data(sum_expr, *this->get_lr(), this->get_rk(), this->get_epsilon());
        // auto &temp = *this;
        // Je dois rajouter ca sinon ca plante avec forward_T_M...
        LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(sum_expr, *this->m_tree_data->m_low_rank_generator, this->get_target_cluster(), this->get_source_cluster(), -1, this->get_epsilon());
        if ((lr.Get_U().nb_rows() == 0) or (lr.Get_U().nb_cols() == 0) or (lr.Get_V().nb_rows() == 0) or (lr.Get_V().nb_cols() == 0)) {
            // this->compute_dense_data(sum_expr);
            if ((target_children.size() > 0) and (source_children.size() > 0)) {
                if ((test_restrict[0] == 0) and (test_restrict[1] == 0)) {
                    for (const auto &target_child : target_children) {
                        for (const auto &source_child : source_children) {
                            HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(target_child.get(), source_child.get());
                            // SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                            SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                            hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                        }
                    }
                } else {
                    this->compute_dense_data(sum_expr);
                    // Matrix<CoefficientPrecision> dense_data(target_cluster.get_size(), source_cluster.get_size());
                    // sum_expr.copy_submatrix(dense_data.nb_rows(), dense_data.nb_cols(), 0, 0, dense_data.data());
                    // this->split(dense_data);
                }
            }

            else if ((target_children.size() == 0) and (source_children.size() > 0)) {
                if (test_restrict[1] == 0) {
                    for (const auto &source_child : source_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(&target_cluster, source_child.get());
                        SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_cluster.get_size(), target_cluster.get_offset(), source_child->get_size(), source_child->get_offset());
                        hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                    }
                } else {
                    // std::cout << "1" << std::endl;

                    this->compute_dense_data(sum_expr);
                    // Matrix<CoefficientPrecision> dense_data(target_cluster.get_size(), source_cluster.get_size());
                    // sum_expr.copy_submatrix(dense_data.nb_rows(), dense_data.nb_cols(), 0, 0, dense_data.data());
                    // this->split(dense_data);
                }
            } else if ((source_children.size() == 0) and (target_children.size() > 0)) {
                if (test_restrict[0] == 0) {
                    for (const auto &target_child : target_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), &source_cluster);
                        SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_cluster.get_size(), source_cluster.get_offset());
                        hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                    }
                } else {

                    this->compute_dense_data(sum_expr);
                    // Matrix<CoefficientPrecision> dense_data(target_cluster.get_size(), source_cluster.get_size());
                    // sum_expr.copy_submatrix(dense_data.nb_rows(), dense_data.nb_cols(), 0, 0, dense_data.data());
                    // this->split(dense_data);
                }
            } else {

                this->compute_dense_data(sum_expr);
            }

        } else {

            this->set_low_rank_data(lr);
        }
        // auto UV = sum_expr.ACA(-1,this->m_tree_data->m_epsilon);
        // if (UV[0].nb_cols() == 1 && UV[0].nb_rows() == 1) {
        //     std::cout << "ACA n'a pas trouvé d'approx du bloc admissible" << std::endl;
        //     this->compute_dense_data(sum_expr);

        // } else {
        //     LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(UV[0], UV[1]);
        //     this->set_low_rank_data(lr);
        // }

    } else if ((target_children.size() == 0) and (source_children.size() == 0)) {
        this->compute_dense_data(sum_expr);
    } else {
        if ((target_children.size() > 0) and (source_children.size() > 0)) {
            if ((test_restrict[0] == 0) and (test_restrict[1] == 0)) {
                for (const auto &target_child : target_children) {
                    for (const auto &source_child : source_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), source_child.get());
                        SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                        // SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_s(target_child->get_size(), source_child->get_size(), target_child->get_offset(), source_child->get_offset());
                        hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                    }
                }
            } else {
                // Il y a des trop gros blocs denses , normalemtn hmatrices =: minblocksize....... c'est a cause de ca qu'on est obligé de cjheker la restrictibilité a chaque fois....

                this->compute_dense_data(sum_expr);
            }
        }

        else if ((target_children.size() == 0) and (source_children.size() > 0)) {
            if (test_restrict[1] == 0) {
                for (const auto &source_child : source_children) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(&target_cluster, source_child.get());
                    SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_cluster.get_size(), target_cluster.get_offset(), source_child->get_size(), source_child->get_offset());
                    hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                }
            } else {
                this->compute_dense_data(sum_expr);
            }
        } else if ((source_children.size() == 0) and (target_children.size() > 0)) {
            if (test_restrict[0] == 0) {
                for (const auto &target_child : target_children) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), &source_cluster);
                    SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_cluster.get_size(), source_cluster.get_offset());
                    hmatrix_child->recursive_build_hmatrix_product(sum_restr);
                }
            } else {
                this->compute_dense_data(sum_expr);
            }
        }
    }
}

////////////////////////
// HMATRIX HMATRIX avec possiblemtn des matrices triangulaires (sinon le test restrict fais que les blocs de 0 force a avoir des gros blocs dense (je comprend pas pourquoi quand j'assemble L au format hiérarchique la partie U a des gros blocs dense)
template <typename CoefficientPrecision, typename CoordinatePrecision>
HMatrix<CoefficientPrecision, CoordinatePrecision> HMatrix<CoefficientPrecision, CoordinatePrecision>::hmatrix_triangular_product(const HMatrix &B, const char L, const char U) const {
    HMatrix root_hmatrix(this->m_tree_data->m_target_cluster_tree, B.m_tree_data->m_source_cluster_tree, this->m_target_cluster, &B.get_source_cluster());
    root_hmatrix.set_admissibility_condition(this->m_tree_data->m_admissibility_condition);
    root_hmatrix.set_low_rank_generator(this->m_tree_data->m_low_rank_generator);
    root_hmatrix.set_eta(this->m_tree_data->m_eta);
    root_hmatrix.set_epsilon(this->m_tree_data->m_epsilon);
    root_hmatrix.set_maximal_block_size(this->m_tree_data->m_maxblocksize);
    root_hmatrix.set_minimal_target_depth(this->m_tree_data->m_minimal_target_depth);
    root_hmatrix.set_minimal_source_depth(this->m_tree_data->m_minimal_source_depth);

    SumExpression<CoefficientPrecision, CoordinatePrecision> root_sum_expression(this, &B);

    root_hmatrix.recursive_build_hmatrix_triangular_product(root_sum_expression, L, U);

    return root_hmatrix;
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_triangular_product(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr, const char L, const char U) {
    auto &target_cluster  = this->get_target_cluster();
    auto &source_cluster  = this->get_source_cluster();
    auto &target_children = target_cluster.get_children();
    auto &source_children = source_cluster.get_children();
    bool admissible       = this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, this->m_tree_data->m_eta);
    auto test_restrict    = sum_expr.is_restrictible();
    bool is_restr         = sum_expr.is_restr();
    if (admissible) {
        // compute low rank approximation
        // std::cout << "________________________________" << std::endl;
        LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(sum_expr, *this->m_tree_data->m_low_rank_generator, this->get_target_cluster(), this->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
        // std::cout << "+++++++++++++++++++++++++++++++++++" << std::endl;
        if ((lr.Get_U().nb_rows() == 0) or (lr.Get_U().nb_cols() == 0) or (lr.Get_V().nb_rows() == 0) or (lr.Get_V().nb_cols() == 0)) {
            // this->compute_dense_data(sum_expr);
            if ((target_children.size() > 0) and (source_children.size() > 0)) {
                if ((test_restrict[0] == 0) and (test_restrict[1] == 0)) {
                    for (const auto &target_child : target_children) {
                        for (const auto &source_child : source_children) {
                            HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(target_child.get(), source_child.get());
                            // SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                            // SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                            SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_triangular(L, U, target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                            hmatrix_child->recursive_build_hmatrix_triangular_product(sum_restr, L, U);
                        }
                    }
                } else {
                    this->compute_dense_data(sum_expr);
                    // Matrix<CoefficientPrecision> dense_data(target_cluster.get_size(), source_cluster.get_size());
                    // sum_expr.copy_submatrix(dense_data.nb_rows(), dense_data.nb_cols(), 0, 0, dense_data.data());
                    // this->split(dense_data);
                }
            }

            else if ((target_children.size() == 0) and (source_children.size() > 0)) {
                if (test_restrict[1] == 0) {
                    for (const auto &source_child : source_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(&target_cluster, source_child.get());
                        // SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_cluster.get_size(), target_cluster.get_offset(), source_child->get_size(), source_child->get_offset());
                        SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_triangular(L, U, target_cluster.get_size(), target_cluster.get_offset(), source_child->get_size(), source_child->get_offset());

                        hmatrix_child->recursive_build_hmatrix_triangular_product(sum_restr, L, U);
                    }
                } else {
                    // std::cout << "1" << std::endl;

                    this->compute_dense_data(sum_expr);
                    // Matrix<CoefficientPrecision> dense_data(target_cluster.get_size(), source_cluster.get_size());
                    // sum_expr.copy_submatrix(dense_data.nb_rows(), dense_data.nb_cols(), 0, 0, dense_data.data());
                    // this->split(dense_data);
                }
            } else if ((source_children.size() == 0) and (target_children.size() > 0)) {
                if (test_restrict[0] == 0) {
                    for (const auto &target_child : target_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(target_child.get(), &source_cluster);
                        // SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_cluster.get_size(), source_cluster.get_offset());
                        SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_triangular(L, U, target_child->get_size(), target_child->get_offset(), source_cluster.get_size(), source_cluster.get_offset());

                        hmatrix_child->recursive_build_hmatrix_triangular_product(sum_restr, L, U);
                    }
                } else {

                    this->compute_dense_data(sum_expr);
                    // Matrix<CoefficientPrecision> dense_data(target_cluster.get_size(), source_cluster.get_size());
                    // sum_expr.copy_submatrix(dense_data.nb_rows(), dense_data.nb_cols(), 0, 0, dense_data.data());
                    // this->split(dense_data);
                }
            } else {

                this->compute_dense_data(sum_expr);
            }

        } else {
            this->set_low_rank_data(lr);
        }
    } else if ((target_children.size() == 0) and (source_children.size() == 0)) {
        this->compute_dense_data(sum_expr);
    } else {
        if ((target_children.size() > 0) and (source_children.size() > 0)) {
            if ((test_restrict[0] == 0) and (test_restrict[1] == 0)) {
                for (const auto &target_child : target_children) {
                    for (const auto &source_child : source_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(target_child.get(), source_child.get());
                        // SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                        // SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_s(target_child->get_size(), source_child->get_size(), target_child->get_offset(), source_child->get_offset());
                        SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_triangular(L, U, target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
                        hmatrix_child->recursive_build_hmatrix_triangular_product(sum_restr, L, U);
                    }
                }
            } else {
                // Il y a des trop gros blocs denses , normalemtn hmatrices =: minblocksize....... c'est a cause de ca qu'on est obligé de cjheker la restrictibilité a chaque fois....

                this->compute_dense_data(sum_expr);
            }
        }

        else if ((target_children.size() == 0) and (source_children.size() > 0)) {
            if (test_restrict[1] == 0) {
                for (const auto &source_child : source_children) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(&target_cluster, source_child.get());
                    // SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_cluster.get_size(), target_cluster.get_offset(), source_child->get_size(), source_child->get_offset());
                    SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_triangular(L, U, target_cluster.get_size(), target_cluster.get_offset(), source_child->get_size(), source_child->get_offset());

                    hmatrix_child->recursive_build_hmatrix_triangular_product(sum_restr, L, U);
                }
            } else {
                this->compute_dense_data(sum_expr);
            }
        } else if ((source_children.size() == 0) and (target_children.size() > 0)) {
            if (test_restrict[0] == 0) {
                for (const auto &target_child : target_children) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(target_child.get(), &source_cluster);
                    // SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_clean(target_child->get_size(), target_child->get_offset(), source_cluster.get_size(), source_cluster.get_offset());
                    SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict_triangular(L, U, target_child->get_size(), target_child->get_offset(), source_cluster.get_size(), source_cluster.get_offset());

                    hmatrix_child->recursive_build_hmatrix_triangular_product(sum_restr, L, U);
                }
            } else {
                this->compute_dense_data(sum_expr);
            }
        }
    }
}

////////////////////////
// HMATRIX HMATRIX
// test passé : on fait varier la taille et epsilon
template <typename CoefficientPrecision, typename CoordinatePrecision>
HMatrix<CoefficientPrecision, CoordinatePrecision> HMatrix<CoefficientPrecision, CoordinatePrecision>::hmatrix_product_new(const HMatrix &B) const {
    HMatrix root_hmatrix(this->m_tree_data->m_target_cluster_tree, B.m_tree_data->m_source_cluster_tree, this->m_target_cluster, &B.get_source_cluster());
    root_hmatrix.set_admissibility_condition(this->m_tree_data->m_admissibility_condition);
    root_hmatrix.set_low_rank_generator(this->m_tree_data->m_low_rank_generator);
    root_hmatrix.set_eta(this->m_tree_data->m_eta);
    root_hmatrix.set_epsilon(this->m_tree_data->m_epsilon);
    root_hmatrix.set_maximal_block_size(this->m_tree_data->m_maxblocksize);
    root_hmatrix.set_minimal_target_depth(this->m_tree_data->m_minimal_target_depth);
    root_hmatrix.set_minimal_source_depth(this->m_tree_data->m_minimal_source_depth);

    SumExpression_update<CoefficientPrecision, CoordinatePrecision> root_sum_expression(this, &B);

    root_hmatrix.recursive_build_hmatrix_product_new(root_sum_expression);

    return root_hmatrix;
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product_new(const SumExpression_update<CoefficientPrecision, CoordinatePrecision> &sum_expr) {
    auto &target_cluster  = this->get_target_cluster();
    auto &source_cluster  = this->get_source_cluster();
    auto &target_children = target_cluster.get_children();
    auto &source_children = source_cluster.get_children();
    bool admissible       = this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, this->m_tree_data->m_eta);
    bool flag             = sum_expr.is_restrectible();
    const int minsize     = 45;
    if (admissible) {
        LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(sum_expr, *this->m_tree_data->m_low_rank_generator, this->get_target_cluster(), this->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
        if ((lr.Get_U().nb_rows() == 0) or (lr.Get_U().nb_cols() == 0) or (lr.Get_V().nb_rows() == 0) or (lr.Get_V().nb_cols() == 0)) {
            // on a pas trouvé de bonne approx normalement il faudrait continuer a descedre
            if (flag) {
                for (const auto &target_child : target_children) {
                    for (const auto &source_child : source_children) {

                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child         = this->add_child(target_child.get(), source_child.get());
                        SumExpression_update<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(*target_child, *source_child);
                        // les mauvaises lr approx ont été mise dans sh et on a un flag a false -> pas de bonne solution on est sur des petits blocs
                        // rg(A+B) < rg(A)+rg(B) mais si lda < rg(A)+rg(B) c'est plus rang faible
                        hmatrix_child->recursive_build_hmatrix_product_new(sum_restr);
                    }
                }
            } else {
                // on peut pas descndre a cause de blocs insécabl"es -> soit trop gros bloc dense ou bien lr qu'on a pas pus update
                this->compute_dense_data(sum_expr);
            }
        } else {
            // sionon on atrouvé une approx
            this->set_low_rank_data(lr);
        }
    } else {
        // pas admissible on descend si on est plus gros que min size ET qu'on est restrictible
        if (target_cluster.get_size() >= minsize && source_cluster.get_size() >= minsize) {
            if (flag) {
                for (const auto &target_child : target_children) {
                    for (const auto &source_child : source_children) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child         = this->add_child(target_child.get(), source_child.get());
                        SumExpression_update<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.Restrict(*target_child, *source_child);
                        // on regarde si on est pas tomber sur des mauvaises low rank
                        hmatrix_child->recursive_build_hmatrix_product_new(sum_restr);
                    }
                }
            } else {
                // std::cout << "big dense block in the block structure" << std::endl;
                this->compute_dense_data(sum_expr);
            }
        } else {
            // std::cout << "petit block a densifier" << std::endl;
            this->compute_dense_data(sum_expr);
        }
    }
}
// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product_new(const Sumexpr<CoefficientPrecision, CoordinatePrecision> &sum_expr) {
//     auto &target_cluster  = this->get_target_cluster();
//     auto &source_cluster  = this->get_source_cluster();
//     auto &target_children = target_cluster.get_children();
//     auto &source_children = source_cluster.get_children();
//     bool admissible       = this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, 10);
//     // std::cout << "en haut de prod sr.size =" << sum_expr.get_sr().size() << std::endl;
//     if (target_cluster.get_size() >= 45 && source_cluster.get_size() >= 45) {
//         if (admissible) {
//             LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(sum_expr, *this->m_tree_data->m_low_rank_generator, this->get_target_cluster(), this->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
//             if ((lr.Get_U().nb_rows() == 0) or (lr.Get_U().nb_cols() == 0) or (lr.Get_V().nb_rows() == 0) or (lr.Get_V().nb_cols() == 0)) {
//                 // on a pas trouvé de bonne approx normalement il faudrait continuer a descedre
//                 this->compute_dense_data(sum_expr);

//             } else {
//                 /// il faudrait vraiment mettre un fast tronsation parce que la c'est vraiment pas opti
//                 this->set_low_rank_data(lr);
//             }

//         } else if (((target_children.size() == 0) and (source_children.size() == 0)) or !(sum_expr.is_restrectible())) {
//             LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(sum_expr, *this->m_tree_data->m_low_rank_generator, this->get_target_cluster(), this->get_source_cluster(), -1, this->m_tree_data->m_epsilon);
//             if ((lr.Get_U().nb_rows() == 0) or (lr.Get_U().nb_cols() == 0) or (lr.Get_V().nb_rows() == 0) or (lr.Get_V().nb_cols() == 0)) {

//                 this->compute_dense_data(sum_expr);

//             } else {
//                 /// il faudrait vraiment mettre un fast tronsation parce que la c'est vraiment pas opti
//                 this->set_low_rank_data(lr);
//             }
//         } else {
//             // if (target_cluster.get_size() > source_cluster.get_size()) {
//             //     for (const auto &target_child : target_children) {
//             //         std::cout << "1111 appelé par " << target_cluster.get_size() << ',' << source_cluster.get_size() << std::endl;
//             //         HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(target_child.get(), &source_cluster);
//             //         Sumexpr<CoefficientPrecision, CoordinatePrecision> sum_restr      = sum_expr.restrict(*target_child, source_cluster);
//             //         hmatrix_child->recursive_build_hmatrix_product_new(sum_restr);
//             //     }
//             // } else if (target_cluster.get_size() < source_cluster.get_size()) {
//             //     for (const auto &source_child : source_children) {
//             //         std::cout << "2222 appelé par " << target_cluster.get_size() << ',' << source_cluster.get_size() << std::endl;

//             //         HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(&target_cluster, source_child.get());
//             //         Sumexpr<CoefficientPrecision, CoordinatePrecision> sum_restr      = sum_expr.restrict(target_cluster, *source_child);
//             //         hmatrix_child->recursive_build_hmatrix_product_new(sum_restr);
//             //     }
//             // } else {
//             for (const auto &target_child : target_children) {
//                 for (const auto &source_child : source_children) {
//                     std::cout << "3333 appelé par " << target_cluster.get_size() << ',' << source_cluster.get_size() << std::endl;

//                     HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(target_child.get(), source_child.get());
//                     Sumexpr<CoefficientPrecision, CoordinatePrecision> sum_restr      = sum_expr.restrict(*target_child, *source_child);
//                     hmatrix_child->recursive_build_hmatrix_product_new(sum_restr);
//                 }
//             }
//             // }
//             // if ((target_children.size() > 0) and (source_children.size() > 0)) {
//             //     for (const auto &target_child : target_children) {
//             //         for (const auto &source_child : source_children) {
//             //             HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(target_child.get(), source_child.get());
//             //             Sumexpr<CoefficientPrecision, CoordinatePrecision> sum_restr      = sum_expr.restrict(*target_child, *source_child);
//             //             hmatrix_child->recursive_build_hmatrix_product_new(sum_restr);
//             //         }
//             //     }
//             // }

//             // else if ((target_children.size() == 0) and (source_children.size() > 0)) {
//             //     for (const auto &source_child : source_children) {
//             //         HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(&target_cluster, source_child.get());
//             //         Sumexpr<CoefficientPrecision, CoordinatePrecision> sum_restr      = sum_expr.restrict(target_cluster, *source_child);
//             //         hmatrix_child->recursive_build_hmatrix_product_new(sum_restr);
//             //     }
//             // } else if ((source_children.size() == 0) and (target_children.size() > 0)) {
//             //     for (const auto &target_child : target_children) {
//             //         HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child = this->add_child(target_child.get(), &source_cluster);
//             //         Sumexpr<CoefficientPrecision, CoordinatePrecision> sum_restr      = sum_expr.restrict(*target_child, source_cluster);
//             //         hmatrix_child->recursive_build_hmatrix_product_new(sum_restr);
//             //     }
//             // }
//         }
//     } else {
//         this->compute_dense_data(sum_expr);
//     }
// }
// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::recursive_build_hmatrix_product(const SumExpression<CoefficientPrecision, CoordinatePrecision> &sum_expr) {
//     auto &target_cluster  = this->get_target_cluster();
//     auto &source_cluster  = this->get_source_cluster();
//     auto &target_children = target_cluster.get_children();
//     auto &source_children = source_cluster.get_children();
//     bool admissible       = this->m_tree_data->m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, this->m_tree_data->m_eta);

//     if (admissible) {
//         LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(sum_expr, *this->m_tree_data->m_low_rank_generator, this->get_target_cluster(), this->get_source_cluster(), -1, 0.001);
//         if ((lr.Get_U().nb_rows() == 0) or (lr.Get_U().nb_cols() == 0) or (lr.Get_V().nb_rows() == 0) or (lr.Get_V().nb_cols() == 0)) {
//             this->compute_dense_data(sum_expr);

//         } else {
//             this->set_low_rank_data(lr);
//         }

//     } else if ((target_children.size() == 0) and (source_children.size() == 0)) {
//         this->compute_dense_data(sum_expr);
//     } else {
//         if ((target_children.size() > 0) and (source_children.size() > 0)) {
//             for (const auto &target_child : target_children) {
//                 for (const auto &source_child : source_children) {
//                     HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), source_child.get());
//                     SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.ultim_restrict(target_child->get_size(), target_child->get_offset(), source_child->get_size(), source_child->get_offset());
//                     hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//                 }
//             }
//         }

//         else if ((target_children.size() == 0) and (source_children.size() > 0)) {
//             for (const auto &source_child : source_children) {
//                 HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(&target_cluster, source_child.get());
//                 SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.ultim_restrict(target_cluster.get_size(), target_cluster.get_offset(), source_child->get_size(), source_child->get_offset());
//                 hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//             }
//         } else if ((source_children.size() == 0) and (target_children.size() > 0)) {
//             for (const auto &target_child : target_children) {
//                 HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_child  = this->add_child(target_child.get(), &source_cluster);
//                 SumExpression<CoefficientPrecision, CoordinatePrecision> sum_restr = sum_expr.ultim_restrict(target_child->get_size(), target_child->get_offset(), source_cluster.get_size(), source_cluster.get_offset());
//                 hmatrix_child->recursive_build_hmatrix_product(sum_restr);
//             }
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
