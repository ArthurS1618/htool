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

using namespace std;
using namespace htool;
vector<double> to_mesh(const string &file_name) {
    ifstream file(file_name);
    string line;
    vector<double> p;
    double zz             = 3.1415;
    bool follow           = true;
    bool reading_vertices = false;

    while ((getline(file, line)) and (follow)) {
        if (line.find("Vertices") != string::npos) {
            reading_vertices = true;
            getline(file, line);
        }
        if (reading_vertices) {
            if (line.find("Edges") != string::npos) {
                follow = false;
            }
            double x, y, z;
            sscanf(line.c_str(), "%lf %lf %lf", &x, &y, &z);
            p.push_back(x);
            p.push_back(y);
            p.push_back(zz);
        }
    }
    file.close();
    p.erase(p.begin());
    p.erase(p.begin());
    p.erase(p.begin());
    p.pop_back();
    p.pop_back();
    p.pop_back();
    p.pop_back();
    p.pop_back();
    p.pop_back();
    return p;
}
template <class CoefficientPrecision, class CoordinatePrecision>
class MatGenerator : public VirtualGenerator<CoefficientPrecision> {
  private:
    Matrix<CoefficientPrecision> mat;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;

  public:
    MatGenerator(Matrix<CoefficientPrecision> &mat0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : mat(mat0), target(target0), source(source0){};

    MatGenerator(int nr, int nc, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0, const string &matfile) : target(target0), source(source0) {
        ifstream file;
        file.open(matfile);
        CoefficientPrecision *data = new CoefficientPrecision[nr * nc];
        std::vector<CoefficientPrecision> file_content;
        if (!file) {
            std::cerr << "Error : Can't open the file" << std::endl;
        } else {
            std::string word;
            std::vector<std::string> file_content;
            int count      = 0;
            char delimiter = ',';
            string line;
            int i = 0;
            while (getline(file, line)) {
                stringstream ss(line);
                string word;
                int j = 0;
                while (getline(ss, word, delimiter)) {
                    CoefficientPrecision d1;
                    stringstream stream1;
                    stream1 << word;
                    stream1 >> d1;
                    data[i * nc + j] = d1;
                    j += 1;
                }
                i += 1;
            }
        }
        file.close();
        Matrix<CoefficientPrecision> temp(nr, nc);
        temp.assign(nr, nc, data, true);
        mat = temp;
    }

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

    // build
    auto root_hmatrix_21 = hmatrix_tree_builder_21.build(generator_21);
    auto root_hmatrix_32 = hmatrix_tree_builder_32.build(generator_32);
    // auto leaf            = root_hmatrix_21.get_leaves();
    // std::vector<const HMatrix<T, htool::underlying_type<T>> *> lowrank;
    // std::vector<const HMatrix<T, htool::underlying_type<T>> *> hmat;
    // std::vector<const LowRankMatrix<T, htool::underlying_type<T>> *> vect_lr;
    // std::cout << leaf.size() << std::endl;
    // for (int k = 0; k < leaf.size(); ++k) {
    //     auto &l = leaf[k];
    //     if (l->is_low_rank()) {
    //         lowrank.push_back(l);
    //         auto lr = l->get_low_rank_data();
    //         vect_lr.push_back(lr);
    //     } else {
    //         hmat.push_back(l);
    //     }
    // }

    // auto &H_children = root_hmatrix_21.get_children();
    // for (int k = 0; k < H_children.size(); ++k) {
    //     auto &Hk = H_children[k];
    //     std::cout << Hk->get_target_cluster().get_size() << ',' << Hk->get_source_cluster().get_size() << std::endl;
    // }
    // SumExpression<T, htool::underlying_type<T>> sumexpr(&root_hmatrix_21, &root_hmatrix_21);
    // auto srestr = sumexpr.Restrict(100, 0, 100, 0);
    // std::cout << srestr.get_sh().size() << ',' << srestr.get_sr().size() << std::endl;
    // auto srestr1 = srestr.Restrict(50, 0, 50, 0);
    // std::cout << srestr1.get_sh().size() << ',' << srestr1.get_sr().size() << std::endl;
    // auto ssr = srestr1.get_sr();
    // auto lr  = ssr[0];
    // std::cout << lr->nb_rows() << ',' << lr->nb_cols() << ',' << lr->rank_of() << std::endl;
    // auto srestr2 = srestr1.Restrict(25, 0, 25, 0);
    // std::cout << srestr2.get_sh().size() << ',' << srestr2.get_sr().size() << std::endl;
    // auto sssr = srestr2.get_sr();
    // for (int k = 0; k < sssr.size(); ++k) {
    //     auto lllr = sssr[k];
    //     std::cout << lllr->nb_rows() << ',' << lllr->nb_cols() << ',' << lllr->rank_of() << std::endl;
    // }

    // SumExpression<T, htool::underlying_type<T>> stest(&root_hmatrix_32, &root_hmatrix_21);
    // std::cout << stest.get_coeff(3, 8) << ',' << reference_dense_matrix(3, 8) << std::endl;
    // std::cout << stest.get_coeff(3, 8) << std::endl;
    // // //////////////////////////////////////
    auto root_hmatrix_31 = root_hmatrix_32.hmatrix_product(root_hmatrix_21);

    // // Compute error
    Matrix<T> hmatrix_to_dense_31(root_hmatrix_31.get_target_cluster().get_size(), root_hmatrix_31.get_source_cluster().get_size());
    copy_to_dense(root_hmatrix_31, hmatrix_to_dense_31.data());
    // std::cout << "?????" << std::endl;
    std::cout << "erreur multiplication" << normFrob(reference_dense_matrix - hmatrix_to_dense_31) / normFrob(reference_dense_matrix) << "\n";

    // // test LU
    // std::cout << "test LU" << std::endl;
    // auto tlu  = LU(reference_dense_matrix);
    // auto LetU = get_lu(tlu);
    // auto L    = LetU.first;
    // auto U    = LetU.second;
    // std::cout << "norm(aA-L*U)/norm(a) " << normFrob(reference_dense_matrix - L * U) / normFrob(reference_dense_matrix) << std::endl;
    // std::cout << "test Lx = y" << std::endl;

    // HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_L(root_cluster_3, root_cluster_2, epsilon, eta, 'N', 'N');
    // HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_U(root_cluster_2, root_cluster_1, epsilon, eta, 'N', 'N');

    // GeneratorTestType generator_L(3, size_2, size_1, p2, p1, root_cluster_2, root_cluster_1);
    // GeneratorTestType generator_U(3, size_3, size_2, p3, p2, root_cluster_3, root_cluster_2);
    // auto ll = L;
    // auto uu = U;
    // generator_L.copy_submatrix(L.nb_rows(), L.nb_cols(), 0, 0, ll.data());
    // generator_U.copy_submatrix(U.nb_rows(), U.nb_cols(), 0, 0, uu.data());
    // auto reflu = L * U;

    // auto Lh = hmatrix_tree_builder_L.build(generator_L);
    // auto Uh = hmatrix_tree_builder_U.build(generator_U);

    // Matrix<T> LhUh(root_cluster_1->get_size(), root_cluster_1->get_size());
    // Matrix<T> lh(root_cluster_1->get_size(), root_cluster_1->get_size());
    // Matrix<T> uh(root_cluster_1->get_size(), root_cluster_1->get_size());
    // copy_to_dense(Lh, lh.data());
    // copy_to_dense(Uh, uh.data());
    // auto LU = Lh.hmatrix_product(Uh);
    // copy_to_dense(LU, LhUh.data());
    // std::cout << "compression L" << Lh.get_compression() << std::endl;
    // std::cout << "compression U" << Uh.get_compression() << std::endl;
    // std::cout << " l contre Lh" << normFrob(L - lh) / normFrob(L) << std::endl;
    // std::cout << " u contre Uh" << normFrob(U - uh) / normFrob(U) << std::endl;
    // std::cout << "LhUh contre LU" << normFrob(L * U - LhUh) / normFrob(L * U) << std::endl;

    // ///////////////////////////////////////////////////////////
    // ////// TEST LU AVEC MATRICE DE POISSON
    // //////////////////////////////////////////////////
    // std::cout << "______________________________________________________" << std::endl;
    // std::vector<htool::underlying_type<T>> points = to_mesh("/work/sauniear/Documents/matrice_test/FreeFem Data/H_LU/mesh_poisson518.txt");

    // ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> recursive_build_strategy(points.size() / 3, 3, points.data(), 2, 2);

    // std::shared_ptr<Cluster<htool::underlying_type<T>>> root_poisson = make_shared<Cluster<htool::underlying_type<T>>>(recursive_build_strategy.create_cluster_tree());

    // MatGenerator<double, htool::underlying_type<double>> Mpoisson(root_poisson->get_size(), root_poisson->get_size(), *root_poisson, *root_poisson, "/work/sauniear/Documents/matrice_test/FreeFem Data/H_LU/Poisson518.txt");
    // Matrix<double> poisson_perm = Mpoisson.get_perm_mat();
    // auto mat                    = poisson_perm;
    // std::cout << "test lu " << std::endl;
    // // std::vector<int> ipiv(518);
    // // int t    = 518;
    // // int *LDA = &t;
    // // int *nr  = &t;
    // // int *nc  = &t;
    // // // Lapack<double>::getrf(nr, nc, mat.data(), LDA, ipiv.data(), 0);
    // auto LetU = LU(poisson_perm);
    // // Matrix<double> MM(518, 518);
    // // MM.assign(518, 518, mat.data(), true);
    // // auto LetU = get_lu(MM);
    // auto L = get_lu(LetU).first;
    // auto U = get_lu(LetU).second;
    // std::cout << normFrob(L * U - poisson_perm) / normFrob(poisson_perm) << std::endl;

    // MatGenerator<double, double> lh(L, *root_poisson, *root_poisson);
    // MatGenerator<double, double> uh(U, *root_poisson, *root_poisson);

    // auto lperm = lh.get_unperm_mat();
    // auto uperm = uh.get_unperm_mat();
    // MatGenerator<double, double> lll(lperm, *root_poisson, *root_poisson);
    // MatGenerator<double, double> uuu(uperm, *root_poisson, *root_poisson);

    // HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder1(root_poisson, root_poisson, epsilon, eta, 'N', 'N');
    // HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder2(root_poisson, root_poisson, epsilon, eta, 'N', 'N');

    // auto Lh = hmatrix_tree_builder1.build(lll);
    // auto Uh = hmatrix_tree_builder2.build(uuu);
    // // auto LhUh = Lh.hmatrix_product(Uh);
    // Matrix<double> ll(518, 518);
    // Matrix<double> uu(518, 518);
    // copy_to_dense(Lh, ll.data());
    // copy_to_dense(Uh, uu.data());
    // // std::cout << Lh.get_compression() << std::endl;
    // std::cout << Uh.get_compression() << std::endl;

    // std::cout << "erreur sur Lh " << normFrob(ll - L) / normFrob(L) << std::endl;
    // std::cout << "erreur uh " << normFrob(U - uu) / normFrob(U) << std::endl;

    // Matrix<double> temp(518, 518);
    // // copy_to_dense(LhUh, temp.data());
    // // std::cout << "erreur a-LhUh " << normFrob(mat - temp) / normFrob(mat) << std::endl;

    // for (auto &child_L : Lh.get_children()) {
    //     std::cout << "_____________________" << std::endl;
    //     std::cout << child_L->get_target_cluster().get_size() << ',' << child_L->get_target_cluster().get_offset() << std::endl;
    //     std::cout << child_L->get_source_cluster().get_size() << ',' << child_L->get_source_cluster().get_offset() << std::endl;
    //     std::cout << "______________________" << std::endl;
    // }
    // auto L1 = Lh.get_block(129, 129, 0, 0);
    // std::cout << L1->get_target_cluster().get_size() << ',' << L1->get_target_cluster().get_offset() << '!' << L1->get_source_cluster().get_size() << ',' << L1->get_source_cluster().get_offset() << std::endl;
    // for (auto &child : L1->get_children()) {
    //     std::cout << child->get_target_cluster().get_size() << ',' << child->get_target_cluster().get_offset() << '!' << child->get_source_cluster().get_size() << ',' << L1->get_source_cluster().get_offset() << std::endl;
    // }

    // std::vector<double> x_sol(root_poisson->get_size(), 1.0);
    // std::vector<double> x_solu(root_poisson->get_size(), 1.0);
    // std::vector<double> yl(root_poisson->get_size(), 0.0);
    // std::vector<double> yu(root_poisson->get_size(), 0.0);
    // Lh.add_vector_product('N', 1.0, x_sol.data(), 0.0, yl.data());
    // Uh.add_vector_product('N', 1.0, x_solu.data(), 0.0, yu.data());
    // std::vector<double> x_u(root_poisson->get_size(), 0.0);
    // // std::cout << "xu " << norm2(x_u) << ',' << x_u.size() << std::endl;
    // std::vector<double> x_l(root_poisson->get_size(), 0.0);
    // Lh.forward_substitution_extract(*root_poisson, yl, x_l);
    // Uh.backward_substitution_extract(root_poisson->get_size(), root_poisson->get_offset(), yu, x_u);
    // //  std::cout << norm2(x_u) << ',' << x_u.size() << std::endl;
    // std::cout << norm2(x_sol - x_l) / norm2(x_sol) << std::endl;
    // std::cout << norm2(x_solu - x_u) / norm2(x_solu) << std::endl;

    // ///////////////////////////////////////
    // //// test LU x =y
    // ///////////////////////////////////
    // std::cout << "__________________________________" << std::endl;
    // std::cout << "test LU x =y" << std::endl;
    // std::vector<double> X(root_poisson->get_size(), 1.0);
    // std::vector<double> Ytemp(root_poisson->get_size(), 0.0);
    // std::vector<double> Y(root_poisson->get_size(), 0.0);
    // Uh.add_vector_product('N', 1.0, X.data(), 0.0, Ytemp.data());
    // Lh.add_vector_product('N', 1.0, Ytemp.data(), 0.0, Y.data());
    // std::vector<double> xtest(root_poisson->get_size(), 0.0);
    // forward_backward(Lh, Uh, *root_poisson, Y, xtest);
    // std::cout << norm2(xtest - X) / norm2(X) << std::endl;

    //  std::cout << "________________________________________" << std::endl;
    //  auto lfoisl = Lh.hmatrix_product(Lh);
    //  Matrix<double> l2(518, 518);
    //  copy_to_dense(lfoisl, l2.data());
    //  std::cout << normFrob(l2 - L * L) / normFrob(L * L) << std::endl;
    //  Lapack<CoefficientPrecision>::getrf(, "A", &m, &n, mat.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);
    //  Matrix<T> LUpoisson    = LU(poisson_perm);
    //  std::cout << normFrob(poisson_perm) << std::endl;
    //  std::cout << normFrob(get_lu(LUpoisson).first * get_lu(LUpoisson).second - poisson_perm) / normFrob(poisson_perm) << std::endl;
    //  std::cout << poisson_perm.nb_rows() << ',' << poisson_perm.nb_cols() << std::endl;
    //  for (int l = 0; l < 10; ++l) {
    //      std::cout << poisson_perm(1, l) << ',';
    //  }

    //    std::cout << "!" << normFrob(reference_dense_matrix - reflu) / normFrob(reference_dense_matrix) << std::endl;
    //    std::cout << "!" << normFrob(reflu - LhUh) / normFrob(reflu) << std::endl;
    //    std::cout << "!" << normFrob(reference_dense_matrix - LhUh) / normFrob(reference_dense_matrix) << std::endl;

    ////////////////////////////////////////////////////////////////
    // Matrix<T> hmatrix_to_dense_21(root_hmatrix_21.get_target_cluster().get_size(), root_hmatrix_21.get_source_cluster().get_size()), hmatrix_to_dense_32(root_hmatrix_32.get_target_cluster().get_size(), root_hmatrix_32.get_source_cluster().get_size());
    // copy_to_dense(root_hmatrix_21, hmatrix_to_dense_21.data());
    // copy_to_dense(root_hmatrix_32, hmatrix_to_dense_32.data());
    // std::cout << normFrob(dense_matrix_21 - hmatrix_to_dense_21) / normFrob(dense_matrix_21) << "\n";
    // std::cout << normFrob(dense_matrix_32 - hmatrix_to_dense_32) / normFrob(dense_matrix_32) << "\n";

    // // HMatrix product
    // // HMatrix result_hmatrix = root_hmatrix_32.prod(root_hmatrix_21);
    // // HMatrix result_hmatrix = prod(root_hmatrix_32,root_hmatrix_21);

    // cout << "TEST ARTHUR POUR BIEN MANIPULER" << endl;
    // // Hmatrix
    // cout << root_hmatrix_21.get_target_cluster().get_size() << ',' << root_hmatrix_21.get_source_cluster().get_size() << endl;
    // //--------> ca a pas l'aire d'(avoir changer
    // // Accés au fils
    // cout << "nb sons" << endl;
    // cout << root_hmatrix_21.get_children().size() << endl;
    // auto &v  = root_hmatrix_21.get_children();
    // auto &h1 = v[0];
    // auto &h2 = v[1];
    // cout << h1->get_target_cluster().get_size() << ',' << h2->get_source_cluster().get_size() << endl;
    // cout << h1->get_target_cluster().get_offset() << ',' << h2->get_target_cluster().get_offset() << endl;

    // // Test sur les sum expression
    // cout << "TEST SUM EXPRESSION" << endl;
    // cout << "____________________" << endl;
    // // SumExpression<T, htool::underlying_type<T>> sumexpr(h1.get(), h2.get());
    // SumExpression<T, htool::underlying_type<T>> sumexpr(&root_hmatrix_32, &root_hmatrix_21);

    // auto tt = sumexpr.get_sh();

    // cout << tt.size() << endl;
    // auto t0 = tt[0];
    // auto t1 = tt[1];
    // cout << " h32 target size , offset " << endl;
    // cout << t0->get_target_cluster().get_size() << ',' << t0->get_target_cluster().get_offset() << endl;
    // cout << "h32 source size , offset " << endl;
    // cout << t0->get_source_cluster().get_size() << ',' << t0->get_source_cluster().get_offset() << endl;
    // cout << "h21 target size , offset " << endl;
    // cout << t0->get_target_cluster().get_size() << ',' << t0->get_target_cluster().get_offset() << endl;
    // cout << "h21 source size , offset " << endl;
    // cout << t0->get_source_cluster().get_size() << ',' << t0->get_source_cluster().get_offset() << endl;

    // cout << "________________________________" << endl;
    // cout << "test multiplication sumexpr vecteur " << endl;

    // std::vector<T> x(size_3, 1.0);

    // std::vector<T> y  = reference_dense_matrix * x;
    // std::vector<T> yy = sumexpr.prod(x);
    // double norm       = 0;
    // for (int k = 0; k < y.size(); ++k) {
    //     norm += (y[k] - yy[k]) * (y[k] - yy[k]);
    // }
    // norm = sqrt(norm);
    // cout << norm << endl;
    // cout << " ca a l'aire ok " << endl;

    // cout << "test sur la restriction" << endl;
    // auto &target_sons = t0->get_target_cluster().get_children();
    // auto &source_sons = t1->get_source_cluster().get_children();
    // for (int i = 0; i < target_sons.size(); ++i) {
    //     auto &ti = target_sons[i];
    //     for (int j = 0; j < source_sons.size(); ++j) {
    //         auto &sj = source_sons[j];
    //         cout << "target_sons size et offset " << ti->get_size() << ',' << ti->get_offset() << endl;
    //         cout << "source sons size et offset " << sj->get_size() << ',' << sj->get_offset() << endl;
    //         SumExpression<T, htool::underlying_type<T>> restriction = sumexpr.Restrict(ti->get_size(), ti->get_offset(), sj->get_size(), sj->get_offset());
    //         cout << "restrict : sh size: " << restriction.get_sh().size() << ", sr size " << restriction.get_sr().size() << endl;
    //         auto SH = restriction.get_sh();
    //         for (int k = 0; k < SH.size() / 2; ++k) {
    //             auto H  = SH[2 * k];
    //             auto &K = SH[2 * k + 1];
    //             cout << " H target-> " << H->get_target_cluster().get_size() << ',' << H->get_target_cluster().get_offset() << endl;
    //             cout << " H source->" << H->get_source_cluster().get_size() << ',' << H->get_source_cluster().get_offset() << endl;
    //             cout << "K target -> " << K->get_target_cluster().get_size() << ',' << K->get_target_cluster().get_offset() << endl;
    //             cout << "K source ->" << K->get_source_cluster().get_size() << ',' << K->get_source_cluster().get_offset() << endl;
    //         }
    //     }
    // }
    // // 4enfants
    // // SumExpression<T, htool::underlying_type<T>> restr = sumexpr.Restrict(100, 100, 0, 0);
    // // auto test                                         = restr.get_sh();
    // // cout << test.size() << endl;
    // // for (int k = 0; k < test.size() / 2; ++k) {
    // //     cout << "fils " << k << endl;
    // //     auto testk  = test[2 * k];
    // //     auto testkk = test[2 * k + 1];
    // //     cout << " offset target et source de A " << endl;
    // //     cout << testk->get_target_cluster().get_offset() << ',' << testk->get_source_cluster().get_offset() << endl;
    // //     cout << " offset target et source de B " << endl;
    // //     cout << testkk->get_target_cluster().get_offset() << ',' << testkk->get_source_cluster().get_offset() << endl;
    // //     // cout << " size target et source de A "<< endl;
    // //     // cout << testk->get_target_cluster().get_size() << ',' << testk->get_source_cluster().get_size() << endl;
    // //     // cout << " size target et source de B "<< endl;
    // //     // cout << testkk->get_target_cluster().get_size() << ',' << testkk->get_source_cluster().get_size() << endl;
    // // }
    // // cout << "-b-b-b--bbb-b-b-b-b-bb-b-b-b-bb-b-b-b-b-b-b-b-b" << endl;
    // // SumExpression<T, htool::underlying_type<T>> restr2 = restr.Restrict(50, 50, 0, 0);
    // // auto test2                                         = restr2.get_sh();
    // // cout << test2.size() << endl;
    // // for (int k = 0; k < test2.size() / 2; ++k) {
    // //     cout << "fils " << k << endl;
    // //     auto testk  = test2[2 * k];
    // //     auto testkk = test2[2 * k + 1];
    // //     cout << " offset target et source de A " << endl;
    // //     cout << testk->get_target_cluster().get_offset() << ',' << testk->get_source_cluster().get_offset() << endl;
    // //     cout << " offset target et source de B " << endl;
    // //     cout << testkk->get_target_cluster().get_offset() << ',' << testkk->get_source_cluster().get_offset() << endl;
    // //     // cout << " size target et source de A "<< endl;
    // //     // cout << testk->get_target_cluster().get_size() << ',' << testk->get_source_cluster().get_size() << endl;
    // //     // cout << " size target et source de B "<< endl;
    // //     // cout << testkk->get_target_cluster().get_size() << ',' << testkk->get_source_cluster().get_size() << endl;
    // // }
    // // ------------------------> Ca à l'aire de marcher
    // // auto &child = root_hmatrix_21.get_children();
    // // for (int k = 0; k < child.size(); ++k) {
    // //     auto &child_k                                         = child[k];
    // //     auto &t                                               = child_k->get_target_cluster();
    // //     auto &s                                               = child_k->get_source_cluster();
    // //     SumExpression<T, htool::underlying_type<T>> sum_restr = sumexpr.Restrict(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    // //     cout << "____________________________" << endl;
    // //     cout << sumexpr.get_sr().size() << ',' << sumexpr.get_sh().size() << endl;
    // //     cout << sum_restr.get_sr().size() << ',' << sum_restr.get_sh().size() << endl;
    // // }
    // cout << "test evaluate" << endl;
    // // // normalement si je lui donne sumexpr = 32*21 je devrais tomber sur le produit

    // Matrix<T> eval = sumexpr.Evaluate();
    // cout << normFrob(eval - reference_dense_matrix) << endl;

    // // // // C'est vraimùent pas des tests incroyables mais pour l'instant tout marche
    // // // // Par contre on a pas pus tester la composante sr , mais bon techniquement c'est des matrices donc il devrait pas il y avoir de pb

    // cout << "hmult" << endl;
    // // shared_ptr<const Cluster<double>> clust1;
    // // clust1 = make_shared<const Cluster<double>>(&root_hmatrix_32.get_target_cluster());
    // // shared_ptr<const Cluster<double>> clust2;
    // // clust2 = make_shared<const Cluster<double>>(&root_hmatrix_21.get_target_cluster());
    // // HMatrix<T, htool::underlying_type<T>> L(root_hmatrix_32.get_root_target(), root_hmatrix_32.get_root_source());
    // // sumexpr.Hmult(&L);
    // HMatrix<T, htool::underlying_type<T>> L = root_hmatrix_32.hmatrix_product(root_hmatrix_21);
    // cout << "hmult ok" << endl;
    // // vector<T> xxx ( L.get_source_cluster().get_size() , 1);
    // // vector<T> yyy ( L.get_target_cluster().get_size(),0);
    // // L.add_vector_product('N',1.0, xxx.data(), 0.0, yyy.data());
    // // cout << norm2 ( yyy-reference_dense_matrix*xxx ) << endl;
    // Matrix<T> L_hmult(L.get_target_cluster().get_size(), L.get_source_cluster().get_size());
    // // cout << "hmult ok" << endl;
    // // auto testmult = L.get_leaves();
    // // auto testleav = root_hmatrix_21.get_leaves();
    // // std::cout << testleav.size() << std::endl;
    // // cout << testmult.size() << endl;
    // // // cout << "leaves ok" << endl;
    // // // vector<T> xxx(L.get_source_cluster().get_size(), 1);
    // // // vector<T> yyy(L.get_target_cluster().get_size(), 0);
    // // // L.add_vector_product('N', 1.0, xxx.data(), 0.0, yyy.data());
    // // // cout << "matrice vecteur " << endl;
    // // // cout << norm2(yyy - reference_dense_matrix * xxx) << endl;
    // copy_to_dense(L, L_hmult.data());
    // cout << "erreur Hmult" << endl;
    // cout << normFrob(L_hmult - reference_dense_matrix) << endl;
    // cout << normFrob(reference_dense_matrix) << endl;
    return is_error;
}
