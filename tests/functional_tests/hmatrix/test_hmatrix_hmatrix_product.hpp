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

    // // Geometry
    vector<double> p1(3 * size_1), p2(3 * size_2), p3(3 * size_3);
    create_disk(3, 0., size_1, p1.data());
    create_disk(3, 0.5, size_2, p2.data());
    create_disk(3, 1., size_3, p3.data());
    std::cout << "création cluster ok" << std::endl;

    // Clustering
    ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> recursive_build_strategy_1(size_1, 3, p1.data(), 2, 2);
    ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> recursive_build_strategy_2(size_2, 3, p2.data(), 2, 2);
    ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> recursive_build_strategy_3(size_3, 3, p3.data(), 2, 2);

    std::shared_ptr<Cluster<htool::underlying_type<T>>> root_cluster_1 = make_shared<Cluster<htool::underlying_type<T>>>(recursive_build_strategy_1.create_cluster_tree());
    std::shared_ptr<Cluster<htool::underlying_type<T>>> root_cluster_2 = make_shared<Cluster<htool::underlying_type<T>>>(recursive_build_strategy_2.create_cluster_tree());
    std::shared_ptr<Cluster<htool::underlying_type<T>>> root_cluster_3 = make_shared<Cluster<htool::underlying_type<T>>>(recursive_build_strategy_3.create_cluster_tree());
    std::cout << "Cluster tree ok" << std::endl;
    // Generator
    GeneratorTestType generator_21(3, size_2, size_1, p2, p1, root_cluster_2, root_cluster_1);
    GeneratorTestType generator_32(3, size_3, size_2, p3, p2, root_cluster_3, root_cluster_2);

    // References
    Matrix<T> dense_matrix_21(root_cluster_2->get_size(), root_cluster_1->get_size()), dense_matrix_32(root_cluster_3->get_size(), root_cluster_2->get_size());
    generator_21.copy_submatrix(dense_matrix_21.nb_rows(), dense_matrix_21.nb_cols(), 0, 0, dense_matrix_21.data());
    generator_32.copy_submatrix(dense_matrix_32.nb_rows(), dense_matrix_32.nb_cols(), 0, 0, dense_matrix_32.data());
    std::cout << "generator ok" << std::endl;

    // HMatrix
    double eta = 10;

    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_21(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_32(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');

    // build
    auto root_hmatrix_21 = hmatrix_tree_builder_21.build(generator_21);
    auto root_hmatrix_32 = hmatrix_tree_builder_32.build(generator_32);
    std::cout << "hmat build ok " << std::endl;

    /////////////////////////////////
    /// test hmatrice matrice

    std::cout << "test hmatrice x matrice " << std::endl;

    Matrix<double> ref_prod = dense_matrix_32 * dense_matrix_21;
    auto test_hm            = root_hmatrix_32.hmat_mat(dense_matrix_21);
    std::cout << "ereur hmat mat :" << normFrob(test_hm - ref_prod) / normFrob(ref_prod) << std::endl;
    std::cout << "test matrice hmatrice " << std::endl;
    auto test_mh = root_hmatrix_21.mat_hmat(dense_matrix_32);
    std::cout << "erreur mat hmat :" << normFrob(ref_prod - test_mh) / normFrob(test_mh) << std::endl;

    clock_t start_time   = std::clock(); // temps de départ
    auto root_hmatrix_31 = root_hmatrix_32.hmatrix_product(root_hmatrix_21);
    clock_t end_time     = clock(); // temps de fin
    double time_taken0   = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "duration multiplication:" << time_taken0 << std::endl;

    // start_time = std::clock(); // temps de départ
    // for (int i = 0; i < root_hmatrix_31.get_target_cluster().get_size(); ++i) {
    //     for (int j = 0; j < root_hmatrix_31.get_source_cluster().get_size(); ++j) {
    //         for (int k = 0; k < root_hmatrix_31.get_source_cluster().get_size(); ++k) {
    //         }
    //     }
    // }
    // end_time          = clock(); // temps de fin
    // double time_taken = double(end_time - start_time) / CLOCKS_PER_SEC;
    // std::cout << "temps mult :" << time_taken << std::endl;

    Matrix<T> hmatrix_to_dense_31(root_hmatrix_31.get_target_cluster().get_size(), root_hmatrix_31.get_source_cluster().get_size());
    copy_to_dense(root_hmatrix_31, hmatrix_to_dense_31.data());
    // std::cout << "?????" << std::endl;
    std::cout << "erreur multiplication" << normFrob(ref_prod - hmatrix_to_dense_31) / normFrob(ref_prod) << "\n";
    std::cout << "proportion low rank :" << root_hmatrix_31.get_compression() << std::endl;

    // //////////////////////////////////////////
    // //// TEST MULT HACKBUSH
    // ////////////////////////////////////////

    // HMatrix<T, htool::underlying_type<T>> test(root_cluster_3, root_cluster_1);
    // HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_31(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
    // Matrix<T> test(root_cluster_1->get_size(), root_cluster_1->get_size());
    // MatGenerator<T, htool::underlying_type<T>> Z(test, *root_cluster_1, *root_cluster_1);

    // build

    // auto root_hmatrix_31 = hmatrix_tree_builder_31.build(Z);

    // //////////////////////////////////
    // //// TEST HMAT ANI
    // /////////////////////////////////
    // std::vector<std::vector<T>> RES;
    // std::vector<std::vector<T>> col_U;
    // for (int k = 0; k < root_hmatrix_21.get_target_cluster().get_size(); ++k) {
    //     col_U.push_back(dense_matrix_21.get_col(k));
    //     std::vector<T> ref(root_hmatrix_21.get_target_cluster().get_size(), 0.0);
    //     RES.push_back(ref);
    // }
    // start_time = clock();
    // root_hmatrix_32.hmat_lr_plus(col_U, RES);
    // end_time = clock();
    // Matrix<T> reference(root_hmatrix_21.get_target_cluster().get_size(), root_hmatrix_21.get_target_cluster().get_size());
    // for (int k = 0; k < root_hmatrix_21.get_target_cluster().get_size(); ++k) {
    //     reference.set_col(k, RES[k]);
    // }
    // std::cout << "time ani " << double(end_time - start_time) / CLOCKS_PER_SEC << std::endl;
    // std::cout << "erreur ani " << normFrob(reference - test31) / normFrob(test31) << std::endl;

    /////////////////////////////////////////////////////////////////////////////////
    // start_time         = std::clock(); // temps de départ
    // auto testr         = root_hmatrix_32.lr_hmat(dense_matrix_21);
    // end_time           = clock(); // temps de fin
    // double time_taken1 = double(end_time - start_time) / CLOCKS_PER_SEC;
    // std::cout << "temps lr hmat :" << time_taken1 << std::endl;
    // std::cout << "erreur lr hmat " << normFrob(test22 - testr) / normFrob(test22) << std::endl;

    // start_time = std::clock(); // temps de départ

    // Matrix<T> reference_dense_matrix = dense_matrix_32 * dense_matrix_21;

    // MM(&root_hmatrix_31, &root_hmatrix_32, &root_hmatrix_21, root_cluster_1.get(), root_cluster_1.get(), root_cluster_1.get());
    // std::cout << normFrob(reference_dense_matrix - test) / normFrob(reference_dense_matrix) << std::endl;

    // clock_t start_time   = std::clock(); // temps de départ
    // auto root_hmatrix_31 = root_hmatrix_32.hmatrix_product(root_hmatrix_21);
    // clock_t end_time     = clock(); // temps de fin
    // double time_taken    = double(end_time - start_time) / CLOCKS_PER_SEC;
    // std::cout << "temps hmult :" << time_taken << std::endl;
    // start_time         = std::clock(); // temps de départ
    // auto testt         = dense_matrix_32 * dense_matrix_21;
    // end_time           = clock(); // temps de fin
    // double time_taken0 = double(end_time - start_time) / CLOCKS_PER_SEC;
    // std::cout << "temps mult :" << time_taken0 << std::endl;
    // Matrix<T> res(root_hmatrix_31.get_target_cluster().get_size(), root_hmatrix_31.get_source_cluster().get_size());
    // copy_to_dense(root_hmatrix_31, res.data());
    // std::cout << "erreur " << normFrob(reference_dense_matrix - res) / normFrob(res) << std::endl;

    // std::vector<double> xxref(root_hmatrix_21.get_source_cluster().get_size(), 1.0);
    // auto yref = dense_matrix_21 * xxref;
    // std::vector<double> xtest(root_cluster_1->get_size(), 1.0);
    // std::vector<double> y(root_cluster_1->get_size(), 0.0);
    // std::vector<double> yy(root_cluster_2->get_size(), 0.0);
    // start_time = std::clock();
    // root_hmatrix_21.add_vector_product('N', 1.0, xtest.data(), 0.0, y.data());
    // end_time           = std::clock();
    // double time_taken1 = double(end_time - start_time) / CLOCKS_PER_SEC;
    // std::cout << "pierre" << norm2(y - yref) / norm2(yref) << ',' << time_taken1 << std::endl;
    // start_time = std::clock();
    // root_hmatrix_21.mat_vec(xtest, yy);
    // end_time           = std::clock();
    // double time_taken2 = double(end_time - start_time) / CLOCKS_PER_SEC;
    // std::cout << "moi" << norm2(yy - yref) / norm2(yref) << ',' << time_taken2 << std::endl;

    // // clock_t start_time1 = std::clock(); // temps de départ
    // // auto test           = dense_matrix_32 * dense_matrix_21;
    // // clock_t end_time1   = clock(); // temps de fin
    // // double time_taken   = double(end_time - start_time) / CLOCKS_PER_SEC;
    // // std::cout << "reg :" << time_taken << std::endl;

    // //    auto root_hmatrix_31 = root_hmatrix_32.hmatrix_product(root_hmatrix_21);
    // //     auto stop            = std::high_resolution_clock::now();
    // //     auto duration        = std::duration_cast<microseconds>(stop - start);
    // //     std::cout << "mult :" << duration.count() << std::endl;
    // //     start     = std::high_resolution_clock::now();
    // //     auto test = std::dense_matrix_32 * dense_matrix_21;
    // //     stop      = std::high_resolution_clock::now();
    // //     duration  = std::duration_cast<microseconds>(stop - start);
    // //     cout << "reg: " << duration.count() << endl;

    // // // Compute error
    // Matrix<T> hmatrix_to_dense_31(root_hmatrix_31.get_target_cluster().get_size(), root_hmatrix_31.get_source_cluster().get_size());
    // copy_to_dense(root_hmatrix_31, hmatrix_to_dense_31.data());
    // // std::cout << "?????" << std::endl;
    // std::cout << "erreur multiplication" << normFrob(reference_dense_matrix - hmatrix_to_dense_31) / normFrob(reference_dense_matrix) << "\n";
    // std::cout << hmatrix_to_dense_31.nb_rows() << ',' << hmatrix_to_dense_31.nb_cols() << std::endl;

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
    // ////// TEST H_LU
    // //////////////////////////////////////////////////
    // std::cout << "______________________________________________________" << std::endl;

    // // // MESH
    // int size = 518;
    // vector<double> points(3 * size);
    // create_disk(3, 0., size, points.data());
    // std::cout << "disque ok " << std::endl;
    // // std::vector<htool::underlying_type<T>> points = to_mesh("/work/sauniear/Documents/matrice_test/FreeFem Data/H_LU/mesh_poisson518.txt");

    // // CLUSTER
    // ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> recursive_build_strategy(points.size() / 3, 3, points.data(), 2, 2);
    // std::shared_ptr<Cluster<htool::underlying_type<T>>> root_poisson = make_shared<Cluster<htool::underlying_type<T>>>(recursive_build_strategy.create_cluster_tree());
    // // GENERATOR
    // GeneratorTestType generator(3, size, size, points, points, root_poisson, root_poisson);
    // std::cout << "generator ok " << std::endl;

    // Matrix<T> reference(root_poisson->get_size(), root_poisson->get_size());
    // std::cout << "init ok " << std::endl;
    // generator.copy_submatrix(reference.nb_rows(), reference.nb_cols(), 0, 0, reference.data());
    // std::cout << "init ok " << std::endl;
    // MatGenerator<double, double> temp(reference, *root_poisson, *root_poisson);
    // Matrix<double> unperm = temp.get_mat();

    // //// LU sur la matrice dense
    // auto LetU = get_lu(LU(reference));
    // auto L    = LetU.first;
    // auto U    = LetU.second;
    // // for (int k = 0; k < 40; ++k) {
    // //     for (int l = 0; l < 40; ++l) {
    // //         std::cout << L(k, l) << ',';
    // //     }
    // //     std::cout << '\n'
    // //               << std::endl;
    // // }
    // // std::cout << "_______________________________________" << std::endl;
    // // for (int k = 0; k < 20; ++k) {
    // //     for (int l = 0; l < 20; ++l) {
    // //         std::cout << U(k, l) << ',';
    // //     }
    // //     std::cout << '\n'
    // //               << std::endl;
    // // }
    // // std::cout << "_______________________________________" << std::endl;
    // // std::cout << normFrob(L) << ',' << normFrob(U) << std::endl;
    // // std::cout << "erreur sur LU : " << normFrob(L * U - reference) << std::endl;

    // // // HMatrix  builder Lh et Uh
    // // double eta = 10;
    // HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builderL(root_poisson, root_poisson, epsilon, eta, 'N', 'N');
    // HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builderU(root_poisson, root_poisson, epsilon, eta, 'N', 'N');
    // MatGenerator<double, double> lh(L, *root_poisson, *root_poisson);
    // MatGenerator<double, double> uh(U, *root_poisson, *root_poisson);

    // auto lperm = lh.get_unperm_mat();
    // auto uperm = uh.get_unperm_mat();
    // MatGenerator<double, double> lll(lperm, *root_poisson, *root_poisson);
    // MatGenerator<double, double> uuu(uperm, *root_poisson, *root_poisson);
    // auto Lh = hmatrix_tree_builderL.build(lll);
    // auto Uh = hmatrix_tree_builderU.build(uuu);

    // Matrix<double> ll(root_poisson->get_size(), root_poisson->get_size());
    // Matrix<double> uu(root_poisson->get_size(), root_poisson->get_size());
    // copy_to_dense(Lh, ll.data());
    // copy_to_dense(Uh, uu.data());
    // std::cout << "erreur sur LU : " << normFrob(L * U - reference);
    // std::cout << "compression Lh" << Lh.get_compression() << std::endl;
    // std::cout << " compression Uh " << Uh.get_compression() << std::endl;

    // std::cout << "erreur sur Lh " << normFrob(ll - L) / normFrob(L) << std::endl;
    // std::cout << "erreur Uh " << normFrob(U - uu) / normFrob(U) << std::endl;

    // std::vector<double> x_sol(root_poisson->get_size(), 1.0);
    // std::vector<double> x_solu(root_poisson->get_size(), 1.0);
    // std::vector<double> yl(root_poisson->get_size(), 0.0);
    // std::vector<double> yu(root_poisson->get_size(), 0.0);
    // Lh.add_vector_product('N', 1.0, x_sol.data(), 0.0, yl.data());
    // Uh.add_vector_product('N', 1.0, x_solu.data(), 0.0, yu.data());
    // std::vector<double> x_u(root_poisson->get_size(), 0.0);
    // std::vector<double> x_l(root_poisson->get_size(), 0.0);
    // Lh.forward_substitution_extract(*root_poisson, yl, x_l);
    // Uh.backward_substitution_extract(root_poisson->get_size(), root_poisson->get_offset(), yu, x_u);
    // std::cout << " erreur Lx=y" << norm2(x_sol - x_l) / norm2(x_sol) << std::endl;
    // std::cout << "erreur Ux = y" << norm2(x_solu - x_u) / norm2(x_solu) << std::endl;

    // // // // HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_21(root_cluster_2, root_cluster_1, epsilon, eta, 'N', 'N');

    // // ///////////////////////////////////////
    // // //// test LU x =y
    // // ///////////////////////////////////
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
