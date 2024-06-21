#include <htool/clustering/cluster_output.hpp>
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>
#include <htool/wrappers/wrapper_lapack.hpp>

#include <mpi.h>
#include <regex>

using namespace std;
using namespace htool;

template <class CoefficientPrecision, class CoordinatePrecision>
Matrix<CoefficientPrecision> get_unperm_mat(const Matrix<CoefficientPrecision> mat, const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source) {
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
//////////////////////////
//// GENERATOR -> Prend Une matrice ou un fichier
//////////////////////////
template <class CoefficientPrecision, class CoordinatePrecision>
class Matgenerator : public VirtualGenerator<CoefficientPrecision> {
  private:
    Matrix<CoefficientPrecision> mat;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;

  public:
    Matgenerator(Matrix<CoefficientPrecision> &mat0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : mat(mat0), target(target0), source(source0){};

    Matgenerator(int nr, int nc, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0, const string &matfile) : target(target0), source(source0) {
        ifstream file;
        file.open(matfile);
        CoefficientPrecision *data = new CoefficientPrecision[nr * nc];
        std::vector<CoefficientPrecision> file_content;
        if (!file) {
            std::cerr << "Error : Can't open the file" << std::endl;
        } else {
            int count = 0;
            // while (file >> word) {
            //     file_content.push_back(word);
            //     count++;
            // }
            // for (int i = 0; i < count; i++) {
            //     string str = file_content[i];
            //     CoefficientPrecision d1;
            //     stringstream stream1;
            //     stream1 << str;
            //     stream1 >> d1;
            //     // data[i] = d1;
            //     int i0             = i / nr;
            //     int j0             = i - i0 * nr;
            //     data[j0 * nr + i0] = d1;
            // }
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

/////////////////////////////
//// MESH
/////////////////////////////

// #include <iostream>
// #include <fstream>
// #include <string>
// #include <vector>
// #include <sstream>

// vector<double> to_mesh(const string &file_name) {
//     ifstream inputFile(file_name);
//     if (!inputFile.is_open()) {
//         std::cerr << "Erreur lors de l'ouverture du fichier." << std::endl;
//     }

//     std::vector<double> allCoordinates;
//     std::string line;

//     bool insideVerticesSection = false;
//     bool firstLine = true;

//     while (std::getline(inputFile, line)) {
//         if (line.find("Vertices") != std::string::npos) {
//             insideVerticesSection = true;
//             continue;
//         }

//         if (insideVerticesSection && line.empty()) {
//             insideVerticesSection = false;
//             break;
//         }

//         if (insideVerticesSection) {
//             std::istringstream stream(line);
//             float value;

//             // Lire et ignorer la première coordonnée
//             if (firstLine) {
//                 stream >> value;
//                 firstLine = false;
//             }

//             // Lire et stocker les coordonnées sauf la dernière
//             while (stream >> value) {
//                 allCoordinates.push_back(value);
//             }

//             // Ignorer la dernière coordonnée en lisant, mais ne pas stocker
//             stream >> value;
//         }
//     }
//     inputFile.close();
//     return allCoordinates ;
// }

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
int main() {
    ///////////////////////////////////////////////
    // test LU
    ///////////////////////////

    //////
    /// RECUPERATION CLUSTER
    ///////////
    std::vector<double> compr_norm, compr_dir, err_norm, err_dir;
    for (int eps = 0; eps < 7; ++eps) {
        // vector<double> points = to_mesh("/work/sauniear/Documents/matrice_test/Freefem/CD/mesh_CD3067.txt");
        vector<double> points = to_mesh("/work/sauniear/Documents/matrice_test/Freefem/CD_hole/size_3227_b10/mesh_withhole_CD3227.txt");

        std::cout << "eps = " << eps << std::endl;
        std::cout << "taille du maillage : " << points.size() / 3 << std::endl;
        int size = points.size() / 3;
        /////
        /// CLUSTER TREE
        ///////
        std::vector<int> depth(7);
        for (int k = 1; k < 8; ++k) {
            depth[k] = k;
        }
        // normal build ------------------> Cluster pour matrice de reference , regular splitting + PCA
        ClusterTreeBuilder<double> recursive_build_strategy;
        recursive_build_strategy.set_minclustersize(20);
        std::shared_ptr<Cluster<double>> root_normal = std::make_shared<Cluster<double>>(recursive_build_strategy.create_cluster_tree(points.size() / 3, 3, points.data(), 2, 2));
        // alternate build -----------------------------> Cluster pour notre méthode

        //////////////////// Méthode alternate
        std::vector<double> bb(3);
        bb[0] = 1;
        bb[1] = 0;
        std::vector<double> bperp(3);
        bperp[0] = 0;
        bperp[1] = 1;
        ClusterTreeBuilder<double> directional_build;
        std::shared_ptr<ConstantDirection<double>> strategy_directional = std::make_shared<ConstantDirection<double>>(bperp);
        directional_build.set_direction_computation_strategy(strategy_directional);
        directional_build.set_minclustersize(20);
        std::shared_ptr<Cluster<double>> root_directional = std::make_shared<Cluster<double>>(directional_build.create_cluster_tree(points.size() / 3, 3, points.data(), 2, 2));

        ///////////////////
        /// save cluster
        ///////////////

        // save_clustered_geometry(*root_normal, 3, points.data(), "b11_geometry_normal_clustering", depth);
        // save_clustered_geometry(*root_alternate, 3, points.data(), "b11_geomerty_alternate_clustering", depth);
        // save_clustered_geometry(*root_directional, 3, points.data(), "b10_hole_geomerty_directional_clustering", depth);

        ///////
        /// RECUPERATION MATRICE
        //////////
        std::string epsString = std::to_string(eps);
        // Matgenerator<double, double> m(size, size, *root_normal, *root_normal, "/work/sauniear/Documents/matrice_test/Freefem/CD/Vertices_3067_b10_eps6.txt");
        Matgenerator<double, double> m(size, size, *root_normal, *root_normal, "/work/sauniear/Documents/matrice_test/Freefem/CD_hole/size_3227_b10/Vertices_3227_b10_withhole_eps" + epsString + ".txt");

        Matrix<double> LU = m.get_mat();
        auto M            = LU;

        std::cout << "matrice et mesh recupérée" << std::endl;
        /////
        /// LU
        ///////

        std::vector<int> ipiv(size, 0.0);
        int info = -1;
        Lapack<double>::getrf(&size, &size, LU.data(), &size, ipiv.data(), &info);
        std::cout << "LU ok " << std::endl;
        int infoo = -1;
        Matrix<double> id(size, size);
        Matrix<double> U(size, size);
        Matrix<double> L(size, size);
        // Demande de calculer la taille requise pour le travail
        for (int k = 0; k < size; ++k) {
            id(k, k) = 1;
        }
        for (int k = 0; k < size; ++k) {
            L(k, k) = 1;
            U(k, k) = LU(k, k);
            for (int l = 0; l < k; ++l) {
                L(k, l) = LU(k, l);
                U(l, k) = LU(l, k);
            }
        }

        std::cout << "erreur decomposition LU " << normFrob((L * U) - M) / normFrob(M) << std::endl;

        // Alloue le tableau de travail avec la taille requise
        std::vector<int> ipivo(size);
        int lwork = size * size;
        std::vector<double> work(lwork);
        // Effectue l'inversion
        Lapack<double>::getri(&size, LU.data(), &size, ipiv.data(), work.data(), &lwork, &info);
        std::cout << "inversion ok" << std::endl;

        auto p    = M * LU;
        auto norm = normFrob(LU);
        std::cout << "ereur sur l'inverse " << std::endl;
        std::cout << info << "," << normFrob(p - id) / sqrt(size) << std::endl;

        // H martrices
        double epsilon = 1e-7;

        Matgenerator<double, double> mat(LU, *root_normal, *root_normal);
        Matgenerator<double, double> matd(LU, *root_directional, *root_directional);
        // Matgenerator<double, double> mata(LU, *root_alternate, *root_alternate);

        // /////////////////////:
        // // Tests pour grandes matrices
        // //////////////////////
        // // normal build -----------------------> Matrices de reference
        // double epsilon0 = 0.001;
        // double eta      = 10.0;
        double epsilon0 = 1e-6;
        double eta      = 100.0;

        HMatrixTreeBuilder<double, double> hmatrix_normal_builder(*root_normal, *root_normal, epsilon, eta, 'N', 'N', -1, -1, -1);

        auto hmatrix_normal(hmatrix_normal_builder.build(mat));
        std::cout << "normal build ok" << std::endl;

        // // directional builder -------------------------> Avec notre condition d'adissibilité et tree builder
        // double epsilon1 = 0.0005;
        // double eta1     = 13.0;
        HMatrixTreeBuilder<double, double> hmatrix_directional_builder(*root_directional, *root_directional, epsilon, eta, 'N', 'N', -1, -1, -1);
        std::shared_ptr<Bande<double>> directional_admissibility = std::make_shared<Bande<double>>(bb, root_directional->get_permutation());
        hmatrix_directional_builder.set_admissibility_condition(directional_admissibility);
        auto hmatrix_directional(hmatrix_directional_builder.build(matd));
        std::cout << "directional build ok" << std::endl;

        // // Alternate Builder
        // double epsilon2 = 0.0000001;
        // HMatrixTreeBuilder<double, double> hmatrix_alternate_builder(root_alternate, root_alternate, epsilon0, eta, 'N', 'N');
        // std::shared_ptr<Bande<double>> alternate_admissibility = std::make_shared<Bande<double>>(bb, root_alternate->get_permutation());
        // hmatrix_alternate_builder.set_admissibility_condition(alternate_admissibility);
        // auto hmatrix_alternate(hmatrix_alternate_builder.build(mata));
        // std::cout << "aternate build ok" << std::endl;

        Matrix<double> normal_dense(size, size);
        copy_to_dense(hmatrix_normal, normal_dense.data());
        auto nd = get_unperm_mat(normal_dense, hmatrix_normal.get_target_cluster(), hmatrix_normal.get_source_cluster());

        Matrix<double> directional_dense(size, size);
        copy_to_dense(hmatrix_directional, directional_dense.data());
        auto dd = get_unperm_mat(directional_dense, hmatrix_directional.get_target_cluster(), hmatrix_directional.get_source_cluster());

        // Matrix<double> alternate_dense(size, size);
        // copy_to_dense(hmatrix_alternate, alternate_dense.data());
        // auto ad = get_unperm_mat(alternate_dense, hmatrix_alternate.get_target_cluster(), hmatrix_alternate.get_source_cluster());

        std::cout << "_________________________________" << std::endl;
        std::cout << "info normal " << std::endl;
        std::cout << "comrpession : " << hmatrix_normal.get_compression() << std::endl;
        std::cout << "erreur : " << normFrob(LU - nd) / norm << std::endl;
        std::cout << "_________________________________" << std::endl;

        std::cout << "info directional " << std::endl;
        std::cout << "compression : " << hmatrix_directional.get_compression() << std::endl;
        std::cout << "erreur : " << normFrob(dd - LU) / norm << std::endl;
        std::cout << "_________________________________" << std::endl;

        compr_dir.push_back(hmatrix_directional.get_compression());
        compr_norm.push_back(hmatrix_normal.get_compression());
        err_dir.push_back(normFrob(dd - LU) / norm);
        err_norm.push_back(normFrob(nd - LU) / norm);

        // std::cout << "info alternate " << std::endl;
        // std::cout << "compression : " << hmatrix_alternate.get_compression() << std::endl;
        // std::cout << "erreur : " << normFrob(ad - LU) / norm << std::endl;
        // std::cout << "rank info :" << hmatrix_alternate.get_rank_info() << std::endl;
        // std::cout << "_________________________________" << std::endl;

        ///////////////////////
        ///// save hmat
        //////////////////////
        // hmatrix_normal.save_plot("hmatrix7_b11_eps6_normal_3067");
        // hmatrix_directional.save_plot("hmatrix7_b11_eps6_directional_3067");
        // hmatrix_alternate.save_plot("hmatrix6_b11_eps1_alternate_3067");
    }
    std::cout << " normal " << std::endl;
    for (int k = 0; k < compr_dir.size(); ++k) {
        std::cout << err_norm[k] << ',';
    }
    std::cout << std::endl;

    for (int k = 0; k < compr_dir.size(); ++k) {
        std::cout << compr_norm[k] << ',';
    }
    std::cout << std::endl;
    std::cout << "directional " << std::endl;
    for (int k = 0; k < compr_dir.size(); ++k) {
        std::cout << err_dir[k] << ',';
    }
    std::cout << std::endl;

    for (int k = 0; k < compr_dir.size(); ++k) {
        std::cout << compr_dir[k] << ',';
    }
    std::cout << std::endl;
    std::cout << std::endl;
    return 0;
}