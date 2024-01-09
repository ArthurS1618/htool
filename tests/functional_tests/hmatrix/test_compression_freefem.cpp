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
            std::string word;
            std::vector<std::string> file_content;
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
    vector<double> points = to_mesh("/work/sauniear/Documents/matrice_test/Freefem/CD/mesh_CD3067.txt");

    std::cout << std::endl;
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
    ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> normal_build(points.size() / 3, 3, points.data(), 2, 2);
    std::shared_ptr<Cluster<double>> root_normal = make_shared<Cluster<double>>(normal_build.create_cluster_tree());

    // alternate build -----------------------------> Cluster pour notre méthode

    //////////////////// Méthode alternate
    std::vector<double> bb(3);
    bb[0] = 1;
    bb[1] = -1;
    std::vector<double> bperp(3);
    bperp[0] = 1;
    bperp[1] = 1;
    ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> alternate_build(points.size() / 3, 3, points.data(), 2, 2);
    std::shared_ptr<Cluster<double>> root_alternate = make_shared<Cluster<double>>(alternate_build.create_alternate_tree(bb, bperp));

    /////////////////// Méthode fixe direction
    // std::vector<double> bperp(3);
    // bperp[1] = 1;
    ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> directional_build(points.size() / 3, 3, points.data(), 2, 2);
    std::shared_ptr<Cluster<double>> root_directional = make_shared<Cluster<double>>(directional_build.create_directional_tree(bperp));

    ///////////////////
    /// save cluster
    ///////////////

    // save_clustered_geometry(*root_normal, 3, points.data(), "b11_geometry_normal_clustering", depth);
    save_clustered_geometry(*root_alternate, 3, points.data(), "b11_geomerty_alternate_clustering", depth);
    // save_clustered_geometry(*root_directional, 3, points.data(), "b11_geomerty_directional_clustering", depth);

    ///////
    /// RECUPERATION MATRICE
    //////////
    Matgenerator<double, double> m(size, size, *root_normal, *root_normal, "/work/sauniear/Documents/matrice_test/Freefem/CD/Vertices_3067_b10_eps6.txt");

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
    double epsilon = 0.0000001;

    Matgenerator<double, double> mat(LU, *root_normal, *root_normal);
    Matgenerator<double, double> matd(LU, *root_directional, *root_directional);
    Matgenerator<double, double> mata(LU, *root_alternate, *root_alternate);

    // /////////////////////:
    // // Tests pour grandes matrices
    // //////////////////////
    // // normal build -----------------------> Matrices de reference
    // double epsilon0 = 0.001;
    // double eta      = 10.0;
    double epsilon0 = 0.000001;
    double eta      = 100.0;

    HMatrixTreeBuilder<double, double> hmatrix_normal_builder(root_normal, root_normal, epsilon0, eta, 'N', 'N');
    auto hmatrix_normal(hmatrix_normal_builder.build(mat));
    std::cout << "normal build ok" << std::endl;

    // // directional builder -------------------------> Avec notre condition d'adissibilité et tree builder
    // double epsilon1 = 0.0005;
    // double eta1     = 13.0;
    double epsilon1 = 0.000001;
    double eta1     = 100.0;
    HMatrixTreeBuilder<double, double> hmatrix_directional_builder(root_directional, root_directional, epsilon1, eta1, 'N', 'N');
    std::shared_ptr<Bande<double>> directional_admissibility = std::make_shared<Bande<double>>(bb, root_directional->get_permutation());
    hmatrix_directional_builder.set_admissibility_condition(directional_admissibility);
    auto hmatrix_directional(hmatrix_directional_builder.build(matd));
    std::cout << "directional build ok" << std::endl;

    // // Alternate Builder
    double epsilon2 = 0.0000001;
    HMatrixTreeBuilder<double, double> hmatrix_alternate_builder(root_alternate, root_alternate, epsilon0, eta, 'N', 'N');
    std::shared_ptr<Bande<double>> alternate_admissibility = std::make_shared<Bande<double>>(bb, root_alternate->get_permutation());
    hmatrix_alternate_builder.set_admissibility_condition(alternate_admissibility);
    auto hmatrix_alternate(hmatrix_alternate_builder.build(mata));
    std::cout << "aternate build ok" << std::endl;

    Matrix<double> normal_dense(size, size);
    copy_to_dense(hmatrix_normal, normal_dense.data());
    auto nd = get_unperm_mat(normal_dense, hmatrix_normal.get_target_cluster(), hmatrix_normal.get_source_cluster());

    Matrix<double> directional_dense(size, size);
    copy_to_dense(hmatrix_directional, directional_dense.data());
    auto dd = get_unperm_mat(directional_dense, hmatrix_directional.get_target_cluster(), hmatrix_directional.get_source_cluster());

    Matrix<double> alternate_dense(size, size);
    copy_to_dense(hmatrix_alternate, alternate_dense.data());
    auto ad = get_unperm_mat(alternate_dense, hmatrix_alternate.get_target_cluster(), hmatrix_alternate.get_source_cluster());

    std::cout << "_________________________________" << std::endl;
    std::cout << "info normal " << std::endl;
    std::cout << "comrpession : " << hmatrix_normal.get_compression() << std::endl;
    std::cout << "erreur : " << normFrob(LU - nd) / norm << std::endl;
    std::cout << "rank info :  " << hmatrix_normal.get_rank_info() << std::endl;
    std::cout << "_________________________________" << std::endl;

    std::cout << "info directional " << std::endl;
    std::cout << "compression : " << hmatrix_directional.get_compression() << std::endl;
    std::cout << "erreur : " << normFrob(dd - LU) / norm << std::endl;
    std::cout << "rank info :" << hmatrix_directional.get_rank_info() << std::endl;
    std::cout << "_________________________________" << std::endl;

    std::cout << "info alternate " << std::endl;
    std::cout << "compression : " << hmatrix_alternate.get_compression() << std::endl;
    std::cout << "erreur : " << normFrob(ad - LU) / norm << std::endl;
    std::cout << "rank info :" << hmatrix_alternate.get_rank_info() << std::endl;
    std::cout << "_________________________________" << std::endl;

    ///////////////////////
    ///// save hmat
    //////////////////////
    // hmatrix_normal.save_plot("hmatrix7_b11_eps6_normal_3067");
    // hmatrix_directional.save_plot("hmatrix7_b11_eps6_directional_3067");
    // hmatrix_alternate.save_plot("hmatrix6_b11_eps1_alternate_3067");
    std::cout << "save hmatrix done" << std::endl;

    ///////////////////
    /// save cluster
    ///////////////

    // save_clustered_geometry(*root_normal, 3, points.data(), "b11_geometry_normal_clustering", depth);
    // save_clustered_geometry(*root_alternate, 3, points.data(), "b11_geomerty_alternate_clustering", depth);
    // save_clustered_geometry(*root_directional, 3, points.data(), "b11_geomerty_directional_clustering", depth);

    ////////////////////
    //// tests svd des blocs extradiagonaux

    // auto perm         = matd.get_perm_mat();
    // auto &sons        = hmatrix_directional.get_children();
    // string outputname = "svd_eps2_b11_directional";
    // int rep           = 0;
    // for (auto &s : sons) {
    //     std::cout << "hey" << std::endl;
    //     int nr = s->get_target_cluster().get_size();
    //     int nc = s->get_source_cluster().get_size();
    //     std::cout << nr << ',' << nc << std::endl;
    //     Matrix<double> matt(nr, nc);

    //     for (int k = 0; k < nr; ++k) {
    //         for (int l = 0; l < nc; ++l) {
    //             matt(k, l) = perm(k + s->get_target_cluster().get_offset(), l + s->get_source_cluster().get_offset());
    //         }
    //     }
    //     // copy_to_dense(*s, matt.data());
    // int lda  = nr;
    // int ldu  = nr;
    // int ldvt = nc;
    // lwork    = -1;
    // int infooo;
    // std::vector<double> singular_values(std::min(nr, nc));
    // Matrix<double> u(nr, nr);
    // // std::vector<T> vt (n*n);
    // Matrix<double> vt(nc, nc);
    // std::vector<double> workk(std::min(nc, nr));
    // std::vector<double> rwork(5 * std::min(nr, nc));
    // std::cout << "pour " << rep << " norme du bloc= " << normFrob(matt) << std::endl;

    // Lapack<double>::gesvd("A", "A", &nr, &nc, matt.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, workk.data(), &lwork, rwork.data(), &infooo);
    //     lwork = (int)std::real(work[0]);
    //     workk.resize(lwork);
    //     Lapack<double>::gesvd("A", "A", &nr, &nc, matt.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, workk.data(), &lwork, rwork.data(), &infooo);
    //     std::cout << "svd  ________________________________________" << std::endl;
    //     std::cout << std::endl;
    //     std::cout << "write svd" << std::endl;
    //     std::ofstream outputfile((outputname + '_' + std::to_string(rep) + ".csv").c_str());

    //     if (outputfile) {
    //         for (int k = 0; k < singular_values.size(); ++k)
    //             outputfile << singular_values[k] << std::endl;
    //     }
    //     outputfile.close();
    //     rep += 1;
    // }
    // std::cout << "svd written" << std::endl;

    // ///////////////////////////
    // /// Test sur la svd du bloc en bas a droite pour expliquer la diff entre tol = e-6 et tol = e-7
    // ///////////////////////////

    // auto &B1 = hmatrix_directional.get_children()[3]->get_children()[3]->get_children()[3]->get_children()[1];
    // auto &t1 = B1->get_target_cluster();
    // auto &s1 = B1->get_source_cluster();
    // std::cout << "t1 : " << t1.get_size() << ',' << t1.get_offset() << std::endl;
    // std::cout << "s1:" << s1.get_size() << ',' << s1.get_offset() << std::endl;
    // Matrix<double> up(t1.get_size(), s1.get_size());
    // Matrix<double> down(s1.get_size(), t1.get_offset());
    // for (int k = 0; k < t1.get_size(); ++k) {
    //     for (int l = 0; l < s1.get_size(); ++l) {
    //         up(k, l)   = perm(k + t1.get_offset(), l + s1.get_offset());
    //         down(l, k) = perm(l + s1.get_offset(), k + t1.get_offset());
    //     }
    // }
    // int nr   = t1.get_size();
    // int nc   = s1.get_size();
    // int lda  = nr;
    // int ldu  = nr;
    // int ldvt = nc;
    // lwork    = -1;
    // int infooo;
    // std::vector<double> singular_values(std::min(nr, nc));
    // Matrix<double> u(nr, nr);
    // // std::vector<T> vt (n*n);
    // Matrix<double> vt(nc, nc);
    // std::vector<double> workk(std::min(nc, nr));
    // std::vector<double> rwork(5 * std::min(nr, nc));

    // Lapack<double>::gesvd("A", "A", &nr, &nc, up.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, workk.data(), &lwork, rwork.data(), &infooo);
    // lwork = (int)std::real(work[0]);
    // workk.resize(lwork);
    // Lapack<double>::gesvd("A", "A", &nr, &nc, up.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, workk.data(), &lwork, rwork.data(), &infooo);
    // std::cout << "svd  ________________________________________" << std::endl;
    // std::cout << std::endl;
    // std::cout << "write svd" << std::endl;
    // std::ofstream outputfile1("svd_eps3_up.csv");

    // if (outputfile1) {
    //     for (int k = 0; k < singular_values.size(); ++k)
    //         outputfile1 << singular_values[k] << std::endl;
    // }
    // outputfile1.close();
    // std::cout << "svd up ok" << std::endl;

    // nr    = s1.get_size();
    // nc    = t1.get_size();
    // lda   = nr;
    // ldu   = nr;
    // ldvt  = nc;
    // lwork = -1;
    // // int infooo;
    // // std::vector<double> singular_values(std::min(nr, nc));
    // Matrix<double> ud(nr, nr);
    // // std::vector<T> vt (n*n);
    // Matrix<double> vtd(nc, nc);
    // std::vector<double> workkk(std::min(nc, nr));
    // std::vector<double> rworkk(5 * std::min(nr, nc));

    // Lapack<double>::gesvd("A", "A", &nr, &nc, down.data(), &lda, singular_values.data(), ud.data(), &ldu, vtd.data(), &ldvt, workkk.data(), &lwork, rworkk.data(), &infooo);
    // lwork = (int)std::real(work[0]);
    // workkk.resize(lwork);
    // Lapack<double>::gesvd("A", "A", &nr, &nc, down.data(), &lda, singular_values.data(), ud.data(), &ldu, vtd.data(), &ldvt, workkk.data(), &lwork, rworkk.data(), &infooo);
    // std::cout << "svd  ________________________________________" << std::endl;
    // std::cout << std::endl;
    // std::cout << "write svd" << std::endl;
    // std::ofstream outputfile2("svd_eps3_down.csv");

    // if (outputfile2) {
    //     for (int k = 0; k < singular_values.size(); ++k)
    //         outputfile2 << singular_values[k] << std::endl;
    // }
    // outputfile2.close();
    // std::cout << "done " << std::endl;

    //////////////////foramtage Hmatrix//////////

    // Matrix<double> L(size, size);
    // Matrix<double> U(size, size);
    // for (int k = 0; k < size; ++k) {
    //     L(k, k) = 1;
    //     U(k, k) = LU(k, k);
    //     for (int l = 0; l < k; ++l) {
    //         L(k, l) = LU(k, l);
    //         U(l, k) = LU(l, k);
    //     }
    // }
    // std::cout << "erreur LU :" << normFrob(L * U - M) / normFrob(M) << std::endl;
    // /// inverse de l matrice avec L et U
    // Matrix<double> Um(size, size);
    // Matrix<double> Lm(size, size);

    // for (int k = 0; k < size; ++k) {
    //     Lm(k, k) = 1;
    //     for (int i = k + 1; i < size; ++i) {
    //         for (int j = k; j < i; ++j) {
    //             Lm(i, k) += -L(i, j) * Lm(j, k);
    //         }
    //     }
    // }
    // Matrix<double> id(size, size);
    // for (int k = 0; k < size; ++k) {
    //     id(k, k) = 1;
    // }
    // std::cout << "!!!!!!!!!!!!!" << std::endl;
    // auto ll = L * Lm;
    // for (int k = 0; k < 10; ++k) {
    //     for (int l = 0; l < 10; ++l) {
    //         std::cout << ll(k, l) << ',';
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "erreur inverse" << normFrob(ll - id) / sqrt(size) << std::endl;
    // std::cout << "erreur LU " << normFrob(L * U - M) / normFrob(M) << std::endl;
    // std::cout << "ipiv " << std::endl;
    // double test = 0;
    // for (int k = 0; k < size; ++k) {
    //     test = std::max(test, std::abs(ipiv[k] - k - 1) * 1.0);
    // }
    // std::cout << test << std::endl;
    //////////
    /// ASSEMBLAGE HMATRICES
    ///////////

    ////
    /// test pour bien gerer de le passage avec les permutations....
    ///////////
    // double epsilon = -1.0;
    // int eta        = 10;
    // HMatrixTreeBuilder<double, double> hmatrix_m_builder(root_normal, root_normal, epsilon, eta, 'N', 'N');
    // HMatrixTreeBuilder<double, double> hmatrix_L_builder(root_normal, root_normal, epsilon, eta, 'N', 'N');
    // HMatrixTreeBuilder<double, double> hmatrix_U_builder(root_normal, root_normal, epsilon, eta, 'N', 'N');
    // // HMatrixTreeBuilder<double, double> hmatrix_tree_builder_U(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');

    // hmatrix_m_builder.set_eta(eta);
    // hmatrix_m_builder.set_rk(5);
    // Matgenerator<double, double> gen_L(L, *root_normal, *root_normal);
    // auto H(hmatrix_m_builder.build(gen_test));
    // Matrix<double> mm(size, size);
    // copy_to_dense(H, mm.data());
    // auto unperm = get_unperm_mat(mm, H.get_target_cluster(), H.get_source_cluster());
    // std::cout << "epsilon " << H.get_epsilon() << std::endl;
    // std ::cout << "error " << normFrob(LU - unperm) << std::endl;
    // std::cout << " compression " << H.get_compression() << std::endl;

    // auto Lh(hmatrix_L_builder.build(gen_L));
    // // auto H(hmatrix_m_builder.build(m));

    // std::cout << " info Lh " << std::endl;

    // Matrix<double> Lh_dense(size, size);
    // copy_to_dense(Lh, Lh_dense.data());

    // double cl = Lh.get_compression();
    // std::cout << "compression " << cl << std::endl;
    // auto l_perm = get_unperm_mat(Lh_dense, Lh.get_target_cluster(), Lh.get_source_cluster());
    // std::cout << "erreur " << normFrob(L - l_perm) / normFrob(L) << std::endl;

    // std::cout << "Uh info " << std::endl;
    // Matrix<double> Uh_dense(size, size);
    // copy_to_dense(Uh, Uh_dense.data());
    // double cu   = Uh.get_compression();
    // auto u_perm = get_unperm_mat(Uh_dense, Uh.get_target_cluster(), Uh.get_source_cluster());
    // std::cout << "compression " << cu << std::endl;
    // std::cout << "erreur " << normFrob(U - u_perm) / normFrob(U) << std::endl;

    // Matrix<double> ref(size, size);
    // copy_to_dense(H, ref.data());
    // Matrix<double> prod(size, size);
    // copy_to_dense(Lh.hmatrix_product(Uh), prod.data());
    // std::cout << normFrob(prod - M) / normFrob(M) << std::endl;

    //////////////////////////////////////////
    // vector<double> points = to_mesh("/work/sauniear/Documents/matrice_test/FreeFem Data/Regulier/Vertices4396/mesh_reg_vertices4396.txt");
    // ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> normal_build(points.size() / 3, 3, points.data(), 2, 2);
    // std::shared_ptr<Cluster<double>> root_normal = make_shared<Cluster<double>>(normal_build.create_cluster_tree());
    // MatGenerator<double, double> m(4396, 4396, *root_normal, *root_normal, "/work/sauniear/Documents/matrice_test/FreeFem Data/Regulier/Vertices4396/inv_Vertices4396_reg_b10_esp0.txt");
    // Matrix<double> M = m.get_mat();
    // for (int k = 0; k < 10; ++k) {
    //     std::cout << M(1, k) << std::endl;
    // }

    ///////////////////////:
    // Tests pour grandes matrices
    ////////////////////////
    // double eta            = 1.0;
    // double epsilon        = 0.0001;
    // string filename       = "/work/sauniear/Documents/matrice_test/FreeFem Data/Regulier/Vertices12035/inv_Vertices12035_reg_b10_esp0.txt";
    // vector<double> points = to_mesh("/work/sauniear/Documents/matrice_test/FreeFem Data/Regulier/Vertices12035/mesh_reg_vertices12035.txt");
    // std::cout << points.size() / 3 << endl;
    // // elliptique
    // ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> normal_build(points.size() / 3, 3, points.data(), 2, 2);
    // std::shared_ptr<Cluster<double>> root_normal = make_shared<Cluster<double>>(normal_build.create_cluster_tree());
    // MatGenerator<double, double> M10(12035, 12035, *root_normal, *root_normal, filename);
    //  HMatrixTreeBuilder<double, double> hmatrix_normal_builder(root_normal, root_normal, epsilon, eta, 'N', 'N');
    //  auto hmatrix_el(hmatrix_normal_builder.build(M10));

    // // directionel
    // ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> directional_build(points.size() / 3, 3, points.data(), 2, 2);
    // HMatrixTreeBuilder<double, double> hmatrix_directional_builder(root_normal, root_normal, epsilon, eta, 'N', 'N');
    // std::vector<double> b10(3, 0.0);
    // b10[0]                                                         = 1;
    // std::shared_ptr<Directional<double>> directional_admissibility = std::make_shared<Directional<double>>(b10);
    // hmatrix_directional_builder.set_admissibility_condition(directional_admissibility);
    // auto hmatrix_dir(hmatrix_directional_builder.build(M10));

    // Matrix<double> reference = M10.get_perm_mat();
    // Matrix<double> elliptique;
    // Matrix<double> directional;
    // copy_to_dense(hmatrix_el, elliptique.data());
    // copy_to_dense(hmatrix_dir, directional.data());
    // auto rk_el  = hmatrix_el.get_rank_info();
    // auto rk_dir = hmatrix_dir.get_rank_info();
    // std::cout << "error" << std::endl;
    // std::cout << "elliptique " << normFrob(elliptique - reference) / normFrob(reference) << std::endl;
    // std::cout << "directional " << normFrob(directional - reference) / normFrob(reference) << std::endl;
    // std::cout << "compression" << std::endl;
    // std::cout << "elliptique " << hmatrix_el.get_compression() << std::endl;
    // std::cout << "directional " << hmatrix_dir.get_compression() << std::endl;
    // std::cout << " rank " << std::endl;
    // std::cout << rk_el[0] << ',' << rk_el[1] << ',' << rk_el[2] << std::endl;
    // std::cout << rk_dir[0] << ',' << rk_dir[1] << ',' << rk_dir[2] << std::endl;

    //     // hmatrix_normal_builder.set_admissibility_condition(directional_admissibility10);
    //     b10[0] = 1;
    //     MatGenerator<double, double> M10(1970, 1970, *root_normal, *root_normal, file_name10);

    // /// TESTS POUR ANI
    // std::vector<double> error10;
    // std::vector<double> error11;
    // std::vector<double> compression10;
    // std::vector<double> compression11;
    // std::vector<std::vector<double>> rank10;
    // std::vector<std::vector<double>> rank11;

    // std::vector<double> errorheavy;
    // std::vector<double> compressionheavy;
    // std::vector<std::vector<double>> rankheavy;

    // double eta     = 10;
    // double epsilon = 0.0001;

    // string path10 = "/work/sauniear/Documents/matrice_test/FreeFem Data/Regulier/Vertices1970/inv_Vertices1970_reg_b10_esp";
    // string path11 = "/work/sauniear/Documents/matrice_test/FreeFem Data/Regulier/Vertices1970/inv_Vertices1970_reg_b11_esp";
    // // int pow_eps           = 0;
    // // string eps            = to_string(pow_eps);
    // // string file_name10    = path10 + eps + ".txt";
    // vector<double> points = to_mesh("/work/sauniear/Documents/matrice_test/FreeFem Data/Regulier/Vertices1970/mesh_reg_vertices1970.txt");
    // // ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> normal_build(points.size() / 3, 3, points.data(), 2, 2);
    // // std::shared_ptr<Cluster<double>> root_normal = make_shared<Cluster<double>>(normal_build.create_cluster_tree());
    // // MatGenerator<double, double> M(root_normal->get_size(), root_normal->get_size(), *root_normal, *root_normal, file_name10);
    // // HMatrixTreeBuilder<double, double> hmatrix_builder(root_normal, root_normal, epsilon, eta, 'N', 'N');
    // // auto hmatrix(hmatrix_builder.build(M));
    // // Matrix<double> reference = M.get_perm_mat();
    // // Matrix<double> test(root_normal->get_size(), root_normal->get_size());
    // // copy_to_dense(hmatrix, test.data());
    // // std::cout << "error" << std::endl;
    // // std::cout << normFrob(test - reference) / normFrob(reference) << std::endl;

    // // TEST DIRECTIONAL PLUS
    // for (int k = 0; k < 1; ++k) {
    //     int pow_eps        = k;
    //     string eps         = to_string(pow_eps);
    //     string file_name10 = path10 + eps + ".txt";
    //     string file_name11 = path11 + eps + ".txt";
    //     std::vector<double> b10(3, 0.0);
    //     b10[0] = 1;
    //     std::vector<double> b11(3, 0.0);
    //     b11[0] = 1;
    //     b11[1] = 1;
    //     // ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> normal_build(points.size() / 3, 3, points.data(), 2, 2);

    //     ClusterTreeBuilder<double, dumb_direction_10<double>, RegularSplitting<double>> normal_build_10(points.size() / 3, 3, points.data(), 2, 2);
    //     ClusterTreeBuilder<double, dumb_direction_11<double>, RegularSplitting<double>> normal_build_11(points.size() / 3, 3, points.data(), 2, 2);

    //     // ClusterTreeBuilder<double, dir, RegularSplitting<double>> normal_build(points.size() / 3, 3, points.data(), factory.create(), 2, 2);
    //     //  ClusterTreeBuilder<double, dumb_direction<double>(b10), RegularSplitting<double>> normal_build(points.size() / 3, 3, points.data(), 2, 2);
    //     std::shared_ptr<Cluster<double>> root_normal_10 = make_shared<Cluster<double>>(normal_build_10.create_cluster_tree());
    //     std::shared_ptr<Cluster<double>> root_normal_11 = make_shared<Cluster<double>>(normal_build_11.create_cluster_tree());
    //     MatGenerator<double, double> M10(root_normal_10->get_size(), root_normal_10->get_size(), *root_normal_10, *root_normal_10, file_name10);
    //     MatGenerator<double, double> M11(root_normal_11->get_size(), root_normal_11->get_size(), *root_normal_11, *root_normal_11, file_name11);

    //     // auto mm = M10.get_mat();
    //     // for (int k = 0; k < 25; ++k) {
    //     //     std::cout << mm(1, k) << ',';
    //     // }
    //     // std::cout << std::endl;

    //     HMatrixTreeBuilder<double, double> hmatrix_builder_10(root_normal_10, root_normal_10, epsilon, eta, 'N', 'N');
    //     HMatrixTreeBuilder<double, double> hmatrix_builder_11(root_normal_11, root_normal_11, epsilon, eta, 'N', 'N');

    //     // ICI ON CHOISIT LA CONDITION D'ADMISSIBILITE
    //     // std::shared_ptr<directional_plus<double>> directional_10 = std::make_shared<directional_plus<double>>(b10);
    //     // std::shared_ptr<directional_plus<double>> directional_11 = std::make_shared<directional_plus<double>>(b11);
    //     std::cout << "Admissibility" << std::endl;
    //     std::shared_ptr<Convection_Admissibility<double>> directional_10 = std::make_shared<Convection_Admissibility<double>>(b10, points);
    //     std::shared_ptr<Convection_Admissibility<double>> directional_11 = std::make_shared<Convection_Admissibility<double>>(b11, points);

    //     // build H10
    //     std::cout << "build H10" << std::endl;
    //     hmatrix_builder_10.set_admissibility_condition(directional_10);
    //     auto H10(hmatrix_builder_10.build(M10));
    //     std::cout << "H10 assembled" << std::endl;

    //     // reference
    //     Matrix<double> reference10 = M10.get_perm_mat();
    //     Matrix<double> test10(root_normal_10->get_size(), root_normal_10->get_size());
    //     copy_to_dense(H10, test10.data());

    //     double er10              = normFrob(test10 - reference10) / normFrob(reference10);
    //     double compr10           = H10.get_compression();
    //     std::vector<double> rk10 = H10.get_rank_info();
    //     // build H11
    //     hmatrix_builder_11.set_admissibility_condition(directional_11);
    //     std::cout << "build H11" << std::endl;
    //     auto H11(hmatrix_builder_11.build(M11));
    //     std::cout << "H11 assembled " << std::endl;

    //     Matrix<double> reference11 = M11.get_perm_mat();
    //     Matrix<double> test11(root_normal_11->get_size(), root_normal_11->get_size());
    //     copy_to_dense(H11, test11.data());
    //     double er11              = normFrob(test11 - reference11) / normFrob(reference11);
    //     double compr11           = H11.get_compression();
    //     std::vector<double> rk11 = H11.get_rank_info();

    //     error10.push_back(er10);
    //     compression10.push_back(compr10);
    //     rank10.push_back(rk10);
    //     error11.push_back(er11);
    //     compression11.push_back(compr11);
    //     rank11.push_back(rk11);
    // }
    // std::cout << "info 10" << std::endl;
    // std::cout << "error : " << std::endl;
    // for (int k = 0; k < error10.size(); ++k) {
    //     std::cout << error10[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "compression : " << std::endl;
    // for (int k = 0; k < compression10.size(); ++k) {
    //     std::cout << compression10[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "rank : " << std::endl;
    // for (int k = 0; k < rank10.size(); ++k) {
    //     auto rk = rank10[k];
    //     std::cout << '[' << rk[0] << ',' << rk[1] << ',' << rk[2] << ']' << ',';
    // }

    // std::cout << "info 11" << std::endl;
    // std::cout << "error : " << std::endl;
    // for (int k = 0; k < error11.size(); ++k) {
    //     std::cout << error11[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "compression : " << std::endl;
    // for (int k = 0; k < compression11.size(); ++k) {
    //     std::cout << compression11[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "rank : " << std::endl;
    // for (int k = 0; k < rank11.size(); ++k) {
    //     auto rk = rank11[k];
    //     std::cout << '[' << rk[0] << ',' << rk[1] << ',' << rk[2] << ']' << ',';
    // }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Hmult
    // HMatrix<double, double> L = hmatrix.hmatrix_product(hmatrix);
    // for (int kk = 0; kk < 6; ++kk) {

    //     ///// tests pour les générateurs et la lecture du maillage
    //     string rep         = std::to_string(kk);
    //     string file_name10 = path10 + rep + ".txt";
    //     string file_name11 = path11 + rep + ".txt";
    //     ////////////////////
    //     // GEOMETRY
    //     // Clustering
    //     vector<double> points = to_mesh("/work/sauniear/Documents/matrice_test/FreeFem Data/Regulier/Vertices1970/mesh_reg_vertices1970.txt");

    //     // Elliptique
    //     // ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> normal_build(points.size() / 3, 3, points.data(), 2, 2);
    //     //  Directional
    //     ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> normal_build(points.size() / 3, 3, points.data(), 2, 2);
    //     std::shared_ptr<Cluster<double>> root_normal = make_shared<Cluster<double>>(normal_build.create_cluster_tree());

    //     // Assemblage hmatrices
    //     //  tree builder -> block_cluster tree
    //     HMatrixTreeBuilder<double, double> hmatrix_normal_builder(root_normal, root_normal, epsilon, eta, 'N', 'N');

    //     // b = 1,0
    //     std::vector<double> b10(3, 0.0);
    //     b10[0] = 1;
    //     MatGenerator<double, double> M10(root_normal->get_size(), root_normal->get_size(), *root_normal, *root_normal, file_name10);
    //     // // // Pour condition directionnelle
    //     // // std::shared_ptr<Directional<double>> directional_admissibility10 = std::make_shared<Directional<double>>(b10);
    //     // // hmatrix_normal_builder.set_admissibility_condition(directional_admissibility10);
    //     // // ///////////////
    //     // // Pour condition hybride
    //     // // std::shared_ptr<Hybride<double>> hybride_admissibility10 = std::make_shared<Hybride<double>>(b10);
    //     // // hmatrix_normal_builder.set_admissibility_condition(hybride_admissibility10);

    //     // // Pour condition directionnell plus
    //     std::shared_ptr<directional_plus<double>> directionplus = std::make_shared<directional_plus<double>>(b10);
    //     hmatrix_normal_builder.set_admissibility_condition(directionplus);
    //     ///////////////
    //     auto hmatrix_10(hmatrix_normal_builder.build(M10));
    //     Matrix<double> reference10 = M10.get_perm_mat();
    //     Matrix<double> test10(root_normal->get_size(), root_normal->get_size());
    //     copy_to_dense(hmatrix_10, test10.data());

    //     // std::shared_ptr<directional_heavy<double>> directionheavy = std::make_shared<directional_heavy<double>>(b10);
    //     // hmatrix_normal_builder.set_admissibility_condition(directionplus);
    //     // auto hmatrix_heavy(hmatrix_normal_builder.build(M10));
    //     // Matrix<double> referenceheavy = M10.get_perm_mat();
    //     // Matrix<double> testheavy(root_normal->get_size(), root_normal->get_size());
    //     // copy_to_dense(hmatrix_heavy, testheavy.data());

    //     // b = 1,1
    //     // std::vector<double> b11(3, 0.0);
    //     // b11[0] = 1;
    //     // b11[1] = 1;
    //     // MatGenerator<double, double> M11(1970, 1970, *root_normal, *root_normal, file_name11);
    //     // // // Por condition directionnelle
    //     // // std::shared_ptr<Directional<double>> directional_admissibility11 = std::make_shared<Directional<double>>(b11);
    //     // // hmatrix_normal_builder.set_admissibility_condition(directional_admissibility10);
    //     // // //////////////////////////////
    //     // // Pour condition hybride
    //     // // std::shared_ptr<Hybride<double>> hybride_admissibility11 = std::make_shared<Hybride<double>>(b11);
    //     // // hmatrix_normal_builder.set_admissibility_condition(hybride_admissibility11);
    //     // ///////////////
    //     // // Pour condition directionnelle plus :
    //     // std::shared_ptr<directional_plus<double>> direction11 = std::make_shared<directional_plus<double>>(b11);
    //     // hmatrix_normal_builder.set_admissibility_condition(direction11);
    //     // auto hmatrix_11(hmatrix_normal_builder.build(M11));
    //     // Matrix<double> reference11 = M11.get_perm_mat();
    //     // Matrix<double> test11(root_normal->get_size(), root_normal->get_size());
    //     // copy_to_dense(hmatrix_11, test11.data());

    //     // Data 10
    //     double er10    = normFrob(test10 - reference10) / normFrob(reference10);
    //     double compr10 = hmatrix_10.get_compression();
    //     auto rk10      = hmatrix_10.get_rank_info();
    //     error10.push_back(er10);
    //     compression10.push_back(compr10);
    //     rank10.push_back(rk10);

    //     // double erheavy    = normFrob(test10);
    //     // double comprheavy = hmatrix_heavy.get_compression();
    //     // auto rrk          = hmatrix_heavy.get_rank_info();
    //     // errorheavy.push_back(erheavy);
    //     // compressionheavy.push_back(comprheavy);
    //     // rankheavy.push_back(rrk);

    //     // // Data 11
    //     // double er11    = normFrob(test11 - reference11) / normFrob(reference11);
    //     // double compr11 = hmatrix_11.get_compression();
    //     // auto rk11      = hmatrix_11.get_rank_info();

    //     // error11.push_back(er11);
    //     // compression11.push_back(compr11);
    //     // rank11.push_back(rk11);
    // }
    // std::cout << "info 10" << std::endl;
    // std::cout << "error : " << std::endl;
    // for (int k = 0; k < error10.size(); ++k) {
    //     std::cout << error10[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "compression : " << std::endl;
    // for (int k = 0; k < compression10.size(); ++k) {
    //     std::cout << compression10[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "rank : " << std::endl;
    // for (int k = 0; k < rank10.size(); ++k) {
    //     auto rk = rank10[k];
    //     std::cout << '[' << rk[0] << ',' << rk[1] << ',' << rk[2] << ']' << ',';
    // }

    // std::cout << "info heavy" << std::endl;
    // std::cout << "error : " << std::endl;
    // for (int k = 0; k < errorheavy.size(); ++k) {
    //     std::cout << errorheavy[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "compression : " << std::endl;
    // for (int k = 0; k < compressionheavy.size(); ++k) {
    //     std::cout << compressionheavy[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "rank : " << std::endl;
    // for (int k = 0; k < rankheavy.size(); ++k) {
    //     auto rk = rankheavy[k];
    //     std::cout << '[' << rk[0] << ',' << rk[1] << ',' << rk[2] << ']' << ',';
    // }

    // std::cout << std::endl;
    // std::cout << std::endl;
    // std::cout << "info 11" << std::endl;
    // std::cout << "error : " << std::endl;
    // for (int k = 0; k < error11.size(); ++k) {
    //     std::cout << error11[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "compression : " << std::endl;
    // for (int k = 0; k < compression11.size(); ++k) {
    //     std::cout << compression11[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "rank : " << std::endl;
    // for (int k = 0; k < rank11.size(); ++k) {
    //     auto rk = rank11[k];
    //     std::cout << '[' << rk[0] << ',' << rk[1] << ',' << rk[2] << ']' << ',';
    // }
    ////////////////////////////////////////////////////////////////////////:
    // FIN TEST BOUCLES /
    ////////////////////////////////////////////////////////

    // POUBELLE

    // Matrix<double> test_normal(root_normal->get_size(), root_normal->get_size());
    // copy_to_dense(normal_hmatrix, test_normal.data());
    // std::cout << "error : " << normFrob(test_m - test_normal) / normFrob(test_m) << std::endl;
    // std::cout << "compression :" << normal_hmatrix.get_compression() << std::endl;

    // save_clustered_geometry(*root_normal, 3, points.data(), "geometry1", std::vector<int>{1, 2, 3});

    // // Assemblage matrices conditions directionnelle
    // ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> directional_build(points.size() / 3, 3, points.data(), 2, 2);
    // std::shared_ptr<Cluster<double>> root_directional = make_shared<Cluster<double>>(directional_build.create_cluster_tree());
    // HMatrixTreeBuilder<double, double> hmatrix_directional_builder(root_directional, root_directional, 0.0001, 1, 'N', 'N');
    // std::shared_ptr<Directional<double>> directional_admissibility = std::make_shared<Directional<double>>(b);
    // hmatrix_directional_builder.set_admissibility_condition(directional_admissibility);

    // // Assemblage matrices conditions directionnelle + elliptique
    // ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> hybride_build(points.size() / 3, 3, points.data(), 2, 2);
    // std::shared_ptr<Cluster<double>> root_hybride = make_shared<Cluster<double>>(hybride_build.create_cluster_tree());
    // HMatrixTreeBuilder<double, double> hmatrix_hybride_builder(root_hybride, root_hybride, 0.0001, 1, 'N', 'N');
    // std::shared_ptr<Hybride<double>> hybride_admissibility = std::make_shared<Hybride<double>>(b);
    // hmatrix_hybride_builder.set_admissibility_condition(hybride_admissibility);

    // Build
    // auto normal_hmatrix(hmatrix_normal_builder.build(test));
    // auto directional_hmatrix(hmatrix_directional_builder.build(test));
    // auto hybride_hmatrix(hmatrix_hybride_builder.build(test));

    // On les met en full pour tester l'erreur
    // Matrix<double> test_normal(root_normal->get_size(), root_normal->get_size());
    // Matrix<double> test_directional(root_directional->get_size(), root_directional->get_size());
    // Matrix<double> test_hybride(root_hybride->get_size(), root_hybride->get_size());
    // copy_to_dense(normal_hmatrix, test_normal.data());
    // copy_to_dense(directional_hmatrix, test_directional.data());
    // copy_to_dense(hybride_hmatrix, test_hybride.data());

    // auto rk_normal      = normal_hmatrix.get_rank_info();
    // auto rk_directional = directional_hmatrix.get_rank_info();
    // auto rk_hybride     = hybride_hmatrix.get_rank_info();
    // std::cout << "epsilon = " << epsilon << std::endl;
    // std::cout << "erreur en norme de Frobenius : " << std::endl;
    // std::cout << " normal splitting : " << normFrob(test_normal - test_m) / normFrob(test_m) << std::endl;
    // std::cout << "directional splitting " << normFrob(test_directional - test_m) / normFrob(test_m) << std::endl;
    // std::cout << " hybride splitting" << normFrob(test_hybride - test_m) / normFrob(test_m) << std::endl;
    // std::cout << "compression " << std::endl;
    // std::cout << "normal splitting " << normal_hmatrix.get_compression() << std::endl;
    // std::cout << "directional splitting" << directional_hmatrix.get_compression() << std::endl;
    // std::cout << "hybride splitting " << hybride_hmatrix.get_compression() << std::endl;
    // std::cout << "Rank" << std::endl;
    // std::cout << "elliptique " << rk_normal[0] << ',' << rk_normal[1] << ',' << rk_normal[2];
    // std::cout << "diretional " << rk_directional[0] << ',' << rk_directional[1] << ',' << rk_directional[2] << std::endl;
    // std::cout << "Hybride " << rk_hybride[0] << ',' << rk_hybride[1] << ',' << rk_hybride[2] << std::endl;
    //}
    // Test pour comparer splitting normal au splitting directionnel

    return 0;
}
