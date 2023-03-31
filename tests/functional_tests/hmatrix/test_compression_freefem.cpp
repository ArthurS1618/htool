#include <htool/clustering/cluster_output.hpp>
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>
#include <mpi.h>
#include <regex>

using namespace std;
using namespace htool;

//////////////////////////
//// GENERATOR -> Prend Une matrice ou un fichier
//////////////////////////
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
        if (!file) {
            std::cerr << "Error : Can't open the file" << std::endl;
        } else {
            std::string word;
            std::vector<std::string> file_content;
            int count = 0;
            while (file >> word) {
                file_content.push_back(word);
                count++;
            }
            for (int i = 0; i < count; i++) {
                string str = file_content[i];
                CoefficientPrecision d1;
                stringstream stream1;
                stream1 << str;
                stream1 >> d1;
                // data[i] = d1;
                int i0             = i / nr;
                int j0             = i - i0 * nr;
                data[j0 * nr + i0] = d1;
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
};

/////////////////////////////
//// MESH
/////////////////////////////

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
    ////////////////////////////////
    // pour test avec boucles
    //////////////////////////////////
    std::vector<double> error10;
    std::vector<double> error11;
    std::vector<double> compression10;
    std::vector<double> compression11;
    std::vector<std::vector<double>> rank10;
    std::vector<std::vector<double>> rank11;

    std::vector<double> errorheavy;
    std::vector<double> compressionheavy;
    std::vector<std::vector<double>> rankheavy;

    double eta     = 10;
    double epsilon = 0.0001;

    string path10         = "/work/sauniear/Documents/matrice_test/FreeFem Data/Regulier/Vertices1970/inv_Vertices1970_reg_b10_esp";
    string path11         = "/work/sauniear/Documents/matrice_test/FreeFem Data/Regulier/Vertices1970/inv_Vertices1970_reg_b11_esp";
    int pow_eps           = 0;
    string eps            = to_string(pow_eps);
    string file_name10    = path10 + eps + ".txt";
    vector<double> points = to_mesh("/work/sauniear/Documents/matrice_test/FreeFem Data/Regulier/Vertices1970/mesh_reg_vertices1970.txt");
    ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> normal_build(points.size() / 3, 3, points.data(), 2, 2);
    std::shared_ptr<Cluster<double>> root_normal = make_shared<Cluster<double>>(normal_build.create_cluster_tree());
    MatGenerator<double, double> M(root_normal->get_size(), root_normal->get_size(), *root_normal, *root_normal, file_name10);
    HMatrixTreeBuilder<double, double> hmatrix_builder(root_normal, root_normal, epsilon, eta, 'N', 'N');
    auto hmatrix(hmatrix_builder.build(M));
    Matrix<double> reference = M.get_perm_mat();
    Matrix<double> test(root_normal->get_size(), root_normal->get_size());
    copy_to_dense(hmatrix, test.data());
    std::cout << "error" << std::endl;
    std::cout << normFrob(test - reference) / normFrob(reference) << std::endl;

    // Hmult
    HMatrix<double, double> L = hmatrix.hmatrix_product(hmatrix);
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
