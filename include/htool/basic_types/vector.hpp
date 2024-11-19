#ifndef HTOOL_BASIC_TYPES_VECTOR_HPP
#define HTOOL_BASIC_TYPES_VECTOR_HPP

#include "../misc/misc.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace htool {
//================================//
//      DECLARATIONS DE TYPE      //
//================================//
typedef std::pair<int, int> Int2;

//================================//
//            VECTEUR             //
//================================//
template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
    if (!v.empty()) {
        out << '[';
        for (typename std::vector<T>::const_iterator i = v.begin(); i != v.end(); ++i)
            std::cout << *i << ',';
        out << "\b]";
    }
    return out;
}

template <typename T>
std::vector<T> operator+(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    std::vector<T> result(a.size(), 0);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<T>());

    return result;
}

template <typename T>
std::vector<T> plus(const std::vector<T> &a, const std::vector<T> &b) {

    return a + b;
}

template <typename T>
std::vector<T> operator-(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    std::vector<T> result(a.size(), 0);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<T>());

    return result;
}

template <typename T>
std::vector<T> minus(const std::vector<T> &a, const std::vector<T> &b) {

    return a - b;
}

template <typename T>
std::vector<T> mult(T value, const std::vector<T> &a) {
    std::vector<T> result(a.size(), 0);
    std::transform(a.begin(), a.end(), result.begin(), [value](T coef) { return coef * value; });

    return result;
}

template <typename T>
std::vector<T> mult(const std::vector<T> &b, T value) {
    return value * b;
}

template <typename T, typename V>
std::vector<T> operator/(const std::vector<T> &a, V value) {
    std::vector<T> result(a.size(), 0);
    std::transform(a.begin(), a.end(), result.begin(), [value](const T &c) { return c / value; });

    return result;
}

template <typename T>
T dprod(const std::vector<T> &a, const std::vector<T> &b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), T(0));
}
template <typename T>
std::complex<T> dprod(const std::vector<std::complex<T>> &a, const std::vector<std::complex<T>> &b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), std::complex<T>(0), std::plus<std::complex<T>>(), [](std::complex<T> u, std::complex<T> v) { return u * std::conj<T>(v); });
}

template <typename T>
underlying_type<T> norm2(const std::vector<T> &u) { return std::sqrt(std::abs(dprod(u, u))); }

template <typename T>
T max(const std::vector<T> &u) {
    return *std::max_element(u.begin(), u.end(), [](T a, T b) { return std::abs(a) < std::abs(b); });
}

template <typename T>
T min(const std::vector<T> &u) {
    return *std::min_element(u.begin(), u.end(), [](T a, T b) { return std::abs(a) < std::abs(b); });
}

template <typename T>
int argmax(const std::vector<T> &u) {
    return std::max_element(u.begin(), u.end(), [](T a, T b) { return std::abs(a) < std::abs(b); }) - u.begin();
}

template <typename T, typename V>
void operator*=(std::vector<T> &a, const V &value) {
    std::transform(a.begin(), a.end(), a.begin(), [value](T &c) { return c * value; });
}

template <typename T, typename V>
void operator/=(std::vector<T> &a, const V &value) {
    std::transform(a.begin(), a.end(), a.begin(), [value](T &c) { return c / value; });
}

template <typename T>
T mean(const std::vector<T> &u) {
    return std::accumulate(u.begin(), u.end(), T(0)) / T(u.size());
}

template <typename T>
int vector_to_bytes(const std::vector<T> vect, const std::string &file) {
    std::ofstream out(file, std::ios::out | std::ios::binary | std::ios::trunc);

    if (!out) {
        std::cout << "Cannot open file: " << file << std::endl; // LCOV_EXCL_LINE
        return 1;                                               // LCOV_EXCL_LINE
    }
    int size = vect.size();
    out.write((char *)(&size), sizeof(int));
    out.write((char *)&(vect[0]), size * sizeof(T));

    out.close();
    return 0;
}

template <typename T>
int bytes_to_vector(std::vector<T> &vect, const std::string &file) {

    std::ifstream in(file, std::ios::in | std::ios::binary);

    if (!in) {
        std::cout << "Cannot open file: " << file << std::endl; // LCOV_EXCL_LINE
        return 1;                                               // LCOV_EXCL_LINE
    }

    int size = 0;
    in.read((char *)(&size), sizeof(int));
    vect.resize(size);
    in.read((char *)&(vect[0]), size * sizeof(T));

    in.close();
    return 0;
}

// To be used with dlmread
template <typename T>
int matlab_save(std::vector<T> vector, const std::string &file) {
    std::ofstream out(file);
    out << std::setprecision(18);
    if (!out) {
        std::cout << "Cannot open file: " << file << std::endl;
        return 1;
    }

    // out<<rows<<" "<<cols<<std::endl;
    for (int i = 0; i < vector.size(); i++) {
        out << std::real(vector[i]);
        if (std::imag(vector[i]) < 0) {
            out << std::imag(vector[i]) << "i\t";
        } else if (std::imag(vector[i]) == 0) {
            out << "+" << 0 << "i\t";
        } else {
            out << "+" << std::imag(vector[i]) << "i\t";
        }
        out << std::endl;
    }
    out.close();
    return 0;
}

// std::vector<double> generateGaussianVector(int n) {
//     std::vector<double> gaussianVector;
//     gaussianVector.reserve(n);

//     // Générateur de nombres aléatoires
//     std::random_device rd;                          // Graine aléatoire
//     std::mt19937 gen(rd());                         // Générateur Mersenne Twister
//     std::uniform_real_distribution<> dis(0.0, 1.0); // Distribution uniforme entre 0 et 1

//     // Utilisation de la méthode Box-Muller
//     for (int i = 0; i < n / 2; ++i) {
//         double u1 = dis(gen);
//         double u2 = dis(gen);

//         // Transformation de Box-Muller
//         double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
//         double z1 = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);

//         // Ajouter z0 et z1 au vecteur
//         gaussianVector.push_back(z0);
//         gaussianVector.push_back(z1);
//     }

//     // Si n est impair, on génère un nombre supplémentaire
//     if (n % 2 == 1) {
//         double u1 = dis(gen);
//         double u2 = dis(gen);
//         double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
//         gaussianVector.push_back(z0);
//     }

//     return gaussianVector;
// }
std::vector<double> gaussian_vector(int n, double mean = 0.0, double std_dev = 1.0) {
    // Générateur de nombres aléatoires avec une seed basée sur le temps
    std::random_device rd;
    std::mt19937 generator(rd());

    // Distribution normale
    std::normal_distribution<double> distribution(mean, std_dev);

    // Remplir le vecteur avec des valeurs gaussiennes
    std::vector<double> vec(n);
    for (int i = 0; i < n; ++i) {
        vec[i] = distribution(generator);
    }

    return vec;
}

//================================//
//      CLASSE SUBVECTOR          //
//================================//

template <typename T>
class SubVec {

  private:
    std::vector<T> &U;
    const std::vector<int> &I;
    const int size;

  public:
    SubVec(std::vector<T> &&U0, const std::vector<int> &I0) : U(U0), I(I0), size(I0.size()) {}
    SubVec(const SubVec &); // Pas de constructeur par recopie

    T &operator[](const int &k) { return U[I[k]]; }
    const T &operator[](const int &k) const { return U[I[k]]; }

    template <typename RhsType>
    T operator,(const RhsType &rhs) const {
        T lhs = 0.;
        for (int k = 0; k < size; k++) {
            lhs += U[I[k]] * rhs[k];
        }
        return lhs;
    }

    int get_size() const { return this->size; }

    friend std::ostream &operator<<(std::ostream &os, const SubVec &u) {
        for (int j = 0; j < u.size; j++) {
            os << u[j] << "\t";
        }
        return os;
    }
};

} // namespace htool

#endif
