#ifndef HTOOL_LAPACK_HPP
#define HTOOL_LAPACK_HPP

#include "../misc/define.hpp"

#if defined(__powerpc__) || defined(INTEL_MKL_VERSION)
#    define HTOOL_LAPACK_F77(func) func
#else
#    define HTOOL_LAPACK_F77(func) func##_
#endif

#define HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(C, T, B, U)                                                                                                                            \
    void HTOOL_LAPACK_F77(B##gesvd)(const char *, const char *, const int *, const int *, U *, const int *, U *, U *, const int *, U *, const int *, U *, const int *, int *);      \
    void HTOOL_LAPACK_F77(C##gesvd)(const char *, const char *, const int *, const int *, T *, const int *, U *, T *, const int *, T *, const int *, T *, const int *, U *, int *); \
    void HTOOL_LAPACK_F77(B##getrf)(const int *, const int *, U *, const int *, int *, int *);                                                                                      \
    void HTOOL_LAPACK_F77(B##getri)(const int *, U *, const int *, int *, U *, const int *, int *);                                                                                 \
    void HTOOL_LAPACK_F77(B##laswp)(const int *, U *, const int *, const int *, const int *, int *, const int *);

#if !defined(PETSC_HAVE_BLASLAPACK)
#    ifndef _MKL_H_
#        ifdef __cplusplus
extern "C" {
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(z, std::complex<double>, d, double)
}
#        else
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(c, void, s, float)
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(z, void, d, double)
#        endif // __cplusplus
#    endif     // _MKL_H_
#endif

#ifdef __cplusplus
namespace htool {
/* Class: Lapack
 *
 *  A class that wraps some LAPACK routines for dense linear algebra.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template <class K>
struct Lapack {
    /* Function: gesvd
     *  computes the singular value decomposition (SVD). */
    static void gesvd(const char *, const char *, const int *, const int *, K *, const int *, underlying_type<K> *, K *, const int *, K *, const int *, K *, const int *, underlying_type<K> *, int *);
    static void getrf(const int *, const int *, K *, const int *, int *, int *);
    static void getri(const int *, K *, const int *, int *, K *, const int *, int *);
    static void laswp(const int *, K *, const int *, const int *, const int *, int *, const int *);
    // static void stpsv(const char *, const char *, const char *, const int *, K *, K *, const int *);
};

#    define HTOOL_GENERATE_LAPACK_COMPLEX(C, T, B, U)                                                                                                                                                                             \
        template <>                                                                                                                                                                                                               \
        inline void Lapack<U>::gesvd(const char *jobu, const char *jobvt, const int *m, const int *n, U *a, const int *lda, U *s, U *u, const int *ldu, U *vt, const int *ldvt, U *work, const int *lwork, U *, int *info) {      \
            HTOOL_LAPACK_F77(B##gesvd)                                                                                                                                                                                            \
            (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);                                                                                                                                                  \
        }                                                                                                                                                                                                                         \
        template <>                                                                                                                                                                                                               \
        inline void Lapack<T>::gesvd(const char *jobu, const char *jobvt, const int *m, const int *n, T *a, const int *lda, U *s, T *u, const int *ldu, T *vt, const int *ldvt, T *work, const int *lwork, U *rwork, int *info) { \
            HTOOL_LAPACK_F77(C##gesvd)                                                                                                                                                                                            \
            (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);                                                                                                                                           \
        }                                                                                                                                                                                                                         \
        template <>                                                                                                                                                                                                               \
        inline void Lapack<U>::getrf(const int *m, const int *n, U *a, const int *lda, int *ipiv, int *info) {                                                                                                                    \
            HTOOL_LAPACK_F77(B##getrf)                                                                                                                                                                                            \
            (m, n, a, lda, ipiv, info);                                                                                                                                                                                           \
        }                                                                                                                                                                                                                         \
        template <>                                                                                                                                                                                                               \
        inline void Lapack<U>::getri(const int *n, U *a, const int *lda, int *ipiv, U *work, const int *lwork, int *info) {                                                                                                       \
            HTOOL_LAPACK_F77(B##getri)                                                                                                                                                                                            \
            (n, a, lda, ipiv, work, lwork, info);                                                                                                                                                                                 \
        }                                                                                                                                                                                                                         \
        template <>                                                                                                                                                                                                               \
        inline void Lapack<U>::laswp(const int *n, U *a, const int *lda, const int *k1, const int *k2, int *ipiv, const int *incx) {                                                                                              \
            HTOOL_LAPACK_F77(B##laswp)                                                                                                                                                                                            \
            (n, a, lda, k1, k2, ipiv, incx);                                                                                                                                                                                      \
        }                                                                                                                                                                                                                         \
        // template <>                                                                                                                                                                                                               \
        // inline void Lapack<U>::stpsv(const char *uplo, const char *trans, const char *diag, const int *n, U *AP, U *x, const int *incx) {                                                                                         \
        //     HTOOL_LAPACK_F77(B##stpsv)                                                                                                                                                                                            \
        //     (uplo, trans, diag, n, AP, x, incx);                                                                                                                                                                                  \
        // }
HTOOL_GENERATE_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_LAPACK_COMPLEX(z, std::complex<double>, d, double)
} // namespace htool
#endif // __cplusplus
#endif // HTOOL_LAPACK_HPP
