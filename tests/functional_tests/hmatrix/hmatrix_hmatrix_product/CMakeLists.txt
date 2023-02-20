#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

#=== lrmat_SVD

add_executable(Test_hmatrix_hmatrix_product_double test_hmatrix_hmatrix_product_double.cpp)
target_link_libraries(Test_hmatrix_hmatrix_product_double htool)
add_dependencies(build-tests-hmatrix-product Test_hmatrix_hmatrix_product_double)
add_test(Test_hmatrix_hmatrix_product_double Test_hmatrix_hmatrix_product_double)
