add_custom_target(build-tests-lrmat)
add_dependencies(build-tests build-tests-lrmat)
add_subdirectory(lrmat)

add_custom_target(build-tests-hmatrix-build)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-build)
add_subdirectory(hmatrix_build)

add_custom_target(build-tests-hmatrix-product)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-product)
add_subdirectory(hmatrix_product)

add_custom_target(build-tests-hmatrix-addition)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-addition)
add_subdirectory(hmatrix_addition)

add_custom_target(build-tests-hmatrix-triangular-solve)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-triangular-solve)
add_subdirectory(hmatrix_triangular_solve)

# add_executable(Test_LU test_lu.cpp)
# target_link_libraries(Test_LU htool)
# add_test(Test_LU Test_LU)

# add_executable(Perf_Hmatrix perf_hmatrix.cpp)
# target_link_libraries(Perf_Hmatrix htool)
# add_test(Perf_Hmatrix Perf_Hmatrix)
# add_executable(Test_LHU test_HLU.cpp)
# target_link_libraries(Test_HLU htool)
# add_test(Test_HLU Test_HLU)

add_custom_target(build-tests-hmatrix-factorization)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-factorization)
add_subdirectory(hmatrix_factorization)

add_custom_target(build-tests-hmatrix-air-quality)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-air-quality)
add_subdirectory(test_ifp)

add_custom_target(build-tests-sum-expression)
add_dependencies(build-tests-hmatrix build-tests-sum-expression)
add_subdirectory(sum_expression)


# add_executable(Test_HMULT test_hmult.cpp)
# target_link_libraries(Test_HMULT htool)
# add_test(Test_HMULT Test_HMULT)
