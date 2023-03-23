add_custom_target(build-tests-lrmat)
add_dependencies(build-tests build-tests-lrmat)
add_subdirectory(lrmat)

add_custom_target(build-tests-sum-expression)
add_dependencies(build-tests build-tests-sum-expression)
add_subdirectory(sum_expression)

add_custom_target(build-tests-hmatrix-build)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-build)
add_subdirectory(hmatrix_build)

add_custom_target(build-tests-hmatrix-product)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-product)
add_subdirectory(hmatrix_product)

add_custom_target(build-tests-hmatrix-hmatrix-product)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-hmatrix-product)
add_subdirectory(hmatrix_hmatrix_product)
