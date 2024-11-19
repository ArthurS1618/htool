add_executable(Test_air_quality test_air_quality.cpp)
target_link_libraries(Test_air_quality htool)
add_dependencies(build-tests-hmatrix-air-quality Test_air_quality)
add_test(Test_air_quality Test_air_quality)


add_executable(Test_freefem test_freefem.cpp)
target_link_libraries(Test_freefem htool)
add_dependencies(build-tests-hmatrix-air-quality Test_freefem)
add_test(Test_freefem Test_freefem)

add_executable(Test_redressement test_redressement.cpp)
target_link_libraries(Test_redressement htool)
add_dependencies(build-tests-hmatrix-air-quality Test_redressement)
add_test(Test_redressement Test_redressement)

add_executable(Test_obstacle test_redressement_interpolation.cpp)
target_link_libraries(Test_obstacle htool)
add_dependencies(build-tests-hmatrix-air-quality Test_obstacle)
add_test(Test_obstacle Test_obstacle)