cmake_minimum_required(VERSION 3.5)

if($ENV{TRAVIS_BUILD_NUMBER})
    project(DlibClustering VERSION 0.0.$ENV{TRAVIS_BUILD_NUMBER})
else()
    project(DlibClustering)
endif()

find_package (Threads REQUIRED)
find_package(BLAS REQUIRED)

add_executable(rclst multiclass_classification_ex.cpp classi.h)
add_executable(rclss user.cpp classi.h)

set_target_properties(rclst PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        )

target_link_libraries(rclst
        dlib
        blas
        lapack
        ${CMAKE_THREAD_LIBS_INIT}
        )

set_target_properties(rclss PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        )

target_link_libraries(rclss
        dlib
        blas
        lapack
        )

install(TARGETS rclst RUNTIME DESTINATION bin)
install(TARGETS rclss RUNTIME DESTINATION bin)
INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/dataset.csv DESTINATION bin)
set(CPACK_GENERATOR DEB)

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

set(CPACK_PACKAGE_CONTACT lismut@gmail.com)

include(CPack)