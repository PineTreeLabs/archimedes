cmake_minimum_required(VERSION 3.22)

# Define base directories
set(ARCHIMEDES_PROJECT_DIR "../../archimedes")

# Archimedes-generated project sources
set(ARCHIMEDES_Project_Src
    ${ARCHIMEDES_PROJECT_DIR}/cfilter.c
    ${ARCHIMEDES_PROJECT_DIR}/cfilter_kernel.c
)

# Archimedes include directories
set(ARCHIMEDES_Include_Dirs
    ${ARCHIMEDES_PROJECT_DIR}
)

# Create Archimedes library
add_library(archimedes STATIC)

# Add sources to the library
target_sources(archimedes PRIVATE
    ${ARCHIMEDES_Project_Src}
)

# Link the math library
target_link_libraries(archimedes PRIVATE m)

# Set include directories
target_include_directories(archimedes PUBLIC ${ARCHIMEDES_Include_Dirs})

# Make include directories available to parent scope
set(ARCHIMEDES_Include_Dirs ${ARCHIMEDES_Include_Dirs} PARENT_SCOPE)