# Heat Equation solved with HPC tools 
This miniapp is a simple use case of the Kokkos, MPI and PDI libraries for solving the linear heat equation on a 2D cartesian mesh with periodic boundary condition. In particular, it showcases the use of `Kokkos::parallel_for, Kokkos::parallel_reduce` and `PDI_MULTI_EXPOSE`in a MPI context.

- Kokkos (https://github.com/kokkos/kokkos)
- MPI (https://github.com/open-mpi)
- PDI (https://gitlab.maisondelasimulation.fr/pdidev/pdi)

Kokkos allows to write architecture agnostic kernels (one kernel can compile on both CPUs and GPUs). MPI is used to perform a standard cartesian domain decomposition shared among the processes. PDI is used to seamlessly couple of the MPI simulation code with the parallel hdf5 library to handle I/O.

- hdf5 (https://github.com/HDFGroup/hdf5)

# Parallel strategy

![](https://github.com/rbourgeois33/heat-equation-hpc-tools/gif.gif)
![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)

## How to get source

* `git clone --recurse-submodules https://github.com/rbourgeois33/heat-equation-hpc-tools.git`

## Necessary modules

* `module load gcc/11.2.0/gcc-4.8.5 hdf5/1.10.7/gcc-11.2.0-openmpi openmpi/4.1.1/gcc-11.2.0 cuda/11.7.0/gcc-11.2.0 cmake/3.21.4/gcc-11.2.0`

Note: Cuda is only necessary if compiling for Nvidia GPUs. For AMD GPUs, import HIP. For CPUs, no GPU library is needed.

## Load PDI
### First compilation ? Compile PDI

* `cd heat_equation/vendor/pdi`
* `mkdir build`
* `cd build`
```
cmake -DCMAKE_INSTALL_PREFIX=$PWD/../../install_pdi -DUSE_HDF5=SYSTEM -DBUILD_HDF5_PARALLEL=ON  -DUSE_yaml=EMBEDDED -DUSE_paraconf=EMBEDDED -DBUILD_SHARED_LIBS=ON -DBUILD_FORTRAN=OFF -DBUILD_BENCHMARKING=OFF -DBUILD_SET_VALUE_PLUGIN=OFF -DBUILD_TESTING=OFF -DBUILD_DECL_NETCDF_PLUGIN=OFF -DBUILD_USER_CODE_PLUGIN=OFF ..
```
* `make -j 16`
* `make install`
* `unset PDI_DIR`
* `. ../../install_pdi/share/pdi/env.bash`
* `cd heat_diffusion/`

### Else, just re-load PDI
* `. path_to_pdi_install/share/pdi/env.bash`

## Create build folder
* `cd heat_equation/`
* `mkdir build`
* `cd build`

## Configure cmake 

For a single CPU
* `cmake -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release ..`

For CPUs with OpenMP
* `cmake -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON ..`


For Nvidia GPUs
* `cmake -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_X=ON ..`

with 

 - `DKokkos_ARCH_X=DKokkos_ARCH_AMPERE80` for Nvidia A100
 - `DKokkos_ARCH_X=DKokkos_ARCH_VOLTA70` for Nvidia V100
 - `DKokkos_ARCH_X=DKokkos_ARCH_PASCAL60` for Nvidia P100

## Compile and run code
* `make -j 16`

Use the slurm scripts in heat_equation/ copied in the build folder and adapted to your computing center to launch jobs

## Plot the outputs
* `Use plotter.py from your build directory to generate visual outputs`

# File description
- `src/main.cpp` Initializes Kokkos, MPI and PDI instances and lauch the heat equation resolution.
- `src/heat_equation.cpp` Standart heat equation solver. Includes the parameters choice (physical and numerical), Kokkos kernels for initialisation, finite difference update and reduction as well as the writing of the solution with PDI.
- `src/mpi_decomposion.hpp` Class that contains all the MPI related routines: 2D cartesian decomposition, neighbors indexes and communications.
- `io.yml` Descriptive layer for PDI I/O.
