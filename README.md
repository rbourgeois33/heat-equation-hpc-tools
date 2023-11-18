## Guide adapted from nova++

# 0: Clone

* `git clone --recurse-submodules ssh://git@gitlab.erc-atmo.eu:30000/remi.bourgeois/heat_equation.git` `

# 1: Load libraries

* `module load gcc/11.2.0/gcc-4.8.5 hdf5/1.10.7/gcc-11.2.0-openmpi openmpi/4.1.1/gcc-11.2.0 cuda/11.7.0/gcc-11.2.0 cmake/3.21.4/gcc-11.2.0`

# 2: Load PDI, First compilation ever ? Compile PDI

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

# Else, just re-load PDI
* `. path_to_pdi_install/share/pdi/env.bash`
* `. /gpfs/users/bourgeoisr/expe_heat_equation/heat_equation/vendor/install_pdi/share/pdi/env.bash`

# 3: Create build folder
* `cd heat_equation/`
* `mkdir build`
* `cd build`

# 4: Configure cmake, ruche, A100

* `cmake -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON ..`

# Ruche, V100

* `cmake -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON ..`

# Ruche, P100

* `cmake -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL60=ON ..`

# Ruche, CPU, openMP

* `cmake -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_OPENMP=ON ..`

# 5: Compile code
* `make -j 16`

# 6: Run the code
* `Use the slurm scripts in heat_equation/ to launch jobs`


