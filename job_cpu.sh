#!/bin/bash
#SBATCH --job-name=out_heat
#SBATCH --output=%x.o%j
#SBATCH --time=00:00:30
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu_short     # (see available partitions)

# To clean and to load the same modules at the compilation phases
module purge
module load gcc/11.2.0/gcc-4.8.5 hdf5/1.10.7/gcc-11.2.0-openmpi openmpi/4.1.1/gcc-11.2.0 cmake/3.21.4/gcc-11.2.0
. path_to_pdi_install/share/pdi/env.bash

# echo of commands
set -x

# To compute in the submission directory
cd ${SLURM_SUBMIT_DIR}

# number of OpenMP threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} 
# Binding OpenMP Threads of each MPI process on cores
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# execution 
# with 'ntasks' MPI processes
# with 'cpus-per-task' OpenMP threads per MPI process
pdirun srun ./my_app
