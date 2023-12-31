#!/bin/bash
#SBATCH --job-name=out_heat
#SBATCH --output=%x.o%j
#SBATCH --time=00:00:30 # (see available partitions)
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4

# To clean and load modules defined at the compile and link phases
module purge
module load gcc/11.2.0/gcc-4.8.5 hdf5/1.10.7/gcc-11.2.0-openmpi openmpi/4.1.1/gcc-11.2.0 cuda/11.7.0/gcc-11.2.0 cmake/3.21.4/gcc-11.2.0
. path_to_pdi_install/share/pdi/env.bash

# echo of commands
set -x

# To compute in the submission directory
cd ${SLURM_SUBMIT_DIR}

# execution
pdirun srun ./my_app