#!/bin/bash
#SBATCH --job-name=out_heat
#SBATCH --output=%x.o%j
#SBATCH --time=01:00:00 # (see available partitions)
#SBATCH --partition=gpua100   
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

# To clean and load modules defined at the compile and link phases
module purge
module load gcc/11.2.0/gcc-4.8.5 hdf5/1.10.7/gcc-11.2.0-openmpi openmpi/4.1.1/gcc-11.2.0 cuda/11.7.0/gcc-11.2.0 cmake/3.21.4/gcc-11.2.0
. /gpfs/users/bourgeoisr/expe_heat_equation/heat_equation/vendor/install_pdi/share/pdi/env.bash 

# echo of commands
set -x

# To compute in the submission directory
cd ${SLURM_SUBMIT_DIR}

# execution
pdirun srun ./my_app