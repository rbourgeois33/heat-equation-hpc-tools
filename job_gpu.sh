#!/bin/bash
#SBATCH --job-name=sgpu
#SBATCH --output=%x.o%j
#SBATCH --exclusive
#SBATCH --time=01:00:00 # (see available partitions)
#SBATCH --partition=gpu   
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1

# To clean and load modules defined at the compile and link phases
module purge
module load gcc/11.2.0/gcc-4.8.5 hdf5/1.10.7/gcc-11.2.0-openmpi openmpi/4.1.1/gcc-11.2.0 cuda/11.7.0/gcc-11.2.0 cmake/3.21.4/gcc-11.2.0
. /gpfs/users/bourgeoisr/expe_heat_equation/heat_equation/vendor/install_pdi/share/pdi/env.bash 

# echo of commands
set -x

# To compute in the submission directory
cd ${SLURM_SUBMIT_DIR}

# execution
pdirun -n 2 ./my_app