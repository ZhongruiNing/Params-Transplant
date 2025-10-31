#!/bin/bash
#SBATCH -J Grid_Params_Trans
#SBATCH -p cp6
#SBATCH -N 7
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=2
#SBATCH -o %J.out
#SBATCH -e %J.err

module purge
module load MPI/openmpi/4.1.2-mpi-x-gcc12.2
module load ucx/1.14.0-th-mt
export LD_LIBRARY_PATH=/fs2/software/openmpi/4.1.2-mpi-x-gcc12.2/lib/:$LD_LIBRARY_PATH
export OMPI_MCA_pml=ob1

source ~/.bashrc
conda activate zhongrui

srun python3 -u Grids_Params_Transplant_PS_MPI.py
# python3 -u Merge_Grids_Params_Transplant_PS_Results_MPI.py
