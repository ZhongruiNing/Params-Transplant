#!/bin/bash
#SBATCH -J Params_trans
#SBATCH -p cp6
#SBATCH -N 10
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH -o Params_Trans_%J.out
#SBATCH -e Params_Trans_%J.err

module purge
module load MPI/openmpi/4.1.2-mpi-x-gcc12.2
module load ucx/1.14.0-th-mt
export LD_LIBRARY_PATH=/fs2/software/openmpi/4.1.2-mpi-x-gcc12.2/lib/:$LD_LIBRARY_PATH
export OMPI_MCA_pml=ob1

source ~/.bashrc
conda activate zhongrui
srun python3 -u Params_Transplant_MPI.py
python3 -u Merge_Params_Transplant_Results_MPI.py