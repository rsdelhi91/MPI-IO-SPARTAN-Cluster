#!/bin/bash
#SBATCH --job-name="cloud"
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
module load Python/3.4.3-goolf-2015a
#SBATCH --mem-per-cpu=15000mb
#SBATCH --core-per-task=1
#SBATCH --mca shmem_mmap_enable_nfs_warning=0
mpirun -np 8 python mpigeneric.py
