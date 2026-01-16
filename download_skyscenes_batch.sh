#!/bin/bash
#PBS -m bea
#PBS -M stefano.savian.ext@leonardocompany.com
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -j oe  
#PBS -r n
#PBS -q gpu

#CUDA_VISIBLE_DEVICES=2
echo "pbs dir"
echo  $PBS_O_WORKDIR
echo "bash dir"
echo $PWD
echo "cd to pbs dir"
cd $PBS_O_WORKDIR
echo $PWD


module load proxy/proxy_20
PROXY_ENVS="http_proxy=http://10.17.20.110:8080/"
PROXY_ENVS="${PROXY_ENVS},https_proxy=http://10.17.20.110:8080/"
PROXY_ENVS="${PROXY_ENVS},ftp_proxy=http://10.17.20.110:8080/"
PROXY_ENVS="${PROXY_ENVS},HTTP_PROXY=http://10.17.20.110:8080/"
PROXY_ENVS="${PROXY_ENVS},HTTPS_PROXY=http://10.17.20.110:8080/"
PROXY_ENVS="${PROXY_ENVS},FTP_PROXY=http://10.17.20.110:8080/"
BINDS="/archive,/cm,/davinci-1,/etc/resolv.conf,/run"
#singularity shell --nv -B $BINDS --env PATH="\$PATH:$PATH",${PROXY_ENVS} container5
singularity shell --nv --writable  -B $BINDS --env PATH="\$PATH:$PATH",${PROXY_ENVS} container5
source /opt/conda/etc/profile.d/conda.sh

singularity_sif_path=/davinci-1/home/ssavian/container5
singularity_python=/opt/conda/envs/depth_benchmark/bin/python


cd /davinci-1/home/ssavian/CODE/depth_benchmark/scripts

 ./download_skyscenes.sh 
 ./extract_skyscenes.sh /davinci-1/home/ssavian/DATASETS/SkyScenes
