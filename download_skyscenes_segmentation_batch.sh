#!/bin/bash
#PBS -m bea
#PBS -M stefano.savian.ext@leonardocompany.com
#PBS -l select=1:ncpus=4:ngpus=0
#PBS -j oe
#PBS -r n
#PBS -q workq

echo "pbs dir"
echo $PBS_O_WORKDIR
echo "bash dir"
echo $PWD
echo "cd to pbs dir"
cd $PBS_O_WORKDIR
echo $PWD

module load proxy/proxy_20
export http_proxy=http://10.17.20.110:8080/
export https_proxy=http://10.17.20.110:8080/
export ftp_proxy=http://10.17.20.110:8080/
export HTTP_PROXY=http://10.17.20.110:8080/
export HTTPS_PROXY=http://10.17.20.110:8080/
export FTP_PROXY=http://10.17.20.110:8080/

cd /davinci-1/home/ssavian/CODE/depth_benchmark/scripts

./download_skyscenes_segmentation.sh --path /davinci-1/home/ssavian/DATASETS/SkyScenes
