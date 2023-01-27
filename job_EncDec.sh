#!/bin/bash -l

#SBATCH --partition=a100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -A avenka14_gpu

###load modules
ml gcc/9.3.0
ml python/3.7.9
ml cuda/10.2.89
ml py-numpy/1.18.5
ml py-pip/20.2

cd $HOME/data-avenka14/ravi/fac-via-ppg

pip install --user -r fac_clone_reqs.txt
pip install --user torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user librosa
pip install --user tensorboardX==2.5.1
pip install --user inflect
pip install --user Unidecode
pip install --user matplotlib

python3 auto_segment.py