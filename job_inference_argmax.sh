#!/bin/bash -l

#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH -A avenka14_gpu

###load modules
ml gcc/9.3.0
ml python/3.7.9
ml cuda/10.2.89
ml py-numpy/1.18.5
ml py-pip/20.2

cd $HOME/data-avenka14/ravi/fac-ppg

pip install --user -r fac_clone_reqs.txt
pip install --user torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user librosa
pip install --user tensorboardX==2.5.1
pip install --user inflect
pip install --user Unidecode
pip install --user matplotlib

#python3 inference_block_pitch_rate_predictor_flip_counts_vesus.py angry
#python3 inference_block_pitch_rate_predictor_flip_counts_vesus.py happy
#python3 inference_block_pitch_rate_predictor_flip_counts_vesus.py sad
#python3 inference_block_pitch_rate_predictor_flip_counts_vesus.py fear

python3 inference_actor_critic_encoder_vesus_argmax.py angry
python3 inference_actor_critic_encoder_vesus_argmax.py happy
python3 inference_actor_critic_encoder_vesus_argmax.py sad
python3 inference_actor_critic_encoder_vesus_argmax.py fear
