#!/bin/sh
#SBATCH --job-name="DLv3+ 512"
#SBATCH --partition=gpu-invest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:59:59
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/512_train.txt
#SBATCH --error=logs/512_train.txt
#SBATCH --account=ws_00000

export NOW=$( date '+%F-%H-%M-%S' )


echo "Copying training data to ${TMPDIR}..."
mkdir $TMPDIR/cityscapes_toy
cp /storage/workspaces/artorg_aimi/ws_00000/lukaszbinden/datasets/cityscapes_toy.tar.gz $TMPDIR/cityscapes_toy
## /storage/homefs/lz20w714/aimi_storage/lukaszbinden/datasets
## (ddpm) [lz20w714@submit02 datasets]$ tar -cvf cityscapes_toy.tar.gz cityscapes_toy/
tar -xf $TMPDIR/cityscapes_toy/cityscapes_toy.tar.gz -C $TMPDIR/.

python main.py --model deeplabv3plus_mobilenet --dataset cityscapes --gpu_id 0  --lr 0.1  --crop_size 512 --batch_size 32 --output_stride 16 --data_root ${TMPDIR}/cityscapes_toy/ --print_interval 500 --val_interval 5000 --total_itrs 150000 --save_val_results --val_max_size 500 


