now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs
data_dir=./data
use_persistent_dataset=True

mkdir -p $logdir

torchrun --master_port=20503 GF_RL_train.py \
    --data_dir $data_dir \
    --use_persistent_dataset $use_persistent_dataset \
    --logdir $logdir | tee $logdir/$now.txt