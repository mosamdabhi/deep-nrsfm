subject=23
root="../datasets"
if [ ! -d "$root" ]; then
  mkdir "$root"
fi
root+="/$subject"
if [ ! -d "$root" ]; then
  mkdir "$root"
fi

nice -n 10 python3 train.py \
--gpu=1 \
--model_dir=$root \
--save_checkpoints_steps=1000 \
--keep_checkpoint_max=0 \
--max_steps=200000 \
--subject=$subject \
--buffer_size=1000 \
--throttle_secs=60 \
--hparams="num_atoms=135,num_points=21,learning_rate=0.001,loss_norm=2,
batch_size=80,num_dictionaries=20,num_atoms_bottleneck=10"
