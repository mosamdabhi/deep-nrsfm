subject=23
step=784000
root="../datasets/$subject"
output="../datasets/results/"

nice -n 10 python3 evaluate.py \
--gpu=1 \
--model_dir=$root \
--subject=$subject \
--checkpoint=$step \
--error_metrics_dir="$output/$subject.csv" \
--predictions_dir="$output/$subject.npz" \
--hparams="num_atoms=135,num_points=21,learning_rate=0.001,loss_norm=2,
batch_size=80,num_dictionaries=20,num_atoms_bottleneck=10"
