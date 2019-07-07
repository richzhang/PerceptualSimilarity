
TRIAL=${1}
NET=${2}
mkdir checkpoints
mkdir checkpoints/${NET}_${TRIAL}_scratch
python ./train.py --from_scratch --train_trunk --use_gpu --net ${NET} --name ${NET}_${TRIAL}_scratch
python ./test_dataset_model.py --from_scratch --train_trunk --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}_scratch/latest_net_.pth

