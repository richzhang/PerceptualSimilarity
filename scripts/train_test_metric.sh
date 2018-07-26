
TRIAL=${1}
NET=${2}
python ./train.py --use_gpu --net ${NET} --name ${NET}_${TRIAL}
python ./test_dataset_model.py --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}/latest_net_.pth
