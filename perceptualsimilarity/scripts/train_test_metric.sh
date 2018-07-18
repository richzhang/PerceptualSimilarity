
NET=alex
python ./train.py --use_gpu --net ${NET} --name ${NET}
python ./test_dataset_model.py --use_gpu --net ${NET} --model_path ./checkpoints/${NET}/latest_net_.pth

