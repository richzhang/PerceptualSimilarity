
mkdir dataset

# JND Dataset
wget https://perceptual-similarity.s3.us-west-2.amazonaws.com/dataset/jnd.tar.gz -O ./dataset/jnd.tar.gz

mkdir dataset/jnd
tar -xf ./dataset/jnd.tar.gz -C ./dataset
rm ./dataset/jnd.tar.gz

# 2AFC Val set
mkdir dataset/2afc/
wget https://perceptual-similarity.s3.us-west-2.amazonaws.com/dataset/twoafc_val.tar.gz -O ./dataset/twoafc_val.tar.gz

mkdir dataset/2afc/val
tar -xf ./dataset/twoafc_val.tar.gz -C ./dataset/2afc
rm ./dataset/twoafc_val.tar.gz

# 2AFC Train set
mkdir dataset/2afc/
wget https://perceptual-similarity.s3.us-west-2.amazonaws.com/dataset/twoafc_train.tar.gz -O ./dataset/twoafc_train.tar.gz

mkdir dataset/2afc/train
tar -xf ./dataset/twoafc_train.tar.gz -C ./dataset/2afc
rm ./dataset/twoafc_train.tar.gz
