
mkdir dataset

# JND Dataset
wget https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset/jnd.tar.gz -O ./dataset/jnd.tar.gz

mkdir dataset/jnd
tar -xzf ./dataset/jnd.tar.gz -C ./dataset
rm ./dataset/jnd.tar.gz

# 2AFC Val set
mkdir dataset/2afc/
wget https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset/twoafc_val.tar.gz -O ./dataset/twoafc_val.tar.gz

mkdir dataset/2afc/val
tar -xzf ./dataset/twoafc_val.tar.gz -C ./dataset/2afc
rm ./dataset/twoafc_val.tar.gz
