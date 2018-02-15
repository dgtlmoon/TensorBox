#!/bin/bash

echo "Downloading..."
mkdir -p data/tshirtslayer
cd data

wget --continue http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
wget https://raw.githubusercontent.com/bethesirius/ChosunTruck/master/linux/tensorbox/data/inception_v1.ckpt

#mkdir -p overfeat_rezoom && cd overfeat_rezoom
#wget --continue http://russellsstewart.com/s/tensorbox/overfeat_rezoom/save.ckpt-150000v2
#cd ..


echo "Extracting..."
tar xf resnet_v1_101_2016_08_28.tar.gz

if [[ "$2" == '--load_experimental' ]]; then
    tar xf inception_resnet_v2_2016_08_30.tar.gz
    tar xf mobilenet_v1_1.0_224_2017_06_14.tar.gz
fi

cd ..

mkdir -p data/tshirtslayer/tss-images
# Get the TSS data sets
curl https://tshirtslayer.com/tss-lab/json/test_boxes.json > data/tshirtslayer/test_boxes.json
cat data/tshirtslayer/test_boxes.json|python ./fetch-list.py > fetch-list.txt

curl https://tshirtslayer.com/tss-lab/json/val_boxes.json > data/tshirtslayer/val_boxes.json
cat data/tshirtslayer/val_boxes.json|python ./fetch-list.py >> fetch-list.txt

curl https://tshirtslayer.com/tss-lab/json/train_boxes.json > data/tshirtslayer/train_boxes.json
cat data/tshirtslayer/train_boxes.json|python ./fetch-list.py >> fetch-list.txt

echo "downloading..."
cat fetch-list.txt | parallel

echo "Doing the resize to 640x480 :("
find  data/tshirtslayer/tss-images -type f |while read fname
do
  echo -n .
  convert -background black -extent 640x480 $fname /tmp/fixed.jpg
  mv /tmp/fixed.jpg $fname
done
echo



