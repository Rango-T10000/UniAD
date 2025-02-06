#!/bin/bash

wget -c -O v1.0-trainval_meta.tgz "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz" &
wget -c -O v1.0-trainval01_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz" &
wget -c -O v1.0-trainval02_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval02_blobs.tgz" &
wget -c -O v1.0-trainval03_blobs.tgz "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_blobs.tgz" &
wget -c -O v1.0-trainval04_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval04_blobs.tgz" &
wget -c -O v1.0-trainval05_blobs.tgz "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_blobs.tgz" &
wget -c -O v1.0-trainval06_blobs.tgz "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval06_blobs.tgz" &
wget -c -O v1.0-trainval07_blobs.tgz "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_blobs.tgz" &
wget -c -O v1.0-trainval08_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval08_blobs.tgz" &
wget -c -O v1.0-trainval09_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval09_blobs.tgz" &
wget -c -O v1.0-trainval10_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval10_blobs.tgz" &
wget -c -O v1.0-test_meta.tgz "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-test_meta.tgz" &
wget -c -O v1.0-test_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-test_blobs.tgz"


tar -xzvf v1.0-trainval_meta.tgz
tar -xzvf v1.0-trainval01_blobs.tgz
tar -xzvf v1.0-trainval02_blobs.tgz
tar -xzvf v1.0-trainval03_blobs.tgz
tar -xzvf v1.0-trainval04_blobs.tgz
tar -xzvf v1.0-trainval05_blobs.tgz
tar -xzvf v1.0-trainval06_blobs.tgz
tar -xzvf v1.0-trainval07_blobs.tgz
tar -xzvf v1.0-trainval08_blobs.tgz
tar -xzvf v1.0-trainval09_blobs.tgz
tar -xzvf v1.0-trainval10_blobs.tgz
tar -xzvf v1.0-test_meta.tgz
tar -xzvf v1.0-test_blobs.tgz

