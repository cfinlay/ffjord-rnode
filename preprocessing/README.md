### CelebAHQ instructions:
Download Glow's preprocessed dataset.
```
wget https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar
tar -C data/celebahq -xvf celeb-tfr.tar
python extract_celeba_from_tfrecords
```
Move torch files to `../data/celebahq/`

### ImageNet64 instructions:
Retrieve tar files, and place in `../data/imagenet64/
```
wget http://image-net.org/small/train_64x64.tar
wget http://image-net.org/small/valid_64x64.tar
tar -xvf train_64x64.tar 
tar -xvf valid_64x64.tar
mkdir -p ../data/imagenet64/val/
mkdir -p ../data/imagenet64/train/
mv valid_64x64 ../data/imagenet64/val/0
mv train_64x64 ../data/imagenet64/train/0
```
