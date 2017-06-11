mkdir input
wget -O /input http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz

EXTENDED=$1

rm input/flower_photos/*/[3-9]*

