Environment
1. jdk11 is necessary
sudo apt update
https://www.jianshu.com/p/5a25b9535016
apt-get install -y openjdk-17-jdk

this is the model_dir
0. contain customer-defined handler.py, checkpoint path
1. torch-model-archiver --model-name resnet-18 --version 1.0 --model-file ./examples/image_classifier/resnet_18/model.py \
    --serialized-file resnet18-f37072fd.pth --handler image_classifier --extra-files ./examples/image_classifier/index_to_name.json
2. After obtaining mar file, put it to model_store
