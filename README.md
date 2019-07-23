%******************************************************************************************************************%                                  %
% If you find this work useful, please kindly cite our paper as,                                                   %
%																												   %	   
%	@article{zheng2019attributes,																				   %				
%	  title={Attributes Guided Feature Learning for Vehicle Re-identification},                                    %
%	  author={Zheng, Aihua and Lin, Xianmin and Li, Chenglong and He, Ran and Tang, Jin},                          %
%	  journal={arXiv preprint arXiv:1905.08997},                                                                   %
%	  year={2019}                                                                                                  %
%	}                                                                                                              %
%                                                                                                                  %
% This code should be used for research only. Please DO NOT distribute or use it for commercial purpose.          %
%                                      																	           %
% If you have any problem, please contact xmlin1995@gmail.com   												   %						
%******************************************************************************************************************%




### Training a Model for Vehicle Re-Identification
In order to train our models for Vehicle Re-Identification, we start of with an Imagenet pre-trained model and fine tune it on our task. These pre-trained models can be found on the [Tensorflow Models Readme Page](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).

For training a Vehicle Re-Id model, the `trainer_preid.py` script is used. To get a look at all possible arguments, have a look at the script's main method. Here, we'll only show the most important ones.


#### Training a Resnet-50 Baseline on VeRi-776
To run our training script for a Resnet-50 Baseline model with VeRi-776, execute the following while replacing all the <> tags with the corresponding values. 

`python3 trainer_preid.py --output=<output directory> --data=<dataset directory> --dataset-name=Veid_view --batch-size=16 --num-epochs=100 --network-name=resnet_v1_50 --initial-checkpoint=<path to imagenet checkpoint or another checkpoint you want to load> --checkpoint-exclude-scopes=resnet_v1_50/logits --trainable-scopes=resnet_v1_50/logits --no-evaluation`

#### Training the Views Predictor
In order to train our Views model, you need to have a dataset providing views information. In our paper, we used VeRi-776 for training the view predictor, so we label the view information by ourself which can be find on http://www.escience.cn/people/AihuaZheng/Code-Dataset.html. The `trainer_views.py` script can be called as follows:

```
python3 trainer_views.py --output=<output directory> --data=<dataset directory> --dataset-name=Veid_view --batch-size=16 --num-epochs=100 --network-name=resnet_v1_50_views --initial-checkpoint=<path to imagenet checkpoint or another checkpoint you want to load> --checkpoint-exclude-scopes=resnet_v1_50/logits --trainable-scopes=resnet_v1_50/3Views
```

#### Training the Types Predictor and Colors Predictor
The training script of Types and Colors are the same as Views Predictor.


#### Using Tensorboard to track training
During training, you can keep track of the loss and other important numbers by starting Tensorboard.

```tensorboard --logdir=<output directory of the training or a parent folder of it>```

The results can be viewed by opening a browser and go to `localhost:6006`.


### Feature Prediction
To predict the features, run the `predictor_preid.py` script:

```python3 predictor_preid.py --model-dir=<the model to be loaded> --data=<dataset directory> --dataset-name=Veid --batch-size=128 --network-name=resnet_v1_50_views```

The predicted features will be stored in a subfolder of the specified `model-dir` called `predictions`.