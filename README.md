This project involves using unsupervised contrastive representation learning to detect particles in CryoET data, with the goal of deriving biological information from the tomograms. As the first to attempt particle detection in the SHREC 2021 CryoET dataset, the team achieved an impressive AUCROC of 71.6% and F1 Score of 0.672.

To accomplish this, the team used 3D electron microscopy to analyze 10 tomograms, with edge detection algorithms used to identify which frames contained particles. Augmentations such as random contrast, rotation, blur, and cropping were applied before positive and negative pairs were sent for training. To solve similarity conflicts, PPG was used, and representations were learned using contrastive learning. The team used the Xnet loss on the projections and applied k-means clustering to convert the 9216-crop representation to a vector of 0s and 1s to indicate the presence of particles.

The team used various metrics such as F1 score, precision, recall, and confusion matrix to measure performance and analyze the spatial organization of deep nets. They developed a novel optimization routine using KL divergence and cross-entropy loss for spatially normal and puzzled inputs, which proved effective in improving the accuracy of object detection and segmentation tasks.

Overall, this project showcases the team's ability to leverage cutting-edge techniques in unsupervised contrastive representation learning to detect particles in CryoET data and highlights the potential of this approach to derive valuable biological information from complex imaging data.

# SpatialOrg

| exp # | training file                       | test acc | checkpoint                                                              | info                                                                                                      | 
|-------|-------------------------------------|----------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| 1     | train_classic_transformer.py        | 77.05    | scratch2/chkpt/spatial_org/train_classic_transformer/res50              | res50 as backbone and 3 layer transformer as final classifier without any augmentation or additional loss |
| 2     | train_classic_transformer_puzzle.py | 70.11    | scratch2/chkpt/spatial_org/train_classic_transformer_puzzle/res18_lr0.1 | res18 with 3x transformer enc with puzzeling idea with loss weights 0.5, 0.5, 0.05, 0.05                  | 

in both exp 1 and 2 there was an issue with the code. due to dataparallel, orig images were being mixed with the puzzles
images

new exps with dataparallel fixed

| exp # | training file                              | test acc | checkpoint                                                                                                      | info                                                                                                       | 
|-------|--------------------------------------------|----------|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| 0     | baseline pytorch result                    | 69.75    |                                                                                                                 | baseline model which the resnet backbone with a linear classifier                                          |
| 1     | train_classic_transformer.py               | 73.65    | scratch2/chkpt/spatial_org/train_classic_transformer/res18_0.1                                                  | res18 as backbone and 3x transformer enc with only classification loss                                     | 
| 2     | train_classic_transformer_puzzle_rotate.py | 73.45    | scratch2/chkpt/spatial_org/train_classic_transformer_puzzle_rotate/norotate_0.4_0.4_0.1_0.1                     | res18 as backbone and 3x transformer enc with puzzeling idea WITHOUT rotation, loss coefs=0.4,0.4, 0.1,0.1 |
| 3     | train_classic_transformer_puzzle_rotate.py | 71.13    | scratch2/chkpt/spatial_org/train_classic_transformer_puzzle_rotate/rotate_res18_0.4_0.4_0.1_0.1                 | res18 as backbone and 3x transformer enc with puzzeling idea WITH rotation, loss coefs=0.4,0.4, 0.1,0.1    | 
| 4     | train_classic_transformer_puzzle_rotate.py | 72.46    | scratch2/chkpt/spatial_org/train_classic_transformer_puzzle_rotate/norotate_res18_0.4_0.4_0.1_0.1_2xtransformer | exp #2 with 2 transformer encoders                                                                         | 
