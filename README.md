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
