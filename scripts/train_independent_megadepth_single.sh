
# feature gs  --cambridge
python train_feature_gaussian.py  -s datasets/cambridge/GreatCourt -m map_cambridge/GreatCourt -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu  --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001
python train_feature_gaussian.py  -s datasets/cambridge/KingsCollege -m map_cambridge/KingsCollege -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu  --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001
python train_feature_gaussian.py  -s datasets/cambridge/OldHospital -m map_cambridge/OldHospital -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu  --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001
python train_feature_gaussian.py  -s datasets/cambridge/ShopFacade -m map_cambridge/ShopFacade -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu  --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001
python train_feature_gaussian.py  -s datasets/cambridge/StMarysChurch -m map_cambridge/StMarysChurch -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu  --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001

# salient sampling strategy  --cambridge
python salient_sample_strategy.py  -s datasets/cambridge/GreatCourt -m map_cambridge/GreatCourt -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu  --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001
python salient_sample_strategy.py  -s datasets/cambridge/KingsCollege -m map_cambridge/KingsCollege -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu  --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001
python salient_sample_strategy.py  -s datasets/cambridge/OldHospital -m map_cambridge/OldHospital -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu  --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001
python salient_sample_strategy.py  -s datasets/cambridge/ShopFacade -m map_cambridge/ShopFacade -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu  --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001
python salient_sample_strategy.py  -s datasets/cambridge/StMarysChurch -m map_cambridge/StMarysChurch -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu  --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001


# scene_name: only train single scene on scannet and test on cambridge

# feature gs  --megadepth "scene 0016 as an example"
python train_feature_gaussian.py -s datasets/megadepth/0016 -m map_megadepth/0016 -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu --densify_grad_threshold 0.0004 --images "images" --position_lr_init 0.000016 --scaling_lr 0.001

# salient sampling strategy  --megadepth
python salient_sample_strategy.py -s datasets/megadepth/0016 -m map_megadepth/0016 -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu --densify_grad_threshold 0.0004 --images "images" --position_lr_init 0.000016 --scaling_lr 0.001

## scene-independent salient sampling detector  --megadepth
python train_ssnet_independent.py -s datasets/megadepth -m map_megadepth --iterations 30000 --data_device cpu -f sp -g 3dgs --images  "images"

