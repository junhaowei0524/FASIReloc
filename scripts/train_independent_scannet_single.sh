
# feature gs  --7scenes
python train_feature_gaussian.py -s datasets/7scenes/chess/ -m map_7scenes/chess --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""
python train_feature_gaussian.py -s datasets/7scenes/heads/ -m map_7scenes/heads --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""
python train_feature_gaussian.py -s datasets/7scenes/fire/ -m map_7scenes/fire --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""
python train_feature_gaussian.py -s datasets/7scenes/office/ -m map_7scenes/office --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""
python train_feature_gaussian.py -s datasets/7scenes/redkitchen/ -m map_7scenes/redkitchen --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""
python train_feature_gaussian.py -s datasets/7scenes/pumpkin/ -m map_7scenes/pumpkin --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""
python train_feature_gaussian.py -s datasets/7scenes/stairs/ -m map_7scenes/stairs --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""


# salient sampling strategy  --7scenes
python salient_sample_strategy.py -s datasets/7scenes/chess/ -m map_7scenes/chess --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""
python salient_sample_strategy.py -s datasets/7scenes/heads/ -m map_7scenes/heads --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""
python salient_sample_strategy.py -s datasets/7scenes/fire/ -m map_7scenes/fire --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""
python salient_sample_strategy.py -s datasets/7scenes/office/ -m map_7scenes/office --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""
python salient_sample_strategy.py -s datasets/7scenes/redkitchen/ -m map_7scenes/redkitchen --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""
python salient_sample_strategy.py -s datasets/7scenes/pumpkin/ -m map_7scenes/pumpkin --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""
python salient_sample_strategy.py -s datasets/7scenes/stairs/ -m map_7scenes/stairs --iterations 30000 --data_device cpu -f sp -g 3dgs --images  ""


# scene_name: only train single scene on scannet and test on 7scenes

# feature gs  --scannet  "scene scene0075_00 as an example"
python train_feature_gaussian.py -s datasets/scannet/scene0075_00 -m map_scannet/scene0075_00 --iterations 30000 --data_device cpu -f sp -g 3dgs --images "images"

# salient sampling strategy  --scannet
python salient_sample_strategy.py -s datasets/scannet/scene0075_00 -m map_scannet/scene0075_00 --iterations 30000 --data_device cpu -f sp -g 3dgs --images "images"

## scene-independent salient sampling detector  --scannet
python train_ssnet_independent.py -s datasets/scannet -m map_scannet --iterations 30000 --data_device cpu -f sp -g 3dgs --images  "images"

