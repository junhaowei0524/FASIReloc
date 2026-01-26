# two stage camera relocalization pipeline  --7scenes
python fasireloc.py -m map_7scenes/fire --cfg configs/fasireloc_7scenes.yaml --path map_scannet
python fasireloc.py -m map_7scenes/heads --cfg configs/fasireloc_7scenes.yaml --path map_scannet
python fasireloc.py -m map_7scenes/office --cfg configs/fasireloc_7scenes.yaml --path map_scannet
python fasireloc.py -m map_7scenes/pumpkin --cfg configs/fasireloc_7scenes.yaml --path map_scannet
python fasireloc.py -m map_7scenes/redkitchen --cfg configs/fasireloc_7scenes.yaml --path map_scannet
python fasireloc.py -m map_7scenes/stairs --cfg configs/fasireloc_7scenes.yaml --path map_scannet