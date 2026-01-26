

# two stage camera relocalization pipeline  --cambridge
python fasireloc.py -m map_cambridge/GreatCourt --cfg configs/fasireloc_cambridge.yaml --path map_megadepth
python fasireloc.py -m map_cambridge/KingsCollege --cfg configs/fasireloc_cambridge.yaml --path map_megadepth
python fasireloc.py -m map_cambridge/OldHospital --cfg configs/fasireloc_cambridge.yaml --path map_megadepth
python fasireloc.py -m map_cambridge/ShopFacade --cfg configs/fasireloc_cambridge.yaml --path map_megadepth
python fasireloc.py -m map_cambridge/StMarysChurch --cfg configs/fasireloc_cambridge.yaml --path map_megadepth