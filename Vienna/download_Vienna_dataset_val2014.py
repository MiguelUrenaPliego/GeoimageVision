base_dir = '/media/beegfs/home/u186/u186731/ParkingSpaceDetection'

import sys # Add AsfSearch lib to path
sys.path.insert(1, base_dir)

import geometry_utils as utils
import geoimage_dataset as geoimage

import os

dataset_dir = base_dir + '/ViennaDataset/res15cm_overlap15m'

image_name = "lb2014" #wmts layer name

img_download_dir = dataset_dir + "/" + image_name + "/val" + "/img"
mask_download_dir = dataset_dir + "/" + "GT" + "/val" + "/mask"

grid_all_touched = False 

img_size = (1024,1024)
img_res = (0.15,0.15)
overlap = 15

bounds = utils.build_geometry(base_dir + '/ViennaDataset/data/test/grid/test_area_grid.shp')

img_url = "https://mapsneu.wien.gv.at/wmtsneu/1.0.0/WMTSCapabilities.xml"

img_obj = geoimage.WMTS_img(wmts=img_url,wmts_format="image/jpeg",layer=image_name)

ds = geoimage.geoimageDataset(area=bounds,img_size=img_size,img_res=img_res,img_obj=img_obj,all_touched=False,overlap=overlap)
ds.select_all_tiles()
ds.download(img_path = img_download_dir, stop_on_error = False,
            overwrite=False,plot=False)
