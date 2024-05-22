base_dir = '/media/beegfs/home/u186/u186731/ParkingSpaceDetection'

import sys # Add AsfSearch lib to path
sys.path.insert(1, base_dir)

import geometry_utils as utils
import geoimage_dataset as geoimage

import os

dataset_dir = base_dir + '/ViennaDataset/res15cm_overlap15m'

image_name = "lb" #wmts layer name

img_download_dir = dataset_dir + "/" + image_name + "/train" + "/img"
mask_download_dir = dataset_dir + "/" + "GT" + "/train" + "/mask"

grid_all_touched = False 

img_size = (1024,1024)
img_res = (0.15,0.15)
overlap = 15

bounds = utils.build_geometry(base_dir + '/ViennaDataset/data/train/grid/Grid_v2_district2122_32tiles.shp')

img_url = "https://mapsneu.wien.gv.at/wmtsneu/1.0.0/WMTSCapabilities.xml"

img_obj = geoimage.WMTS_img(wmts=img_url,wmts_format="image/jpeg",layer=image_name)

mask_url = "https://data.wien.gv.at/daten/geo?version=1.1.0&service=WFS&request=GetCapabilities"
typename1 = 'ogdwien:FMZKVERKEHR1OGD'
mask_obj_1 = geoimage.PolygonWFSMask(mask_url,typename=typename1,class_column='F_KLASSE',
                                     class_dict={21:1,26:5,27:2,28:2,29:2,30:3,32:1,33:4},background_index=0)
typename2 = 'ogdwien:FMZKVERKEHR2OGD'
mask_obj_2 = geoimage.PolygonWFSMask(mask_url,typename=typename2,class_column='F_KLASSE',class_dict={23:6, 25:7},background_index=0)

mask_obj = geoimage.OverlayMasks([mask_obj_1,mask_obj_2],background_index=0)

ds = geoimage.geoimageDataset(area=bounds,img_size=img_size,img_res=img_res,img_obj=img_obj,mask_obj=mask_obj,all_touched=False,overlap=overlap)
ds.select_all_tiles()
ds.download(img_path = img_download_dir, mask_path = mask_download_dir, stop_on_error = False,
            overwrite=False,plot=False,ignore_only_background_tiles=False)
