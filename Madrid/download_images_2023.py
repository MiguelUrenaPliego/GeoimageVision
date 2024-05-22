base_dir = '/media/beegfs/home/u186/u186731/ParkingSpaceDetection/'

import sys # Add AsfSearch lib to path
sys.path.insert(1, base_dir)

import geometry_utils as utils
import geoimage_dataset as geoimage

import os

dataset_dir = base_dir + '/MadridDataset/res10cm_overlap10m'

image_name = "ORTO_2023_10_90"

grid_all_touched = True 

img_size = (1024,1024)
img_res = (0.1,0.1)
overlap = 10

url = 'https://georaster.madrid.es/ApolloCatalogWMSpublic/service.svc/get?service=WMS&version=1.3.0&REQUEST=GetCapabilities&layers=ORTO_2023_10_90'
#url = 'https://georaster.madrid.es/ApolloCatalogWMSpublic/service.svc/get?service=WMS&version=1.3.0&REQUEST=GetCapabilities&layers=SATELITE_2022_OT_RGB'
#url = 'https://georaster.madrid.es/ApolloCatalogWMSpublic/service.svc/get?service=WMS&version=1.3.0&REQUEST=GetCapabilities&layers=ORTO_2001_E8000_10_10'


#download_list = ['BERRUGUETE','CUATRO CAMINOS','COSTILLARES', 'EL VISO','CASTELLANA','QUINTANA',
#                 'EMBAJADORES','PUERTA DEL ANGEL','LOS ROSALES','ACACIAS','GOYA',
#                 'NUMANCIA','SAN DIEGO','PALOMENAS  BAJAS','VALDERRIVAS','ARCOS',
#                 'PILAR','ORCASITAS','ALMENDRALES','UNIVERSIDAD','ALMAGRO','BELLAS VISTAS',
#                 'GAZTAMBIDE', 'HISPANOAMERICA']

barrios_Madrid = utils.read_file_to_geodataframe(base_dir + '/MadridDataset/bounds/Barrios.shp',crs=4326)
total_bounds = utils.read_file_to_geodataframe(base_dir + '/MadridDataset/bounds/Termino_municipal.shp',crs=4326)
#barrios_Madrid = barrios_Madrid[[i in download_list for i in barrios_Madrid['BARRIO_MAY']]].reset_index(drop=True)
total_selected_areas = utils.build_geometry(barrios_Madrid)

#if len(download_list) != len(barrios_Madrid):
#    raise Exception(f"Not all areas found. Len download list is {len(download_list)} len areas found {len(barrios_Madrid)}")

img_obj = geoimage.WMS_img(wms=url,wms_format="image/jpeg")

ds_all = geoimage.geoimageDataset(area=total_selected_areas,grid_bounds=total_bounds, img_size=img_size,img_res=img_res,
                                  img_obj=img_obj,all_touched=True,overlap=overlap)

names = list(barrios_Madrid['BARRIO_MAY'])
for i in range(len(names)):
    data_name = names[i].replace(" ","_")
    img_download_dir = dataset_dir + "/" + image_name + "/" + data_name + "/img"
    mask_download_dir = dataset_dir + "/" + "GT" + "/" + data_name + "/mask"
    print(f"Downloading {list(barrios_Madrid[i:i+1]['BARRIO_MAY'])[0]} to: img -> {img_download_dir} masks -> {mask_download_dir}")
    area = utils.build_geometry(barrios_Madrid[i:i+1]['geometry'],crs=4326)
    
    ds = geoimage.geoimageDataset(area=area,grid_bounds = total_bounds, img_size=img_size,img_res=img_res,img_obj=img_obj,
                                  all_touched=grid_all_touched,overlap=overlap)
    ds.select_all_tiles()
    ds.download(img_path = img_download_dir, mask_path = mask_download_dir, stop_on_error = False,
                overwrite=False,plot=False,ignore_only_background_tiles=False)
    print(f"{i+1} of {len(names)} folders done")

ds_all.grid.save_metadata(dataset_dir)