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


lsmask_buffer = 30 #meters
snap_dist_close = img_res[0] #meters
snap_dist_far = 2 #meters
min_target_len = 2 #meters
min_pol_area = 5 #m2
max_pol_area = 800 #m2
isolated_pols=True
prefer_small_pols = True

download_list = ['BERRUGUETE','CUATRO CAMINOS','COSTILLARES', 'EL VISO','CASTELLANA','QUINTANA',
                 'EMBAJADORES','PUERTA DEL ANGEL','LOS ROSALES','ACACIAS','GOYA',
                 'NUMANCIA','SAN DIEGO','PALOMENAS  BAJAS','VALDERRIVAS','ARCOS',
                 'PILAR','ORCASITAS','ALMENDRALES','UNIVERSIDAD','ALMAGRO','BELLAS VISTAS',
                 'GAZTAMBIDE', 'HISPANOAMERICA']

barrios_Madrid = utils.read_file_to_geodataframe(base_dir + '/MadridDataset/bounds/Barrios.shp',crs=4326)
total_bounds = utils.read_file_to_geodataframe(base_dir + '/MadridDataset/bounds/Termino_municipal.shp',crs=4326)
barrios_Madrid = barrios_Madrid[[i in download_list for i in barrios_Madrid['BARRIO_MAY']]].reset_index(drop=True)
total_selected_areas = utils.build_geometry(barrios_Madrid)

if len(download_list) != len(barrios_Madrid):
    raise Exception(f"Not all areas found. Len download list is {len(download_list)} len areas found {len(barrios_Madrid)}")

img_obj = geoimage.WMS_img(wms=url,wms_format="image/jpeg")

parking_ls = [
    base_dir + "/MadridDataset/MadridVectorData/03_APARCAMIENTO_L.shp"
]
other_ls = [
    base_dir + "/MadridDataset/MadridVectorData/03_ACERA_ISLETA_L.shp",
    base_dir + "/MadridDataset/MadridVectorData/03_ACERA_NIVEL_L.shp",
    base_dir + "/MadridDataset/MadridVectorData/03_BORDILLO_L.shp",
    base_dir + "/MadridDataset/MadridVectorData/01_EDIFICIO_FACHADA_L.shp",
    base_dir + "/MadridDataset/MadridVectorData/01_MURO_CONTENCION_L.shp",
    base_dir + "/MadridDataset/MadridVectorData/01_MURO_PARED_TAPIA_L.shp",
    base_dir + "/MadridDataset/MadridVectorData/07_SETO_L.shp",
    base_dir + "/MadridDataset/MadridVectorData/01_ALAMBRADA_L.shp",
    base_dir + "/MadridDataset/MadridVectorData/03_VADO_L.shp",
    base_dir + "/MadridDataset/MadridVectorData/03_RAMPA_L.shp",
    base_dir + "/MadridDataset/MadridVectorData/03_PASO_PEATONAL_ELEVADO_L.shp",
    base_dir + "/MadridDataset/MadridVectorData/03_PINTURA_BORDE_L.shp",
    base_dir + "/MadridDataset/MadridVectorData/06_CARRIL_BICI_L.shp",
    #base_dir + "/VectorData/01_PATIO_L.shp"
]

building_polygons = [
    base_dir + "/MadridDataset/MadridVectorData/01_EDIFICIO_P.shp",
]
sidewalk_polygons = [
    base_dir + "/MadridDataset/MadridVectorData/03_ACERA_P.shp",
    base_dir + "/MadridDataset/MadridVectorData/03_ACERA_NIVEL_P.shp",
]
pool_polygons = [
    base_dir + "/MadridDataset/MadridVectorData/05_PISCINA_P.shp",
]
street_polygons = [
    base_dir + "/MadridDataset/MadridVectorData/VIALES.shp",
]
bike_lane_polygons = [
    base_dir + "/MadridDataset/MadridVectorData/06_CARRIL_BICI_P.shp",
]

total_mask_area = utils.buffer_in_m(total_selected_areas,max(img_res)*max(img_size)*2)
total_mask_obj_1 = geoimage.PolygonMask(building_polygons,values=1,background_index=0,area=total_mask_area)
total_mask_obj_2 = geoimage.PolygonMask(street_polygons,values=2,background_index=0,area=total_mask_area)
total_mask_obj_3 = geoimage.PolygonMask(sidewalk_polygons,values=3,background_index=0,area=total_mask_area)
total_mask_obj_4 = geoimage.PolygonMask(pool_polygons,values=4,background_index=0,area=total_mask_area)
total_mask_obj_5 = geoimage.PolygonMask(bike_lane_polygons,values=5,background_index=0,area=total_mask_area)
total_mask_obj_6 = geoimage.LineStringMask(parking_ls,other_ls,values = 6, background_index = 0, area=total_mask_area,
                                buffer=lsmask_buffer, snap_dist_far=snap_dist_far, snap_dist_close=snap_dist_close,
                                min_target_len=min_target_len, min_pol_area = min_pol_area, max_pol_area = max_pol_area, min_edges = 5, 
                                prefer_small_pols=prefer_small_pols, isolated_pols=isolated_pols)

total_mask_obj = geoimage.OverlayMasks([total_mask_obj_1,total_mask_obj_2,total_mask_obj_3,total_mask_obj_4,total_mask_obj_5,total_mask_obj_6],background_index=0)

ds_all = geoimage.geoimageDataset(area=total_selected_areas,grid_bounds=total_bounds, img_size=img_size,img_res=img_res,
                                  img_obj=img_obj,mask_obj=total_mask_obj,all_touched=True,overlap=overlap)

names = list(barrios_Madrid['BARRIO_MAY'])
for i in range(len(names)):
    data_name = names[i].replace(" ","_")
    img_download_dir = dataset_dir + "/" + image_name + "/" + data_name + "/img"
    mask_download_dir = dataset_dir + "/" + "GT" + "/" + data_name + "/mask"
    print(f"Downloading {list(barrios_Madrid[i:i+1]['BARRIO_MAY'])[0]} to: img -> {img_download_dir} masks -> {mask_download_dir}")
    area = utils.build_geometry(barrios_Madrid[i:i+1]['geometry'],crs=4326)
    mask_area = utils.buffer_in_m(area,max(img_res)*max(img_size)*2)
    mask_obj_1 = geoimage.PolygonMask(total_mask_obj_1.target_geoms,values=1,background_index=0,area=mask_area)
    mask_obj_2 = geoimage.PolygonMask(total_mask_obj_2.target_geoms,values=2,background_index=0,area=mask_area)
    mask_obj_3 = geoimage.PolygonMask(total_mask_obj_3.target_geoms,values=3,background_index=0,area=mask_area)
    mask_obj_4 = geoimage.PolygonMask(total_mask_obj_4.target_geoms,values=4,background_index=0,area=mask_area)
    mask_obj_5 = geoimage.PolygonMask(total_mask_obj_5.target_geoms,values=5,background_index=0,area=mask_area)
    mask_obj_6 = geoimage.LineStringMask(total_mask_obj_6.target_geoms,total_mask_obj_6.other_geoms,values = 6, background_index = 0, area=mask_area,
                                    buffer=lsmask_buffer, snap_dist_far=snap_dist_far, snap_dist_close=snap_dist_close,
                                    min_target_len=min_target_len, min_pol_area = min_pol_area, max_pol_area = max_pol_area, min_edges = 5, 
                                    prefer_small_pols=prefer_small_pols, isolated_pols=isolated_pols)

    mask_obj = geoimage.OverlayMasks([mask_obj_1,mask_obj_2,mask_obj_3,mask_obj_4,mask_obj_5,mask_obj_6],background_index=0)

    
    ds = geoimage.geoimageDataset(area=area,grid_bounds = total_bounds, img_size=img_size,img_res=img_res,img_obj=img_obj,mask_obj=mask_obj,
                                  all_touched=grid_all_touched,overlap=overlap)
    ds.select_all_tiles()
    ds.download(img_path = img_download_dir, mask_path = mask_download_dir, stop_on_error = False,
                overwrite=False,plot=False,ignore_only_background_tiles=False)
    print(f"{i+1} of {len(names)} folders done")

ds_all.grid.save_metadata(dataset_dir)