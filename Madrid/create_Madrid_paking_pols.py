base_dir = '/media/beegfs/home/u186/u186731/ParkingSpaceDetection/'

import sys # Add AsfSearch lib to path
sys.path.insert(1, base_dir)

import geometry_utils as utils
import geoimage_dataset as geoimage

import os, copy

lsmask_buffer = 30 #meters
snap_dist_close = 0.1 #meters
snap_dist_far = 2 #meters
min_target_len = 2 #meters
min_pol_area = 5 #m2
max_pol_area = 800 #m2
isolated_pols=True
prefer_small_pols = True

barrios_Madrid = utils.read_file_to_geodataframe(base_dir + '/MadridDataset/bounds/Barrios.shp',crs=4326)
names = list(barrios_Madrid['BARRIO_MAY'])

parking_spaces = [
    base_dir + "/MadridDataset/MadridVectorData/03_APARCAMIENTO_L.shp"
]
other_paths = [
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


target_geoms = utils.build_geometries(parking_spaces,crs=4326,has_z=False)
other_geoms = utils.build_geometries(other_paths,crs=4326,has_z=False)
print("Files opened")
#if type(area) != type(None):
#    target_geoms = utils.geoseries_crop(target_geoms,area,both_geoms=False)
#
#    if type(other_paths) != type(None):
#        other_geoms = utils.geoseries_crop(other_geoms, area,both_geoms=False)

#target_geoms = target_geoms.to_crs(target_geoms.estimate_utm_crs(datum_name='WGS 84'))
#target_geoms = utils.round_coordinates(target_geoms,3,has_z=False)
#self.target_geoms = self.target_geoms.simplify(self.snap_dist_close)
#target_geoms = target_geoms.to_crs(4326)
print("Target geoms loaded")
#if type(other_paths) != type(None):
        #other_geoms = other_geoms.to_crs(other_geoms.estimate_utm_crs(datum_name='WGS 84'))
        #other_geoms = utils.round_coordinates(other_geoms,3,has_z=False)
        #self.other_geoms = self.other_geoms.simplify(self.snap_dist_close)
        #other_geoms = other_geoms.to_crs(4326)

print("Data loaded. Starting polygonization.")


for i in range(len(barrios_Madrid)):
    _target_geoms = utils.geoseries_crop(copy.deepcopy(target_geoms), barrios_Madrid.geometry[i:i+1])
    _other_geoms = utils.geoseries_crop(copy.deepcopy(other_geoms), barrios_Madrid.geometry[i:i+1])
    name = base_dir + "/MadridDataset/MadridVectorData/" + names[i] + "ParkingPols.gpkg"
    if os.path.isfile(name):
        print(f"{name} already exists. Skipping polygonization")
        #parking_pols = utils.read_file_to_geoseries(name)
    else:
        print(f"Creating {name}")

        parking_pols = utils.linestrings_to_polygons(copy.deepcopy(_target_geoms),copy.deepcopy(_other_geoms),bounds = barrios_Madrid.geometry[i], snap_dist_close=snap_dist_close,snap_dist_far=snap_dist_far,
                                                    min_target_len=min_target_len, min_pol_area = min_pol_area, max_pol_area = max_pol_area, min_edges = 5, 
                                        prefer_small_pols=prefer_small_pols, isolated_pols=isolated_pols)

        parking_pols = parking_pols[parking_pols.is_valid].reset_index(drop=True)
        parking_pols.to_file(name)
    """
    utm = parking_pols.estimate_utm_crs()
    parking_pols_utm = utils.merge_geoseries(parking_pols.to_crs(utm))
    target_union = utils.merge_geoseries(_target_geoms.to_crs(utm))
    bounds = barrios_Madrid.geometry.to_crs(utm)
    target_union = target_union.intersection(bounds[i])
    inter = utils.intersect_geoseries(parking_pols_utm.buffer(snap_dist_close*2),target_union)
    if len(inter) == 0:
        print(f"{names[i]} -> Has no parking spaces")
    else:
        print(f"{names[i]} -> {inter[0].length * 100 / target_union[0].length} % of parking spaces covered by polygons")
    """
    print(f"{i+1} of {len(barrios_Madrid)} areas processed")