#osm
import ipyleaflet
import IPython.display
import ipyvolume.pylab as p3
import pyproj
import shapely
from shapely.geometry import shape
from shapely.ops import transform
import math
import pyvista as pv
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon, Point, LineString
import os
from pyproj import Transformer
import open3d as o3d
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np
import shapely.geometry
import folium

#center_lat = 13.086083  #Setting the origin
#center_lon = 80.209621
#LOCATION_STR = "anna nagar"

# Your polygon coordinates (longitude, latitude)
#polygon_coords = [
#    (80.200628, 13.090962),
#    (80.200333, 13.082804),
#    (80.217861, 13.082192),
#    (80.218230, 13.090638)  # Closing the polygon
#]

# Set up default values for resolution
spp_default = 4096
resx_default = 1024
resy_default = 768

# Define camera settings
camera_settings = {
    "rotation": (0, 0, -90),  # Assuming Z-up orientation
    "fov": 42.854885
}

# Define material colors. This is RGB 0-1 formar https://rgbcolorpicker.com/0-1
material_colors = {
    "mat-itu_concrete": (0.539479, 0.539479, 0.539480),
    "mat-itu_marble": (0.701101, 0.644479, 0.485150),
    "mat-itu_metal": (0.219526, 0.219526, 0.254152),
    "mat-itu_wood": (0.043, 0.58, 0.184),
    "mat-itu_wet_ground": (0.91,0.569,0.055),
}

transformer = Transformer.from_crs("EPSG:4326", "EPSG:26915")
center_26915 = transformer.transform(center_lat,center_lon)
sionna_center_x = center_26915[0]
sionna_center_y = center_26915[1]
sionna_center_z = 0

scene = ET.Element("scene", version="2.1.0")
# Add defaults
ET.SubElement(scene, "default", name="spp", value=str(spp_default))
ET.SubElement(scene, "default", name="resx", value=str(resx_default))
ET.SubElement(scene, "default", name="resy", value=str(resy_default))
# Add integrator
integrator = ET.SubElement(scene, "integrator", type="path")
ET.SubElement(integrator, "integer", name="max_depth", value="12")

# Define materials
for material_id, rgb in material_colors.items():
    bsdf_twosided = ET.SubElement(scene, "bsdf", type="twosided", id=material_id)
    bsdf_diffuse = ET.SubElement(bsdf_twosided, "bsdf", type="diffuse")
    ET.SubElement(bsdf_diffuse, "rgb", value=f"{rgb[0]} {rgb[1]} {rgb[2]}", name="reflectance")

# Add emitter
emitter = ET.SubElement(scene, "emitter", type="constant", id="World")
ET.SubElement(emitter, "rgb", value="1.000000 1.000000 1.000000", name="radiance")

# Add camera (sensor)
sensor = ET.SubElement(scene, "sensor", type="perspective", id="Camera")
ET.SubElement(sensor, "string", name="fov_axis", value="x")
ET.SubElement(sensor, "float", name="fov", value=str(camera_settings["fov"]))
ET.SubElement(sensor, "float", name="principal_point_offset_x", value="0.000000")
ET.SubElement(sensor, "float", name="principal_point_offset_y", value="-0.000000")
ET.SubElement(sensor, "float", name="near_clip", value="0.100000")
ET.SubElement(sensor, "float", name="far_clip", value="10000.000000")
sionna_transform = ET.SubElement(sensor, "transform", name="to_world")
ET.SubElement(sionna_transform, "rotate", x="1", angle=str(camera_settings["rotation"][0]))
ET.SubElement(sionna_transform, "rotate", y="1", angle=str(camera_settings["rotation"][1]))
ET.SubElement(sionna_transform, "rotate", z="1", angle=str(camera_settings["rotation"][2]))
camera_position = np.array([0, 0, 100])  # Adjust camera height
ET.SubElement(sionna_transform, "translate", value=" ".join(map(str, camera_position)))
sampler = ET.SubElement(sensor, "sampler", type="independent")
ET.SubElement(sampler, "integer", name="sample_count", value="$spp")
film = ET.SubElement(sensor, "film", type="hdrfilm")
ET.SubElement(film, "integer", name="width", value="$resx")
ET.SubElement(film, "integer", name="height", value="$resy")

# Set up coordinate reference systems
wsg84 = pyproj.CRS("epsg:4326")
lambert = pyproj.CRS("epsg:26915")
transformer = pyproj.Transformer.from_crs(wsg84, lambert, always_xy=True)

# Transform coordinates from lon/lat to projected CRS
coords = [transformer.transform(lon, lat) for lon, lat in polygon_coords]

# Create the polygon
aoi_polygon = shapely.geometry.Polygon(coords)

# Get centroid for further calculations
center_x = aoi_polygon.centroid.x
center_y = aoi_polygon.centroid.y

# Convert coordinates to (lat, lon) for folium
polygon_latlon = [(lat, lon) for lon, lat in polygon_coords]

# Center map at the centroid of the polygon
center_lat = sum(lat for lat, lon in polygon_latlon) / len(polygon_latlon)
center_lon = sum(lon for lat, lon in polygon_latlon) / len(polygon_latlon)

m = folium.Map(location=[center_lat, center_lon], zoom_start=16)

# Add the polygon to the map
folium.Polygon(
    locations=polygon_latlon,
    color="blue",
    fill=True,
    fill_opacity=0.2
).add_to(m)

LOCATION_DIR = f"{LOCATION_STR}_{center_x}_{center_y}"

# Create Directories
os.makedirs(f"./simple_scene/{LOCATION_DIR}",exist_ok=True)
os.makedirs(f"./simple_scene/{LOCATION_DIR}/mesh",exist_ok=True)  

# Utility Function
def points_2d_to_poly(points, z):
    """Convert a sequence of 2d coordinates to a polydata with a polygon."""
    faces = [len(points), *range(len(points))]
    poly = pv.PolyData([p + (z,) for p in points], faces=faces)
    return poly

wsg84 = pyproj.CRS("epsg:4326")
lambert = pyproj.CRS("epsg:26915")
transformer = pyproj.Transformer.from_crs(wsg84, lambert, always_xy=True)
coords = [transformer.transform(lon, lat) for lon, lat in polygon_coords]

ground_polygon = shapely.geometry.Polygon(coords)
z_coordinates = np.full(len(ground_polygon.exterior.coords), 0)  # Assuming the initial Z coordinate is zmin
exterior_coords = ground_polygon.exterior.coords
oriented_coords = list(exterior_coords)

# Ensure counterclockwise orientation
if ground_polygon.exterior.is_ccw:
    oriented_coords.reverse()
points = [(coord[0]-center_x, coord[1]-center_y) for coord in oriented_coords]

# bounding polygon
boundary_points_polydata = points_2d_to_poly(points, z_coordinates[0])
edge_polygon = boundary_points_polydata
footprint_plane = edge_polygon.delaunay_2d()
footprint_plane.points[:] = (footprint_plane.points - footprint_plane.center)*1.5 + footprint_plane.center
pv.save_meshio(f"simple_scene/{LOCATION_DIR}/mesh/ground.ply",footprint_plane)

material_type = "mat-itu_wet_ground"
sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-ground")
ET.SubElement(sionna_shape, "string", name="filename", value=f"simple_scene/{LOCATION_DIR}/mesh/ground.ply")
bsdf_ref = ET.SubElement(sionna_shape, "ref", id=material_type, name="bsdf")
ET.SubElement(sionna_shape, "boolean", name="face_normals",value="true")

wsg84 = pyproj.CRS("epsg:4326")
lambert = pyproj.CRS("epsg:4326")
transformer = pyproj.Transformer.from_crs(wsg84, lambert, always_xy=True)
coords = [transformer.transform(lon, lat) for lon, lat in polygon_coords]

osm_polygon = shapely.geometry.Polygon(coords)
# Query the OpenStreetMap data
buildings = ox.features_from_polygon(osm_polygon, tags={'building': True})

# Filter buildings that intersect with the polygon
filtered_buildings = buildings[buildings.intersects(osm_polygon)]

buildings_list = filtered_buildings.to_dict('records')
source_crs = pyproj.CRS(filtered_buildings.crs)
target_crs = pyproj.CRS('EPSG:26915')
transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True).transform
for idx, building in enumerate(buildings_list):
    # Convert building geometry to a shapely polygon
    building_polygon = shape(building['geometry'])
    if building_polygon.geom_type != 'Polygon':
        continue
    building_polygon = transform(transformer, building_polygon)

    # Get building levels safely
    levels_value = building.get('building:levels')
    try:
        levels = float(levels_value)
        if math.isnan(levels):
            raise ValueError
        building_height = levels * 3.5
    except (TypeError, ValueError):
        building_height = 3.5  # Default

    z_coordinates = np.full(len(building_polygon.exterior.coords), 0)  # Assuming the initial Z coordinate is zmin
    exterior_coords = building_polygon.exterior.coords
    oriented_coords = list(exterior_coords)
    
    # Ensure counterclockwise orientation
    if building_polygon.exterior.is_ccw:
        oriented_coords.reverse()
    points = [(coord[0]-center_x, coord[1]-center_y) for coord in oriented_coords]

    # bounding polygon
    boundary_points_polydata = points_2d_to_poly(points, z_coordinates[0])
    edge_polygon = boundary_points_polydata
    footprint_plane = edge_polygon.delaunay_2d()
    footprint_plane = footprint_plane.triangulate()
    footprint_3D = footprint_plane.extrude((0, 0, building_height), capping=True)
    footprint_3D.save(f"simple_scene/{LOCATION_DIR}/mesh/building_{idx}.ply")
    local_mesh = o3d.io.read_triangle_mesh(f"simple_scene/{LOCATION_DIR}/mesh/building_{idx}.ply")
    o3d.io.write_triangle_mesh(f"simple_scene/{LOCATION_DIR}/mesh/building_{idx}.ply", local_mesh)
    material_type = "mat-itu_marble"
    # Add shape elements for PLY files in the folder
    sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-building_{idx}")
    ET.SubElement(sionna_shape, "string", name="filename", value=f"simple_scene/{LOCATION_DIR}/mesh/building_{idx}.ply")
    bsdf_ref = ET.SubElement(sionna_shape, "ref", id= material_type, name="bsdf")
    ET.SubElement(sionna_shape, "boolean", name="face_normals",value="true")

def convert_lane_to_numeric(lane):
    try:
        return int(lane)
    except ValueError:
        try:
            return float(lane)
        except ValueError:
            return None

        # Helper function to calculate edge geometry if missing
def calculate_edge_geometry(u, v, data):
    u_data = graph.nodes[u]
    v_data = graph.nodes[v]
    return LineString([(u_data['x'], u_data['y']), (v_data['x'], v_data['y'])])

G = ox.graph_from_polygon(polygon = osm_polygon, simplify= False, retain_all=True,truncate_by_edge=True,network_type = 'all')
graph = ox.project_graph(G, to_crs='epsg:26915')
ox.plot_graph(graph)