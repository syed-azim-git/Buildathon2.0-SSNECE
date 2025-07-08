# Create a list to store GeoDataFrames for each road segment
gdf_roads_list = []
# Set the fixed Z coordinate for the buffer polygons
Z0 = .25  # You can adjust this value based on the desired elevation of the roads
# Create a list to store the meshes
mesh_list = []
mesh_collection = pv.PolyData()
# Iterate over each edge in the graph
for u, v, key, data in graph.edges(keys=True, data=True):
    # Check if the edge has geometry, otherwise create geometries from the nodes
    if 'geometry' not in data:
        data['geometry'] = calculate_edge_geometry(u, v, data)

    # Get the lanes attribute for the edge
    lanes = data.get('lanes', 1)  # Default to 1 lane if lanes attribute is not available

    if not isinstance(lanes, list):
        lanes = [lanes]
        
    # Convert lane values to numeric (integers or floats) using the helper function
    num_lanes = [convert_lane_to_numeric(lane) for lane in lanes]

    # Filter out None values (representing non-numeric lanes) and calculate the road width
    num_lanes = [lane for lane in num_lanes if lane is not None]
    road_width = num_lanes[0] * 3.5
    # Buffer the LineString with the road width and add Z coordinate
    line_buffer = data['geometry'].buffer(road_width)
    # Convert the buffer polygon to a PyVista mesh
    exterior_coords = line_buffer.exterior.coords
    z_coordinates = np.full(len(line_buffer.exterior.coords), Z0)
    oriented_coords = list(exterior_coords)
    # Ensure counterclockwise orientation
    if line_buffer.exterior.is_ccw:
        oriented_coords.reverse()
    points = [(coord[0]-center_x, coord[1]-center_y) for coord in oriented_coords]
    # bounding polygon
    boundary_points_polydata = points_2d_to_poly(points, z_coordinates[0])
    mesh = boundary_points_polydata.delaunay_2d()
    # Add the mesh to the list
    mesh_collection = mesh_collection + mesh
    mesh_list.append(mesh)
output_file = f"simple_scene/{LOCATION_DIR}/mesh/road_mesh_combined.ply"
pv.save_meshio(output_file,mesh_collection)
material_type = "mat-itu_concrete"
# Add shape elements for PLY files in the folder
sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-roads_{idx}")
ET.SubElement(sionna_shape, "string", name="filename", value=f"simple_scene/{LOCATION_DIR}/mesh/road_mesh_combined.ply")
bsdf_ref = ET.SubElement(sionna_shape, "ref", id= material_type, name="bsdf")
ET.SubElement(sionna_shape, "boolean", name="face_normals",value="true")


# Create and write the XML file
tree = ET.ElementTree(scene)
xml_string = ET.tostring(scene, encoding="utf-8")
xml_pretty = minidom.parseString(xml_string).toprettyxml(indent="    ")  # Adjust the indent as needed

with open(f"simple_scene/{LOCATION_DIR}/simple_OSM_scene.xml", "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_pretty)


from sionna.rt import load_scene, Camera
scene = load_scene(r"E:\Syed Azim\Buildathon\final\simple_scene\anna nagar_1237696.5495839962_18539279.253864203\simple_OSM_scene.xml")

mi_scene = scene.mi_scene

# Get bounding box (AABB)
bbox = mi_scene.shapes()


# Initialize min and max bounds
min_corner = np.array([np.inf, np.inf, np.inf])
max_corner = np.array([-np.inf, -np.inf, -np.inf])

for shape in mi_scene.shapes():
    bbox = shape.bbox()  # Get shape bounding box
    min_corner = np.minimum(min_corner, bbox.min)
    max_corner = np.maximum(max_corner, bbox.max)

print("Scene bounding box min corner:", min_corner)
print("Scene bounding box max corner:", max_corner)

# Ground corners at z = min z
ground_corners = [
    (min_corner[0], min_corner[1], min_corner[2]),
    (min_corner[0], max_corner[1], min_corner[2]),
    (max_corner[0], max_corner[1], min_corner[2]),
    (max_corner[0], min_corner[1], min_corner[2])
]

for i, corner in enumerate(ground_corners, 1):
    print(f"Corner {i}: {corner}")


center=((min_corner[0]+max_corner[0])/2,(min_corner[1]+max_corner[1])/2,(min_corner[2]+max_corner[2])/2)
position = np.array([center[0], center[1], center[2] + 1400], dtype=np.float32)
center=np.array([center[0],center[1],center[2]])
camera = Camera(position=position)
camera.look_at(center)

fig = scene.render(camera=camera)

fig.show()