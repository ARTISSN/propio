import bpy
import bmesh
import mathutils
import math
import json
from pathlib import Path
import numpy as np

FACE_MESH_INDICES = None

def setup_scene(ambient_color=None):
    """Clear existing scene and set up basic environment."""
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Set world background to a dim color for global illumination
    world = bpy.data.worlds["World"]
    bg = world.node_tree.nodes["Background"]
    if ambient_color is not None:
        bg.inputs[0].default_value = (*ambient_color, 1)
    else:
        bg.inputs[0].default_value = (0.15, 0.18, 0.22, 1)
    bg.inputs[1].default_value = 1.0  # Strength, increase for more ambient light

def transform_imported_mesh(obj, image_size=(512, 512)):
    """Apply necessary transformations to match mesh_utils_r.py coordinate system and center mesh."""
    global FACE_MESH_INDICES
    mesh = obj.data
    
    # Create transformation matrix
    # 1. First rotate 180° about Z (negate X and Y)
    # 2. Then rotate 180° about Y (negate X and Z)
    #rot_matrix = mathutils.Matrix.Rotation(math.radians(180), 4, 'Z')
    trans_matrix = mathutils.Matrix.Translation(mathutils.Vector((0.5, 0.5, 0)))
    # Set origin to geometry center
    #bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    # Set origin to center of mass
    #bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    
    # Move object to origin
    #obj.location = (0.0, 0.0, 0.0)
    
    # Ensure we're in OBJECT mode
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Make this object active
    bpy.context.view_layer.objects.active = obj
    
    # Create rotation matrix for Z-axis rotation
    rot_matrix = mathutils.Matrix.Rotation(math.radians(180), 4, 'Z')

    #rot_matrix = mathutils.Matrix.Rotation(math.radians(180), 4, 'Y')
    # Apply transformations to vertices
    for vertex in mesh.vertices:
        # Apply rotation
        vertex.co = rot_matrix @ vertex.co
        vertex.co = trans_matrix @ vertex.co
        vertex.co.x = -vertex.co.x

    mesh.update()
    return obj

def import_and_process_mesh(obj_path):
    """Import OBJ file and process it for iris separation."""
    # Import the OBJ file
    bpy.ops.wm.obj_import(filepath=obj_path)
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.context.object.modifiers["Subdivision"].levels = 2
    
    # Save current selected object as "face"
    face = bpy.context.object
    
    # Apply transformations to match mesh_utils_r coordinate system
    face = transform_imported_mesh(face)
    
    # Duplicate the face object
    face_dup = face.copy()
    face_dup.data = face.data.copy()
    face_dup.name = "face_dup"
    bpy.context.collection.objects.link(face_dup)
    
    return face, face_dup

def process_irises(face_dup):
    """Process the iris geometry."""
    bpy.context.view_layer.objects.active = face_dup
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    
    bm = bmesh.from_edit_mesh(face_dup.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    
    # First deselect everything
    for v in bm.verts:
        v.select = False
    for f in bm.faces:
        f.select = False
    
    # Find and select iris faces
    iris_faces = [f for f in bm.faces if all(v.index in range(468, 478) for v in f.verts)]
    for f in iris_faces:
        f.select = True
        # Also select the face's vertices and edges for proper extrusion
        for v in f.verts:
            v.select = True
        for e in f.edges:
            e.select = True
    
    # Extrude the faces
    ret = bmesh.ops.extrude_face_region(bm, geom=iris_faces)
    
    # Get the new vertices
    extruded_verts = [e for e in ret['geom'] if isinstance(e, bmesh.types.BMVert)]
    
    # Compute average normal for iris faces (or use a fixed direction)
    avg_normal = mathutils.Vector((0, 0, 0))
    for f in iris_faces:
        print(f.normal)
        avg_normal += f.normal
    avg_normal.normalize()
    print(avg_normal)
    
    # Move the new vertices along the normal (e.g., 2mm out)
    prism_depth = -0.02  # Adjust as needed
    bmesh.ops.translate(bm, verts=extruded_verts, vec=avg_normal * prism_depth)

    # Select all iris faces and new geometry
    for f in iris_faces:
        f.select = True
        for v in f.verts:
            v.select = True
        for e in f.edges:
            e.select = True
    
    # Also select all extruded vertices and their connected faces
    for v in extruded_verts:
        v.select = True
        for f in v.link_faces:
            f.select = True
            for e in f.edges:
                e.select = True
    
    # Update the mesh
    bmesh.update_edit_mesh(face_dup.data)
    
    # Separate irises
    before_objects = set(bpy.data.objects)
    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode='OBJECT')
    after_objects = set(bpy.data.objects)
    
    irises = list(after_objects - before_objects)[0]
    return irises

def create_material(name, properties):
    """Create a material with specified properties."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    # Set material properties
    if "Kd" in properties:  # Diffuse color
        bsdf.inputs["Base Color"].default_value = (*properties["Kd"], 1.0)
    if "Ks" in properties:  # Specular color
        # In newer Blender versions, "Specular" is now "Specular IOR Level"
        bsdf.inputs["Specular IOR Level"].default_value = properties["Ks"][0]  # Use first component
    if "Ns" in properties:  # Specular exponent (roughness)
        # Convert Ns (0-1000) to roughness (0-1), inverted
        roughness = 1.0 - min(properties["Ns"] / 1000.0, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
    if "d" in properties:  # Transparency
        bsdf.inputs["Alpha"].default_value = properties["d"]
        if properties["d"] < 1.0:
            mat.blend_method = 'BLEND'
    
    # Set some default values for better material appearance
    bsdf.inputs["Metallic"].default_value = 0.0  # Non-metallic
    bsdf.inputs["IOR"].default_value = 1.45  # Common value for skin/organic materials
    #bsdf.inputs["Clearcoat"].default_value = 0.1  # Slight clearcoat for skin sheen
    
    return mat

def load_material_properties():
    """Load material properties from the MTL file."""
    mtl_path = Path(__file__).parent / "material.mtl"
    print(f"Loading materials from {mtl_path}")
    
    materials = {}
    current_material = None
    
    with open(mtl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'newmtl':
                    current_material = parts[1]
                    print(f"Found material: {current_material}")
                    materials[current_material] = {}
                elif current_material is not None:
                    if parts[0] in ['Kd', 'Ks', 'Ka']:
                        # Color properties
                        materials[current_material][parts[0]] = [float(x) for x in parts[1:4]]
                    elif parts[0] in ['Ns', 'd', 'Ni']:
                        # Scalar properties
                        materials[current_material][parts[0]] = float(parts[1])
            except Exception as e:
                print(f"Error parsing line {line_num}: {line}")
                print(f"Error: {str(e)}")
    
    print(f"Loaded {len(materials)} materials: {', '.join(materials.keys())}")
    return materials

def import_reference_mesh():
    """Import the reference mesh to get material assignments."""
    ref_obj_path = str(Path(__file__).parent / "material.obj")
    ref_mtl_path = str(Path(__file__).parent / "material.mtl")
    
    print(f"Importing reference mesh from {ref_obj_path}")
    print(f"Using material file {ref_mtl_path}")
    
    # Import reference mesh with materials
    bpy.ops.wm.obj_import(
        filepath=ref_obj_path,
        directory=str(Path(ref_obj_path).parent),
        files=[{"name": Path(ref_obj_path).name}],
        forward_axis='Y',
        up_axis='Z',
        use_split_objects=False,
        use_split_groups=False,
        import_vertex_groups=True
    )
    
    ref_obj = bpy.context.selected_objects[0]
    
    # If no materials were imported, try to load them manually
    if len(ref_obj.data.materials) == 0:
        print("No materials imported automatically, loading from MTL...")
        material_properties = load_material_properties()
        
        # Create and assign materials
        for mat_name, properties in material_properties.items():
            mat = create_material(mat_name, properties)
            ref_obj.data.materials.append(mat)
    
    print(f"Reference object has {len(ref_obj.data.materials)} materials:")
    for mat in ref_obj.data.materials:
        print(f"- {mat.name}")
    
    # Create mapping of face indices to material names
    material_map = {}
    default_material = ref_obj.data.materials[0] if ref_obj.data.materials else None
    
    for poly in ref_obj.data.polygons:
        verts = tuple(sorted([v for v in poly.vertices]))
        if poly.material_index < len(ref_obj.data.materials):
            #print(f"Polygon {poly.index} has material index {poly.material_index}")
            mat_name = ref_obj.data.materials[poly.material_index].name
        else:
            mat_name = default_material.name if default_material else 'face'
        material_map[verts] = mat_name
        
    print(f"Created material map with {len(material_map)} face assignments")
    
    # Delete reference mesh as we only need the mapping
    bpy.ops.object.delete()
    
    return material_map

def get_base_material_name(name):
    """Extract base material name by removing numeric suffixes."""
    # Split by '.' and take the first part to remove numeric suffixes
    return name.split('.')[0]

def calculate_region_color(image_path, mask_indices):
    """Calculate average color of a region in an image using Blender's built-in image handling."""
    try:
        # Load image using Blender's image handling
        img = bpy.data.images.load(str(image_path))
        
        # Get image dimensions
        width = img.size[0]
        height = img.size[1]
        
        # Get pixel data (returns RGBA values from 0-1)
        pixels = list(img.pixels[:])
        
        # Create mask for the region
        total_r = total_g = total_b = 0
        pixel_count = 0
        
        # Sample pixels at mask indices
        for idx in mask_indices:
            if 0 <= idx < len(FACE_MESH_INDICES):
                x, y = FACE_MESH_INDICES[idx]
                if 0 <= x < width and 0 <= y < height:
                    # Calculate pixel index (4 channels: RGBA)
                    pixel_idx = int((y * width + x) * 4)
                    if pixel_idx + 2 < len(pixels):
                        total_r += pixels[pixel_idx]
                        total_g += pixels[pixel_idx + 1]
                        total_b += pixels[pixel_idx + 2]
                        pixel_count += 1
        
        # Calculate average color
        if pixel_count > 0:
            avg_color = [total_r/pixel_count, total_g/pixel_count, total_b/pixel_count]
            # Colors are already in 0-1 range in Blender
            return avg_color
            
        # Remove the loaded image from memory
        bpy.data.images.remove(img)
            
    except Exception as e:
        print(f"Warning: Error calculating region color: {str(e)}")
    
    return [0.8, 0.8, 0.8]  # Default color if anything fails

def get_region_indices():
    """Get indices for different facial regions."""
    # These are example indices - you'll need to adjust these based on your face mesh
    return {
        'face': list(range(0, 468)),  # All face indices except specific features
        'lips': list(range(0, 16)) + list(range(76, 96)),  # Lip indices
        'eyebrows': list(range(282, 296)) + list(range(156, 173)),  # Eyebrow indices
    }

def setup_materials(obj, irises, character_dir, frame_name):
    """Set up materials using the reference mesh for assignments and image colors."""
    material_properties = load_material_properties()
    
    # Clear existing materials
    obj.data.materials.clear()
    irises.data.materials.clear()
    
    # Get reference face image path for the specific frame
    face_image_path = character_dir / "processed" / "maps" / "faces" / f"{frame_name}.png"
    if not face_image_path.exists():
        print(f"Warning: Reference face image not found at {face_image_path}")
    
    # Get region indices
    region_indices = get_region_indices()
    
    # Create materials with colors from reference image
    materials = {}
    for mat_name, properties in material_properties.items():
        # Create base material
        materials[mat_name] = create_material(mat_name, properties)
        
        # Update color based on region if image exists
        if face_image_path.exists():
            base_name = get_base_material_name(mat_name)
            if base_name in region_indices:
                avg_color = calculate_region_color(face_image_path, region_indices[base_name])
                # Update material color
                bsdf = materials[mat_name].node_tree.nodes["Principled BSDF"]
                bsdf.inputs["Base Color"].default_value = (*avg_color, 1.0)
                print(f"Set {mat_name} color to {avg_color}")
        
        # Add material to appropriate object
        if 'iris' in get_base_material_name(mat_name):
            print(f"Adding iris material {mat_name}")
            irises.data.materials.append(materials[mat_name])
            irises.active_material_index = len(irises.data.materials) - 1
        obj.data.materials.append(materials[mat_name])
    
    # Continue with material assignment as before...
    try:
        material_map = import_reference_mesh()
        for poly in obj.data.polygons:
            verts = tuple(sorted([v for v in poly.vertices]))
            if verts in material_map:
                mat_name = material_map[verts]
                base_mat_name = get_base_material_name(mat_name)
                mat_idx = -1
                for idx, material in enumerate(obj.data.materials):
                    if get_base_material_name(material.name) == base_mat_name:
                        mat_idx = idx
                        break
                if mat_idx >= 0:
                    poly.material_index = mat_idx
            else:
                poly.material_index = obj.data.materials.find('face')
    except Exception as e:
        print(f"Error during material assignment: {str(e)}")
        default_mat_idx = obj.data.materials.find('face')
        for poly in obj.data.polygons:
            poly.material_index = default_mat_idx
    
    # Print material assignment statistics
    #print("\nMaterial assignment statistics:")
    mat_counts = {mat.name: 0 for mat in obj.data.materials}
    for poly in obj.data.polygons:
        mat_name = obj.data.materials[poly.material_index].name
        mat_counts[mat_name] += 1
    #for mat_name, count in mat_counts.items():
    #    print(f"{mat_name}: {count} polygons")

    # Find face material color
    face_mat = None
    for mat_name, mat in materials.items():
        if get_base_material_name(mat_name) == 'face':
            face_mat = mat
            break
    if face_mat is not None:
        bsdf = face_mat.node_tree.nodes["Principled BSDF"]
        face_color = bsdf.inputs["Base Color"].default_value[:3]
    else:
        face_color = (0.8, 0.8, 0.8)
    return face_color

def setup_camera_and_lighting(cam_pos, cam_dir, suns=None, ambient_color=None):
    """Set up camera and lighting in the scene, including global illumination."""
    # Camera setup
    bpy.ops.object.camera_add(location=cam_pos)
    camera = bpy.context.active_object

    cam_dir = mathutils.Vector(cam_dir).normalized()
    forward = mathutils.Vector((0.0, 0.0, -1.0))
    rotation = forward.rotation_difference(cam_dir)

    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = rotation
    bpy.context.scene.camera = camera

    # Make camera orthographic
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 1.0

    # Remove any existing sun lamps
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Add a sun lamp for each sun direction
    if suns is not None:
        for i, sun in enumerate(suns):
            # Apply OBJ import axis swap and mesh Y flip
            direction = sun["direction"]
            intensity = 1#sun["intensity"]
            sun_vec = mathutils.Vector((
                -direction[0],    # x
                -direction[2],    # y
                -direction[1]     # z
            )).normalized()
            bpy.ops.object.light_add(type='SUN', location=(0, 0, 0))
            light = bpy.context.active_object
            light.data.energy = intensity  # You may want to scale/normalize this
            rot_quat = mathutils.Vector((0, 0, -1)).rotation_difference(sun_vec)
            light.rotation_mode = 'QUATERNION'
            light.rotation_quaternion = rot_quat
            bpy.ops.object.light_add(type='SUN', location=(0, 0, 0))
            light = bpy.context.active_object
            light.data.energy = 0.3*intensity  # You may want to scale/normalize this
            sun_vec = mathutils.Vector((
                direction[0],    # x
                direction[2],    # y
                -direction[1]    # z
            )).normalized()
            rot_quat = mathutils.Vector((0, 0, -1)).rotation_difference(sun_vec)
            light.rotation_mode = 'QUATERNION'
            light.rotation_quaternion = rot_quat
    else:
        # Default sun direction
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 0))
        light = bpy.context.active_object
        light.data.energy = 1
        light.rotation_euler = (math.radians(-90), 0, math.radians(180))

    # Set world background for global illumination
    world = bpy.data.worlds["World"]
    bg = world.node_tree.nodes["Background"]
    if ambient_color is not None:
        bg.inputs[0].default_value = (*ambient_color, 1)
    else:
        bg.inputs[0].default_value = (0.15, 0.18, 0.22, 1)
    bg.inputs[1].default_value = 0.8

def render_scene(output_path, resolution=512):
    """Configure render settings and render the scene."""
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_path
    
    # Enable material preview quality
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Or 'EEVEE' for faster renders
    bpy.context.scene.render.use_high_quality_normals = True
    
    # If using Cycles
    bpy.context.scene.cycles.samples = 128  # Adjust based on quality needs
    bpy.context.scene.cycles.use_denoising = True
    
    bpy.ops.render.render(write_still=True)

def process_character(character_dir, frame_name=None):
    """Process a character's data and render the result."""
    global FACE_MESH_INDICES
    character_dir = Path(character_dir)
    metadata_path = character_dir / "metadata.json"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # If no specific frame is provided, process all frames
    frames_to_process = [frame_name] if frame_name else metadata["frames"].keys()
    
    for frame in frames_to_process:
        if frame not in metadata["frames"]:
            print(f"Frame {frame} not found in metadata")
            continue
            
        frame_data = metadata["frames"][frame]
        processed_dir = character_dir / "processed"
        obj_path = processed_dir / "meshes" / f"{frame}.obj"
        output_path = processed_dir / "renders" / f"{frame}.png"
        output_path.parent.mkdir(exist_ok=True)
        
        if not obj_path.exists():
            print(f"OBJ file not found: {obj_path}")
            continue

        # Read face mesh indices for this frame
        face_mesh_indices = frame_data.get("face_mesh_indices", None)
        if face_mesh_indices is None:
            print(f"Warning: No face_mesh_indices found for frame {frame}")
            continue
        FACE_MESH_INDICES = face_mesh_indices

        # Default camera position and direction if not specified in metadata
        cam_pos = (0, -1, 0)
        cam_dir = (0, 1, 0)
        
        suns = None
        lighting = frame_data.get("lighting", {})
        if "suns" in lighting:
            suns = lighting["suns"]
        
        # Setup and render
        setup_scene()
        face, face_dup = import_and_process_mesh(str(obj_path))
        irises = process_irises(face_dup)
        face_color = setup_materials(face_dup, irises, character_dir, frame)
        setup_camera_and_lighting(cam_pos, cam_dir, suns=suns, ambient_color=face_color)
        render_scene(str(output_path))
        
        print(f"Rendered {frame} to {output_path}")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: blender --background --python blender_render.py -- <character_dir> [frame_name]")
        sys.exit(1)
    
    # The last argument after -- will be either the character_dir or frame_name
    last_arg = sys.argv[-1]
    second_last_arg = sys.argv[-2] if len(sys.argv) > 2 else None
    
    # Check if the last argument looks like a path
    if Path(last_arg).suffix == '' and not Path(last_arg).is_dir():
        # Last arg is not a path, so it must be a frame name
        character_dir = second_last_arg
        frame_name = last_arg
    else:
        # Last arg is a path, so it's the character directory
        character_dir = last_arg
        frame_name = None
    
    print(f"Character directory: {character_dir}")
    print(f"Frame name: {frame_name}")
    
    process_character(character_dir, frame_name)