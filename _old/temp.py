import bpy


# subsurf but better if you already have this command running before i do this split
bpy.ops.object.modifier_add(type='SUBSURF')
bpy.context.object.modifiers["Subdivision"].levels = 2
bpy.ops.object.modifier_move_to_index(modifier="Subdivision", index=0)

# duplicate mesh
bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
bpy.ops.object.editmode_toggle() # enter edit mode

# extrude both eyes
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=469)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=470)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=471)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=472)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=473)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=474)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=475)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=476)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=477)
bpy.ops.mesh.extrude_region_move(MESH_OT_extrude_region={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False}, TRANSFORM_OT_translate={"value":(4.44089e-16, -4.44089e-16, -0.102655), "orient_type":'NORMAL', "orient_matrix":((0.72031, 5.23693e-08, 0.693653), (0.693653, -5.43819e-08, -0.72031), (0, 1, -7.54979e-08)), "orient_matrix_type":'NORMAL', "constraint_axis":(False, False, True), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})

# select left eye and scale
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=478)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=479)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=480)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=481)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=482)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=468)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=469)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=470)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=471)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=472)

bpy.ops.transform.resize(value=(0.7, 1, 0.7), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)

# select right eye and scale
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=473)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=474)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=475)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=476)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=477)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=483)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=484)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=485)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=486)
bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_fill=False, index=487)

bpy.ops.transform.resize(value=(0.7, 1, 0.7), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)

# command to separate eyes into new mesh here

# boolean intersect between eyes and face
bpy.ops.object.modifier_add(type='BOOLEAN')
bpy.context.object.modifiers["Boolean"].object = bpy.data.objects["anxious.003"]
bpy.context.object.modifiers["Boolean"].operation = 'INTERSECT'
bpy.context.object.modifiers["Boolean"].solver = 'FAST'

# shift the new eyes a tiny bit in front of the eyebals
bpy.ops.transform.translate(value=(-0, -0.001, -0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)
