import os
import bpy
import json
import numpy as np
import mathutils as mu

bpy.context.view_layer.update()

scene_objs = bpy.data.objects

cur = bpy.context.selected_objects[0]
name = cur.name.split(".")[0]

data = {"name":name}

bpy.ops.object.mode_set(mode='OBJECT')
selected_verts = [v for v in bpy.context.active_object.data.vertices if v.select]

verts = [] 
if "Engine" in name:
	for i in range(0,len(selected_verts),2):
		print(i)
		v1,v2 = selected_verts[i],selected_verts[i+1]
		v = (v1.co+v2.co)/2

		tup = (v[0],v[1],v[2])
		verts.append(tup)
else:
	for vert in selected_verts:
	    coord = vert.co
	    tup = (coord[0], coord[1], coord[2])
	    verts.append(tup)

data["studs"] = verts

# Parent directory of this file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
piece_data_dir = os.path.join(parent_dir, "piecedata")

# Save the file in the `piecedata` folder in this directory.
with open(os.path.join(piece_data_dir, "{}.json".format(name)), "w") as fp:
	json.dump(data, fp)

