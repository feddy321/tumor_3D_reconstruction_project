import open3d as o3d

mesh = o3d.io.read_triangle_mesh("meshes_out/tumor_mesh.ply")
mesh.compute_vertex_normals()

o3d.visualization.draw_geometries(
    [mesh],
    window_name="PLY Viewer",
    width=1024,
    height=768,
)