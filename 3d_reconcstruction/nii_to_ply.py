import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from skimage.measure import marching_cubes
import open3d as o3d
from dataclasses import dataclass
from typing import Optional

def load_nii(file_path):
    """
    Charge un fichier NIfTI et retourne l'image NIfTI.
    
    Args:
        file_path (str): Le chemin vers le fichier NIfTI.
        
    Returns:
        nibabel.Nifti1Image: L'image NIfTI chargée.
    """
    nii = nib.load(file_path)
    return nii

def nii_to_numpy(nii):
    """
    Convertit une image NIfTI en un tableau numpy 3D.
    
    Args:
        nii (nibabel.Nifti1Image): L'image NIfTI à convertir.
        
    Returns:
        numpy.ndarray: Le volume 3D sous forme de tableau numpy.
    """
    volume = nii.get_fdata() #coords: x y z labels: 0=none, 1=liver, 2=tumor
    return volume

def visualize_slice(volume, slice_index):
    """
    Visualise une coupe 2D d'un volume 3D.
    
    Args:
        volume (numpy.ndarray): Le volume 3D à visualiser.
        slice_index (int): L'index de la coupe à afficher.
    """
    plt.imshow(volume[:, :, slice_index], cmap='gray')
    plt.title(f'Slice {slice_index}')
    plt.axis('off')
    plt.show()




@dataclass
class MeshPostProcessConfig:
    taubin_smooth_iters: int = 30   # moderate smoothing
    taubin_lambda: float = 0.5
    taubin_mu: float = -0.53
    decimate_target_triangles: Optional[int] = 150_000  # set None to disable
    remove_degenerate: bool = True
    remove_duplicated: bool = True
    compute_normals: bool = True


def postprocess_open3d(mesh, cfg: MeshPostProcessConfig):

    if cfg.remove_degenerate:
        mesh.remove_degenerate_triangles()
    if cfg.remove_duplicated:
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Taubin smoothing keeps volume better than basic Laplacian
    if cfg.taubin_smooth_iters and cfg.taubin_smooth_iters > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=int(cfg.taubin_smooth_iters),
            lambda_filter=float(cfg.taubin_lambda),
            mu=float(cfg.taubin_mu),
        )

    # Optional decimation (keeps shape but reduces triangles)
    if cfg.decimate_target_triangles is not None:
        target = int(cfg.decimate_target_triangles)
        if len(mesh.triangles) > target:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)

    if cfg.compute_normals:
        mesh.compute_vertex_normals()

    return mesh

def binary_to_mesh_ply(
    binary_xyz: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    out_ply_path: str,
    iso_level: float = 0.5,
) -> None:
    """
    Convert a binary volume (0/1) in (x,y,z) order into a triangular mesh and save to .ply.
    """
    if binary_xyz.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={binary_xyz.shape}")

    # Ensure binary float volume for marching cubes
    vol = (binary_xyz > 0).astype(np.float32)

    if not np.any(vol):
        raise ValueError("Volume is empty (no 1s). Nothing to mesh.")

    # Marching Cubes:
    # skimage.measure.marching_cubes expects spacing per axis of the array as given.
    verts, faces, normals, values = marching_cubes(
        vol,
        level=iso_level,
        spacing=spacing_xyz,   
        allow_degenerate=False
    )

    # Create Open3D mesh and export to PLY
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

    cfg = MeshPostProcessConfig()
    mesh = postprocess_open3d(mesh, cfg)

    ok = o3d.io.write_triangle_mesh(out_ply_path, mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to write PLY: {out_ply_path}")


# ----------------------------
# Main execution


nii = load_nii("data/segmentation-9.nii")
print(nibabel.aff2axcodes(nii.affine)) #('Left/Right', 'Anterior/Posterior', 'Superior/Inferior')
print(nii.header.get_zooms())  # voxel spacing (mm) : x, y, z

volume = nii_to_numpy(nii)
print(type(volume))  # Devrait afficher: <class 'numpy.ndarray'>
print(volume.shape)  # x y z

valeurs, comptes = np.unique(volume, return_counts=True)

print(valeurs)
print(comptes)


binary_to_mesh_ply(
        binary_xyz=volume,
        spacing_xyz=nii.header.get_zooms()[:3],   # (x,y,z)
        out_ply_path="meshes_out/tumor_mesh.ply"
    )