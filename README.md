# 2nd Part : 3D Surface Reconstruction from Binary Medical Masks (NIfTI → PLY)

In this part we convert 3D binary segmentation masks (e.g., tumor masks) into smooth 3D triangular surface meshes using a standard medical-imaging pipeline:

---

## 1 Data Model: Voxels, Masks, and Geometry

A segmentation mask is a 3D array where each element (a voxel) indicates whether it belongs to the structure of interest.

- `mask[x, y, z] = 1` → inside the target (tumor / liver / etc.)
- `mask[x, y, z] = 0` → background

Medical volumes are sampled on a physical grid. Each voxel corresponds to a real-world size:

- `spacing = (sx, sy, sz)` in millimeters

Correct spacing matters: if you ignore it, the resulting mesh will be geometrically distorted.

A NIfTI file can also provide an affine matrix that maps voxel indices to scanner/world coordinates (mm).  
This includes:
- scaling (spacing),
- axis flips/rotations (orientation),
- translation (origin).

Many workflows extract a mesh using spacing only (mesh is in a consistent local mm-scale), while advanced workflows apply the full affine to position the mesh in scanner coordinates.

---

## 2 From Binary Volume to Surface: Marching Cubes

Marching Cubes is a classic algorithm (Lorensen & Cline, 1987) that extracts a polygonal (triangular) surface from a 3D scalar field.

For each small cube of 8 neighboring voxels, it:
1. checks which corners are inside/outside the object with respect to an isosurface threshold,
2. interpolates where the surface crosses cube edges,
3. emits one of a set of pre-defined triangle configurations.

Binary masks contain values `{0, 1}`.  
Using an iso-level of `0.5` ensures the extracted surface lies between background and foreground.

Marching Cubes returns:
- vertices (3D points in mm if spacing is used),
- faces (triplets of indices defining triangles),
- optionally normals and scalar values.

This mesh is typically dense and can contain discretization artifacts.

---

## 3 Mesh Post-Processing (Optional but Recommended)

Even after volumetric cleaning, the extracted mesh may contain:
- jagged edges ("stair-step" aliasing),
- tiny disconnected fragments,
- degenerate / duplicated triangles,
- non-manifold edges.

Typical cleanup steps include:
- removing degenerate triangles,
- removing duplicate vertices/triangles,
- removing non-manifold edges (optional; can alter topology in rare cases).

A common issue is that Laplacian smoothing tends to shrink meshes. Taubin smoothing mitigates this by alternating two smoothing steps with opposite signs (`lambda`, `mu`), producing smoother surfaces while preserving volume better.

Marching Cubes often produces many triangles. Quadric decimation reduces triangle count while preserving overall shape, improving:
- rendering performance,
- file size,
- downstream processing speed.


