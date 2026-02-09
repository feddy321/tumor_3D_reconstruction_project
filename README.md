# Medical Image Segmentation & 3D Reconstruction

This project implements an end-to-end pipeline for medical image analysis, specifically focusing on **Liver Tumor Segmentation** using CT scans (LiTS dataset). The workflow consists of two main stages:

1. **Deep Learning Segmentation:** Using a U-Net architecture optimized with Optuna to generate binary masks.
2. **3D Surface Reconstruction:** Converting binary NIfTI masks into 3D meshes (PLY) using Marching Cubes.


---

## Project Pipeline

### Part 1: Deep Learning Segmentation (U-Net)

We utilize a U-Net architecture implemented in PyTorch to perform semantic segmentation on CT slices.

* **Model:** Standard U-Net with bilinear upsampling.
* **Optimization:** Hyperparameters (Learning rate, weight decay, batch size) were tuned using **Optuna** with a TPE Sampler and Median Pruner.
* **Tracking:** Experiment tracking and visualization were handled via **WandB**.
* **Inference:** The best model is selected based on the Dice Coefficient. We identify the "Best", "Second Best", and "Mean-close" cases from the test set for analysis.

### Part 2: 3D Surface Reconstruction (NIfTI → PLY)

Once the segmentation masks are generated, we convert them into 3D geometric surfaces.

**1. Data Model:**

* Input: 3D Binary Segmentation Masks (NIfTI).
* Voxel values: `1` (Tumor/Organ), `0` (Background).
* Physicality: We respect voxel spacing `(sx, sy, sz)` to ensure the mesh has correct real-world dimensions (mm).

**2. Marching Cubes Algorithm:**
We use the **Marching Cubes** algorithm (Lorensen & Cline, 1987) to extract a triangular surface from the voxel grid.

* It iterates through the 3D scalar field.
* It determines triangle configurations based on an isosurface threshold (typically 0.5 for binary masks).
* **Output:** Vertices and Faces.

**3. Mesh Post-Processing:**
Raw meshes from Marching Cubes often contain artifacts. We apply:

* **Cleaning:** Removing degenerate triangles and duplicate vertices.
* **Smoothing:** Taubin smoothing is preferred over Laplacian smoothing to preserve volume while removing "stair-step" aliasing.
* **Decimation:** Quadric decimation to reduce triangle count for efficient rendering.

---

## Repository Structure

* **`train.ipynb`**: The main training loop. Loads the LiTS dataset, initializes the U-Net, logs to WandB, and saves checkpoints.
* **`hyperpara_opti.ipynb`**: Contains the Optuna study to find the best hyperparameters (learning rate, etc.) by maximizing the validation Dice score.
* **`inference_unet.ipynb`**: Loads the trained model (`trial-37_best.pth`), performs inference on the test set, calculates Dice scores, and saves the resulting segmentation maps.
* **`3d_reconstruction/`**: All the code for the 3D reconstruction part.
* **`dataset/`**: Scripts to load and wrap the dataset.
* **`unet/`**: Source code for the U-Net architecture.
* **`utility/`**: Miscellaneous code, image logging, computing dice scores...


---

## Project Report

For a detailed analysis of the methodology, mathematical background, and extensive results, please refer to our full project report:

**[Click here](./Tumor_3D_reconstruction.pdf)**