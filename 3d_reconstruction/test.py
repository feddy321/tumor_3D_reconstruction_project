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

nii = load_nii("data/nii_pre_unet/segmentation-9.nii")
volume = nii_to_numpy(nii)
#shape : 
print(volume.shape)

npy = np.load('data/maps_post_unet/post_unet_9.npy')
# Afficher la forme du tableau
print('La forme du tableau est :', npy.shape)

