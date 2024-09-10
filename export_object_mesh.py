import os
from typing import Optional, List

import numpy as np
import trimesh

def mesh_from_path(
    file_path: str,
    # if the model is Y-up instead of Z-up
    # (TODO it may be easier to ask the user to specify a 4x4 matrix instead for the following)
    y_up: bool = True,
    normalize: bool = True,
    scale: Optional[List[float]] = None,
    # Rotation about Z-axis (up)
    rotation_deg: Optional[float] = None,
    # Rotation about X-axis
    tilt_deg: Optional[float] = None,
    translation: Optional[List[float]] = None,
) -> trimesh.Trimesh:
    """This is a copy of Mesh.from_path in src/MVDream-threestudio/threestudio/models/mesh.py.
    Changes in this function and that function should be synchronized.
    """
    mesh: trimesh.Trimesh = trimesh.load(file_path, force="mesh") # type: ignore

    if y_up:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    if normalize:
        mesh.apply_translation(-mesh.centroid)
        # Can also use mesh.scale for the max diameter
        mesh.apply_scale(1 / np.linalg.norm(mesh.vertices, axis=1).mean())
    if scale is not None:
        mesh.apply_scale(scale)
    if rotation_deg is not None:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(rotation_deg * np.pi / 180.0, [0, 0, 1]))
    if tilt_deg is not None:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(tilt_deg * np.pi / 180.0, [1, 0, 0]))
    if translation is not None:
        mesh.apply_translation(translation)

    return mesh
