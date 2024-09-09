# DreamHOI

We present DreamHOI, a novel method for zero-shot synthesis of human-object interactions (HOIs), enabling a 3D human model to realistically interact with any given object based on a textual description.

[Website](https://dreamhoi.github.io/)
[Paper](TODO)

## Installation

Please follow [installation.md](docs/installation.md) to set up your system for DreamHOI.

## Running

To run our pipeline, run
```sh
# inside dreamhoi
python main.py \
  --num_iterations 1 \
  --tag sit-ball \
  --smpl_texture /path/to/smpl/texture.png \
  --prompt "A photo of a person sitting on a ball, high detail, photography" \
  --prompt_human "A photo of a person, high detail, photography" \
  --negative_prompt "missing limbs, missing legs, missing arms" \
  --negative_prompt_human "ball, missing limbs, missing legs, missing arms" \
  --mesh_path /path/to/ball.obj \
  --mesh_translation "[0.0, 0.0, -0.3]" \
  --mesh_scale 0.3 \
  --mesh_rotation_deg 0.0 \
  --mesh_tilt_deg 0.0 \
  --checkpoint_interval 1000 \
  --use_wandb \
  --smpl_variant smplh \
  --openpose_path /path/to/openpose/build/examples/openpose/openpose.bin \
  --openpose_cwd /path/to/openpose
```
where:
* `num_iterations` is the number of times to reinitiate NeRF
* `tag` is a name of this experiment (for naming files)
* `smpl_texture` is the texture of SMPL for the generation, in .png (see [examples](https://dancasas.github.io/projects/SMPLitex/SMPLitex-dataset.html)). If not provided, we output pose parameters (bone rotations) without generating the final human mesh.
* `smpl_shape` is an .npy containing a (10,) array of SMPL shape parameters
* `prompt` is a prompt for supervising the overall HOI
* `prompt_human` is a prompt for supervising the human part
* `negative_prompt` is a negative prompt for supervising the overall HOI
* `negative_prompt_human` is a negative prompt for supervising the human part
* `mesh_path` is path to the mesh of the object (in .obj, .glb, etc)
* `mesh_translation` is where to position the object in the scene (+x is front, +z is up)
* `mesh_scale` is the size of the object
* `mesh_rotation_deg` rotates the object mesh (counterclockwise viewing from above)
* `mesh_tilt_deg` tilts the object mesh
* `checkpoint_interval` is the number of steps before saving a checkpoint model. For reference, by default, 10000 is run for each iteration
* `use_wandb` if set, tries to use weights & biases. This needs `wandb` installed and configured on the system (recommended)
* `smpl_variant` is the variant of SMPL for our pipeline. Currently only `smplh` (default) and `smpl` are supported
* `openpose_dir` is the path to the OpenPose project directory
* `openpose_bin` is the path to the OpenPose built binary file (default is `[openpose_dir]/build/examples/openpose/openpose.bin`)

[`main.py`](main.py) is a wrapper around our pipeline, and you can modify our pipeline by directly modifying it.

##### Extra configurations
You may also add
* `--nerf_init_args arg1=value1 arg2=value2 ...` to supply extra arguments to threestudio as hyperparemters. Commonly used args include:
  * Data:
    * `data.batch_size=8`: batch size for rendering NeRF (due to MVDream, must be a multiple of 4)
    * `data.width=128`, `data.height=128`: resolution for rendering NeRF
    * `data.n_val_views=8`: number of views to render for validation (usually multiple of 4)
    * `system.background.eval_color=[1,1,1]`: RGB value for background during evaluation
  * Training:
    * `system.composed_only`: only use DeepFloyd IF and not MVDream
    * `system.background.random_aug_prob=0.5`: probability of randomly setting the background during training
    * `system.guidance.guidance_scale=50`: guidance scale for MVDream
    * `system.guidance.max_step_percent=0.98`: upper bound of noise level to sample from for MVDream (can be a schedule)
    * `system.composed_renderer.mesh.normalize=...`: whether to normalize the position and scale of the object mesh (default true)
    * `system.composed_guidance.guidance_scale=50`: guidance scale for DeepFloyd IF
    * `system.composed_guidance.max_step_percent=0.98`: upper bound of noise level to sample from for DeepFloyd IF (can be a schedule)
  * Loss weights and additional regularizers:
    * `system.loss.lambda_composed_sds=0.9`: weight of HOI supervision ($\lambda_{\text{SDS-HO}}$ in paper)
    * `system.loss.lambda_composed_individual_sds=0.05`: weight of human-only supervision by DeepFloyd IF ($\lambda_{\text{SDS-H}}$ in paper)
    * `system.loss.lambda_sds=0.05`: weight of human-only supervision by MVDream ($\lambda_{\text{SDS-H-MV}}$ in paper)
    * `system.loss.lambda_sparsity=0.0`: weight to discourage large NeRF. Usually not needed with `lambda_sparsity_above_threshold` set below
    * `system.loss.sparsity_threshold=0.2`: an upper threshold for the average density of NeRF as a portion of a rendered image ($\eta$ in paper)
    * `system.loss.lambda_sparsity_above_threshold=10000.0` penalty for density above `sparsity_threshold` ($\lambda_{\text{SA}}$ in paper; 1000 to 5000 would be a softer bound that may also work)
    * `system.loss.lambda_mesh_occlusion=0.0` weight to discourage NeRF from covering the mesh. Usually not needed with `lambda_mesh_occlusion_above_threshold` set below
    * `system.loss.mesh_occlusion_threshold=0.4` an upper threshold for the proportion of mesh occluded by the NeRF
    * `system.loss.lambda_mesh_occlusion_above_threshold=0.0` penalty for occlusion above `mesh_occlusion_threshold`
    * `system.loss.lambda_opaque=0.0` encourages NeRF to be either fully transparent or fully opaque
    * `system.loss.lambda_intersection=1.0` penalizes intersection between NeRF and object mesh ($\lambda_{\text{I}}$ in paper)
* `--nerf_refit_args arg1=value1 arg2=value2 ...` to supply extra arguments to threestudio as hyperparemters.
  * Mostly the same as parameters above. Note the default values of `lambda_sparsity_above_threshold`, `max_step_percent`, `random_aug_prob`, etc. are adjusted.
* Please also see [threestudio docs](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md) for a more complete set of options, and our config YAML [for NeRF initialization](src/MVDream-threestudio/configs/mvdream-with-deepfloyd-with-mesh.yaml) and [for NeRF refitting](src/MVDream-threestudio/configs/smpl-with-mesh-nerf-if.yaml) for default values.
* Note `system.composed_only` is by default true for NeRF re-fitting, i.e. MVDream is disabled (as discussed in paper). If you want to use MVDream guidance, you need to modify the [config](src/MVDream-threestudio/configs/smpl-with-mesh-nerf-if.yaml) and use `random-multiview-camera-datamodule` for `data_type`, setting parameters under `data` as in the [initialization config](src/MVDream-threestudio/configs/mvdream-with-deepfloyd-with-mesh.yaml).

## Tricks
1. Tune hyperparams
    1. IF/MV ratios
    2. individual time steps
    3. Loss lambdas - sometimes disabling some losses works better
    4. Tune mesh position & rotation (make sure face front)
    5. Learning rate (maybe not so important)
    6. Background (random aug, eval color -> openpose)
2. Prompting
    1. Increase guidance_scale
    2. Make prompt more specific (limb positions etc, IF understands very specific prompts; describe mesh accurately so doesn't make new object)
    3. Negative prompting
3. 

## License
Some parts of this project uses third-party software. See [LICENSE](LICENSE) for their respective notices and licenses.
