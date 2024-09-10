# DreamHOI

We present DreamHOI, a novel method for zero-shot synthesis of human-object interactions (HOIs), enabling a 3D human model to realistically interact with any given object based on a textual description.

[Website](https://dreamhoi.github.io/)
[Paper](TODO)

## Installation

Please carefully follow [installation.md](docs/installation.md) to set up your system for DreamHOI.

## Running

To run our pipeline, run
```sh
# inside dreamhoi
python main.py \
  --num_iterations 1 \
  --tag sit-ball \
  --smpl_texture /path/to/smpl/texture.png \
  --smpl_shape /path/to/smpl/shape.npy \
  --smpl_gender female \
  --smpl_variant smplh \
  --prompt "A photo of a person sitting on a ball, high detail, photography" \
  --prompt_human "A photo of a person, high detail, photography" \
  --negative_prompt "missing limbs, missing legs, missing arms" \
  --negative_prompt_human "ball, missing limbs, missing legs, missing arms" \
  --mesh_path /path/to/ball.obj \
  --mesh_normalize \
  --mesh_translation 0.0 0.0 -0.3 \
  --mesh_scale 0.3 \
  --mesh_rotation_deg 0.0 \
  --mesh_tilt_deg 0.0 \
  --checkpoint_interval 1000 \
  --use_wandb \
  --openpose_dir /path/to/openpose \
  --openpose_bin /path/to/openpose/build/examples/openpose/openpose.bin \
  --nerf_init_args ... \
  --nerf_refit_args ...
```
where:
* `num_iterations` is the number of times to reinitiate NeRF (≥ 0)
* `tag` is a unique name for this run, used for naming files. This also supports resuming from an interrupted run
* `smpl_texture` is the texture of SMPL for the generation, in .png (see [examples](https://dancasas.github.io/projects/SMPLitex/SMPLitex-dataset.html)). If not provided, we output pose parameters (bone rotations) without generating the final human mesh.
* `smpl_shape` is an .npy containing a (10,) array of SMPL shape parameters. Default is to predict shape by SMPLify
* `smpl_gender` is gender for SMPL models to use (male, female, or neutral). Note SMPL+H does not support neutral (see src/MultiviewSMPLifyX/cfg_files/ for defaults)
* `smpl_variant` is the variant of SMPL for our pipeline. Currently only `smplh` (default) and `smpl` are supported
* `prompt` is a prompt for supervising the overall HOI
* `prompt_human` is a prompt for supervising the human part
* `negative_prompt` is a negative prompt for supervising the overall HOI
* `negative_prompt_human` is a negative prompt for supervising the human part
* `mesh_path` is path to the mesh of the object (in a mesh format such as .obj, .glb). Note that we assume the mesh is Y-up, X-front
* `mesh_normalize` if set, normalize the mesh scale so it is approximately unit size
* `mesh_translation` is where to position the object in the scene in x, y, z (+x is front, +z is up)
* `mesh_scale` is a scalar to scale of the object mesh (mesh_scale 0.2–0.4 with mesh_normalize is used in the paper)
* `mesh_rotation_deg` rotates the object mesh (counterclockwise viewing from above)
* `mesh_tilt_deg` tilts the object mesh
* `checkpoint_interval` is the number of steps before saving a checkpoint model. For reference, by default, 10000 is run for each iteration
* `use_wandb` if set, tries to use weights & biases. This needs `wandb` installed and configured on the system (recommended)
* `openpose_dir` is the path to the OpenPose project directory
* `openpose_bin` is the path to the OpenPose built binary file (default is `{openpose_dir}/build/examples/openpose/openpose.bin`)
* `nerf_init_args`, `nerf_refit_args` are optional additional settings for NeRF fitting (see below)

[`main.py`](main.py) is a wrapper around our pipeline, and you can modify our pipeline by directly modifying it.

After running, DreamHOI outputs:
* The SMPL pose parameters of the generated human $\xi$
* The generated human mesh $M_\xi$
* The (transformed) mesh of the given object $M_{\text{obj}}$

Note that as in threestudio, the generated meshes are Z-up, and you can transform back to Y-up by a -π/2 rotation about the X axis.

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
    * `trainer.max_steps=10000`: number of steps to train NeRF during this fitting iteration
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
* For both of the above see [threestudio docs](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md) for a more complete set of options, and our config YAML [for NeRF initialization](https://github.com/hanwenzhu/MVDream-threestudio/blob/main/configs/mvdream-with-deepfloyd-with-mesh.yaml) and [for NeRF refitting](https://github.com/hanwenzhu/MVDream-threestudio/blob/main/configs/smpl-with-mesh-nerf-if.yaml) for default values.
* Note `system.composed_only` is by default true for NeRF re-fitting, i.e. MVDream is disabled (as discussed in paper). If you want to use MVDream guidance, you need to modify the [config](https://github.com/hanwenzhu/MVDream-threestudio/blob/main/configs/smpl-with-mesh-nerf-if.yaml) and use `random-multiview-camera-datamodule` for `data_type`, setting parameters under `data` as in the [initialization config](https://github.com/hanwenzhu/MVDream-threestudio/blob/main/configs/mvdream-with-deepfloyd-with-mesh.yaml).

## Tips to improve generation
* Tune hyperparameters:
  * The mixture ratio between DeepFloyd IF and MVDream `--nerf_*_args system.loss.lambda_*_sds`
    * The difference between DeepFloyd IF and MVDream guidance are discussed in the paper. On a high level, DeepFloyd IF has a much better understanding of text, while MVDream is good at multi-view consistent, detailed 3D generation.
  * Noise levels `--nerf_*_args system.(composed_)guidance.max_step_percent`
    * This parameter controls the upper bound *t* for noise levels, which are sampled from [0.02, *t*] uniformly. A higher value allows the guidance model to edit high-level features (e.g., changing semantic relation) and a lower value allows the guidance model to edit lower-level details. It may also be beneficial to set different values for DeepFloyd IF and MVDream.
  * Guidance scale `--nerf_*_args system.(composed_)guidance.guidance_scale`
    * Increasing the guidance scale (default 50) significantly helps guidance models to follow the text prompt more.
  * Regularizer weights `--nerf_*_args system.loss.lambda_*`
    * The different losses and regularizers are extensively discussed in the paper.
    * Note that sometimes setting these weights to 0, i.e. disabling some regularizers, may work better.
  * Tuning mesh position `--mesh_*`
    * It is beneficial to scale and place the mesh in the scene, so that it is natural for a human in the center of that scene to interact with the mesh.
    * Also make sure the mesh faces the front (+x) to make MVDream guidance work better.
  * Number of NeRF refittings `--num_iterations`
    * Sometimes different `num_iterations` (even 0) may be acceptable; we recommend trying 0–2.
  * Note that most `--nerf_*_args ...` hyperparameters can also follow a schedule; e.g. `system.guidance.max_step_percent=[0,0.98,0.50,8000]` to interpolate from 0.98 to 0.50 over steps 0–8000
* Prompting
  * Make the prompts more specific
    * Describe mesh accurately (e.g. "vintage tv" instead of "tv") to avoid the NeRF from generating a new object.
    * You may describe the semantic relation between the human and the object very specifically ("A person sitting on a ball. Her upper body sits on top of the ball, her elbows rest on her knees, and her feet rest on the floor.") which sometimes improve quality.
  * Negative prompts also help in producing a good human NeRF.
    * Negative prompts like "missing limbs" ensures the model produces an entire human NeRF (not just parts) so the pose estimation stage is much easier.
*	Less important parameters
  * Traning parameters like learning rate, batch size, number of steps per iteration
    * Reducing batch size and number of steps (to say 3000) can reduce training costs but may affect quality
  * Background color `--nerf_*_args system.background.*`

## License
DreamHOI is released under MIT License. Some parts of this project uses third-party software. See [LICENSE](LICENSE) for their respective notices and licenses.
