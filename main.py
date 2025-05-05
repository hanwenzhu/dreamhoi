import argparse
import os
import shutil
import subprocess
import sys
from typing import Optional, List

from omegaconf import OmegaConf

DREAMHOI_PATH = os.path.dirname(os.path.abspath(__file__))
THREESTUDIO_PATH = os.path.join(DREAMHOI_PATH, "src", "MVDream-threestudio")
SMPLIFY_DATA_PATH = os.path.join(DREAMHOI_PATH, "smplify")
SMPLIFY_PATH = os.path.join(DREAMHOI_PATH, "src", "MultiviewSMPLifyX")
PYTHON_PATH = sys.executable or "python"  # or os.path.join(DREAMHOI_PATH, "venv", "bin", "python")

def run_nerf(
    tag: str, prompt: str, prompt_human: str, negative_prompt: str, negative_prompt_human: str,
    mesh_path: str, mesh_normalize: bool, mesh_translation: List[float], mesh_scale: float, mesh_rotation_deg: float, mesh_tilt_deg: float,
    checkpoint_interval: int, use_wandb: bool, args: Optional[List[str]],
    initialization: bool = True, smpl_mesh_path: Optional[str] = None,
):
    """Fit a NeRF with an object mesh using threestudio.

    Args:
        tag: name of the experiment
        initialization: whether to initialize NeRF from scratch or from smpl_mesh_path
        smpl_mesh_path: a mesh from which to initialize the human NeRF
    
    Returns: folder of the outputs in src/MVDream-threestudio/outputs (for postprocessing)
    """
    command = [
        PYTHON_PATH,
        os.path.join(THREESTUDIO_PATH, "launch.py"),
        "--train", "--gpu", "0", # run on first visible GPU
    ]

    if initialization:
        system_name = "mvdream-with-deepfloyd-with-mesh"
    else:
        system_name = "smpl-with-mesh-nerf-if"
    experiment_name = os.path.join(system_name, tag)
    experiment_path = os.path.join(THREESTUDIO_PATH, "outputs", experiment_name)
    last_ckpt_path = os.path.join(experiment_path, "ckpts", "last.ckpt")

    # when last.ckpt is initialized it points to itself
    # this means the directory is initialized but training has not reached first checkpoint
    if os.path.islink(last_ckpt_path) and not os.path.exists(last_ckpt_path):
        print(f"[dreamhoi] Removing {experiment_path} (bad last.ckpt)")
        shutil.rmtree(experiment_path)
    
    if os.path.exists(last_ckpt_path):
        print("[dreamhoi] Found existing model, resuming training")
        command.extend([
            "--config", os.path.join(experiment_path, "configs", "parsed.yaml"),
            f"resume={last_ckpt_path}"
        ])
    else:
        command.extend([
            "--config", os.path.join(THREESTUDIO_PATH, "configs", f"{system_name}.yaml"),
        ])
    
    if smpl_mesh_path is not None:
        command.append(
            f"system.geometry.shape_init=mesh:{smpl_mesh_path}"
        )
    
    command.extend([
        "use_timestamp=false", f"tag={tag}",
        f"checkpoint.every_n_train_steps={checkpoint_interval}",
        f"system.loggers.wandb.enable={use_wandb}",
        f"system.loggers.wandb.name={experiment_name}",
        f"system.composed_prompt_processor.prompt={prompt}",
        f"system.prompt_processor.prompt={prompt_human}",
        f"system.composed_prompt_processor.negative_prompt={negative_prompt}",
        f"system.prompt_processor.negative_prompt={negative_prompt_human}",
        f"system.composed_renderer.mesh_path={mesh_path}",
        f"system.composed_renderer.mesh.normalize={mesh_normalize}",
        f"system.composed_renderer.mesh.scale={mesh_scale}",
        f"system.composed_renderer.mesh.translation={mesh_translation}",
        f"system.composed_renderer.mesh.rotation_deg={mesh_rotation_deg}",
        f"system.composed_renderer.mesh.tilt_deg={mesh_tilt_deg}",
    ])

    if args is not None:
        command.extend(args)

    print(f"[dreamhoi] Running threestudio with command: {command}")
    subprocess.run(command, check=True, cwd=THREESTUDIO_PATH)

    return experiment_name

def run_openpose(
    openpose_dir: str, openpose_bin: Optional[str], rgb_dir: str, keypoints_dir: str
):
    """Runs openpose, with input images from `rgb_dir` and output keypoint JSON files to `keypoints_dir`"""
    if os.path.isfile(os.path.join(keypoints_dir, "99_keypoints.json")):
        print("[dreamhoi] Keypoints exist; skipping openpose stage")
        return

    print("[dreamhoi] Running OpenPose")

    # Binary path
    if openpose_bin is None:
        openpose_bin_path = os.path.join(openpose_dir, "build", "examples", "openpose", "openpose.bin")
    else:
        openpose_bin_path = openpose_bin
    openpose_env_variables = {
        **os.environ,
        "LD_LIBRARY_PATH": (os.path.join(openpose_dir, "build", "src", "openpose") +
            ":" + os.path.join(openpose_dir, "build", "caffe", "lib64") +
            ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    }
    # Run OpenPose
    command = [
        openpose_bin_path,
        "--image_dir", rgb_dir,
        "--write_json", keypoints_dir,
        "--display", "0",
        "--render_pose", "0",
        # predict hand/face keypoints, for SMPL+H/SMPL-X
        "--hand", "--face",
    ]
    print(f"[dreamhoi] Running OpenPose with command: {command}")
    subprocess.run(command, check=True, env=openpose_env_variables, cwd=openpose_dir)
    

def predict_smpl(
    experiment_name: str, smpl_variant, smpl_texture, smpl_shape, smpl_gender,
    openpose_dir, openpose_bin, predict_from="no_mesh"
):
    experiment_config = os.path.join(THREESTUDIO_PATH, "outputs", experiment_name, "configs", "parsed.yaml")
    config = OmegaConf.load(experiment_config)
    trainer_max_steps = config.trainer.max_steps
    experiment_save_dir = os.path.join(THREESTUDIO_PATH, "outputs", experiment_name, "save")
    rgb_dir = os.path.join(experiment_save_dir, f"it{trainer_max_steps}-test-{predict_from}-rgb")
    keypoints_dir = os.path.join(experiment_save_dir, f"it{trainer_max_steps}-test-{predict_from}-openpose")
    metadata_dir = os.path.join(experiment_save_dir, f"it{trainer_max_steps}-test-{predict_from}-metadata")

    run_openpose(openpose_dir=openpose_dir, openpose_bin=openpose_bin, rgb_dir=rgb_dir, keypoints_dir=keypoints_dir)

    data_dir = os.path.join(SMPLIFY_DATA_PATH, experiment_name)
    out_dir = os.path.join(data_dir, smpl_variant)
    os.makedirs(data_dir, exist_ok=True)

    # function for ln -sfn
    def force_symlink(src, dst):
        if os.path.exists(dst):
            assert os.path.islink(dst)
        if os.path.islink(dst):
            os.unlink(dst)
        os.symlink(src, dst)

    force_symlink(rgb_dir, os.path.join(data_dir, "color"))
    force_symlink(keypoints_dir, os.path.join(data_dir, "keypoints"))
    force_symlink(metadata_dir, os.path.join(data_dir, "meta"))

    smpl_mesh_out_path = Path(out_dir) / "smpl_mesh.obj"
    smpl_param_out_path = Path(out_dir) / "smpl_param.pkl"

    if smpl_mesh_out_path.is_file():
    logger.info("Output mesh exists; skipping SMPLifyX stage")

    else:
        # Run SMPLifyX
        print("[dreamhoi] Running SMPLify-X")
        command = [
            PYTHON_PATH, "main.py",
            "--config", f"cfg_files/fit_{smpl_variant}.yaml",
            "--data_folder", data_dir,
            "--output_folder", out_dir,
        ]
        if smpl_shape is not None:
            command.extend([
                "--mesh_betas_fn", smpl_shape
            ])
        if smpl_texture is not None:
            command.extend([
                "--mesh_texture_fn", smpl_texture
            ])
        if smpl_gender is not None:
            command.extend([
                "--gender", smpl_gender
            ])
        print(f"[dreamhoi] Running SMPLify with command: {command}")
        subprocess.run(command, check=True, cwd=SMPLIFY_PATH)
    
    return smpl_mesh_out_path, smpl_param_out_path

def run_full(
    num_iterations, tag, prompt, prompt_human, negative_prompt, negative_prompt_human,
    mesh_path, mesh_normalize, mesh_translation, mesh_scale, mesh_rotation_deg, mesh_tilt_deg,
    checkpoint_interval, use_wandb, nerf_init_args, nerf_refit_args,
    smpl_variant, smpl_texture, smpl_shape, smpl_gender, openpose_dir, openpose_bin
):
    print("[dreamhoi] Running NeRF initialization")
    experiment_name = run_nerf(
        tag=f"{tag}_0", prompt=prompt, prompt_human=prompt_human, negative_prompt=negative_prompt, negative_prompt_human=negative_prompt_human,
        mesh_path=mesh_path, mesh_normalize=mesh_normalize, mesh_translation=mesh_translation, mesh_scale=mesh_scale, mesh_rotation_deg=mesh_rotation_deg, mesh_tilt_deg=mesh_tilt_deg,
        checkpoint_interval=checkpoint_interval, use_wandb=use_wandb, args=nerf_init_args,
        initialization=True
    )
    smpl_mesh_path, _ = predict_smpl(experiment_name, smpl_variant, smpl_texture, smpl_shape, smpl_gender, openpose_dir, openpose_bin)
    for i in range(num_iterations):
        print(f"[dreamhoi] Running NeRF re-fitting iteration {i}")
        experiment_name = run_nerf(
            tag=f"{tag}_{i + 1}", prompt=prompt, prompt_human=prompt_human, negative_prompt=negative_prompt, negative_prompt_human=negative_prompt_human,
            mesh_path=mesh_path, mesh_normalize=mesh_normalize, mesh_translation=mesh_translation, mesh_scale=mesh_scale, mesh_rotation_deg=mesh_rotation_deg, mesh_tilt_deg=mesh_tilt_deg,
            checkpoint_interval=checkpoint_interval, use_wandb=use_wandb, args=nerf_refit_args,
            initialization=False, smpl_mesh_path=smpl_mesh_path,
        )
        smpl_mesh_path, smpl_param_path = predict_smpl(experiment_name, smpl_variant, smpl_texture, smpl_shape, smpl_gender, openpose_dir, openpose_bin)
    return smpl_mesh_path, smpl_param_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the entire DreamHOI pipeline")
    parser.add_argument("--num_iterations", type=int, required=True, help="Number of times T to reinitiate NeRF")
    parser.add_argument("--tag", type=str, required=True, help="Tag description of this experiment (e.g. sit-ball)")
    parser.add_argument("--smpl_texture", type=str, default=None, help="If provided, apply this texture image as the human identity, and output a human mesh using this texture.")
    parser.add_argument("--smpl_shape", type=str, default=None, help="If provided, use this .npy file as human shape parameters (shape (10,) 'betas' in SMPL) for the mesh. If not specified, we predict this parameter using SMPLify-X.")
    parser.add_argument("--smpl_gender", type=str, default=None, help="Which gender for SMPL models to use (male, female, or neutral). See src/MultiviewSMPLifyX/cfg_files/*.yaml for defaults")
    parser.add_argument("--smpl_variant", default="smplh", help="Variant of SMPL to use (smpl or smplh)")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for the HOI (e.g. A photo of person sitting on a ball, high detail, photography)")
    parser.add_argument("--prompt_human", type=str, default="A photo of a person, high detail, photography", help="Prompt for the human-only render (e.g. A photo of a person, high detail, photography)")
    parser.add_argument("--negative_prompt", type=str, default="missing limbs, missing legs, missing arms", help="Negative prompt for the HOI")
    parser.add_argument("--negative_prompt_human", type=str, default="missing limbs, missing legs, missing arms", help="Negative prompt for the human-only render (e.g. ball, missing limbs, missing legs, missing arms)")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to object mesh (e.g. /path/to/ball.obj)")
    parser.add_argument("--mesh_normalize", action="store_true", help="Normalize the mesh scale so it is approximately unit size")
    parser.add_argument("--mesh_translation", type=float, nargs=3, default=[0., 0., 0.], help="Translate the object mesh (+x is front, +z is up)")
    parser.add_argument("--mesh_scale", type=float, default=0.5, help="Scale the object mesh size by a constant")
    parser.add_argument("--mesh_rotation_deg", type=float, default=0., help="Rotate the object mesh (counterclockwise viewing from above)")
    parser.add_argument("--mesh_tilt_deg", type=float, default=0., help="Tilt the object mesh")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Save checkpoint every number of steps (default 10000 steps total)")
    parser.add_argument("--use_wandb", action="store_true", help="Use weights & biases (recommended)")
    parser.add_argument("--openpose_dir", type=str, required=True, help="Path to OpenPose project directory")
    parser.add_argument("--openpose_bin", type=str, default=None, help="Path to OpenPose binary (/path/to/openpose.bin). Default is {openpose_dir}/build/examples/openpose/openpose.bin")
    parser.add_argument("--nerf_init_args", type=str, nargs="*", default=None, help="Extra threestudio arguments for NeRF initialization")
    parser.add_argument("--nerf_refit_args", type=str, nargs="*", default=None, help="Extra threestudio arguments for NeRF re-fitting")

    args = parser.parse_args()

    smpl_mesh_path, smpl_param_path = run_full(
        num_iterations=args.num_iterations, tag=args.tag, prompt=args.prompt, prompt_human=args.prompt_human, negative_prompt=args.negative_prompt, negative_prompt_human=args.negative_prompt_human,
        mesh_path=args.mesh_path, mesh_normalize=args.mesh_normalize, mesh_translation=args.mesh_translation, mesh_scale=args.mesh_scale, mesh_rotation_deg=args.mesh_rotation_deg, mesh_tilt_deg=args.mesh_tilt_deg,
        checkpoint_interval=args.checkpoint_interval, use_wandb=args.use_wandb, nerf_init_args=args.nerf_init_args, nerf_refit_args=args.nerf_refit_args,
        smpl_variant=args.smpl_variant, smpl_texture=args.smpl_texture, smpl_shape=args.smpl_shape, smpl_gender=args.smpl_gender,
        openpose_dir=args.openpose_dir, openpose_bin=args.openpose_bin
    )
    print(f"[dreamhoi] Run finished")
    print(f"  Resulting human mesh: {smpl_mesh_path}")
    print(f"  Resulting SMPL parameters: {smpl_param_path}")
    
    from export_object_mesh import mesh_from_path
    object_mesh = mesh_from_path(file_path=args.mesh_path, y_up=True, normalize=args.mesh_normalize, scale=args.mesh_scale, rotation_deg=args.mesh_rotation_deg, tilt_deg=args.mesh_tilt_deg, translation=args.mesh_translation)
    object_mesh_out_path = os.path.join(os.path.dirname(smpl_mesh_path), "object_mesh.obj")
    object_mesh.export(object_mesh_out_path)
    print(f"  Given object mesh: {object_mesh_out_path}")
