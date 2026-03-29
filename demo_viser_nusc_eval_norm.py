# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
import os
import glob
import time
import threading
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from vggt.models.vggt_decoder_global import VGGT_decoder_global
from vggt.utils.load_fn import load_and_preprocess_images_pos
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_NUSC_ROOT = REPO_ROOT / "data" / "nuscenes"
DEFAULT_CKPT = REPO_ROOT / "ckpt" / "checkpoint_5.pt"
SENSOR_LIST = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def transform_matrix(rotation: list[float], translation: list[float], inverse: bool = False) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    if inverse:
        matrix = np.linalg.inv(matrix)
    return matrix

class NuscenesDatasetEval():
    def __init__(
        self,
        dataroot: str | Path = DEFAULT_NUSC_ROOT,
        version: str = "v1.0-trainval",
    ):
        super().__init__()
        self.dataroot = Path(dataroot)
        self.version = version
        self.cam_list = SENSOR_LIST
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=True)
        self.sequence_list = self.nusc.scene

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        start_index: int = 0,
    ) -> dict:
        images_path = []
        extris = []
        intris = []
        if seq_index is None:
            seq_index = 0
        if img_per_seq is None:
            img_per_seq = 15

        scene = self.sequence_list[seq_index]
        current_token = scene["first_sample_token"]

        for _ in range(start_index):
            sample = self.nusc.get("sample", current_token)
            if not sample["next"]:
                raise ValueError(
                    f"Scene {seq_index} does not have start_index={start_index}."
                )
            current_token = sample["next"]

        for _ in range(img_per_seq):
            sample = self.nusc.get("sample", current_token)
            for cam_pos in self.cam_list:
                sample_data = self.nusc.get("sample_data", sample["data"][cam_pos])
                ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
                calibrated = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

                global_to_ego = transform_matrix(
                    ego_pose["rotation"], ego_pose["translation"], inverse=True
                )
                ego_to_camera = transform_matrix(
                    calibrated["rotation"], calibrated["translation"], inverse=True
                )
                global_to_camera = ego_to_camera @ global_to_ego

                images_path.append(str(self.dataroot / sample_data["filename"]))
                extris.append(global_to_camera[:3, :4])
                intris.append(np.asarray(calibrated["camera_intrinsic"], dtype=np.float32))

            if not sample["next"]:
                break
            current_token = sample["next"]

        return [images_path, extris, intris]

def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str | list[str] | None = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        
    
    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
        
    if use_point_map == False:
        depth_map = pred_dict["depth"]  # (S, H, W, 1)
        depth_conf = pred_dict["depth_conf"]  # (S, H, W)
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
        
    else:
        world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
        conf_map = pred_dict["world_points_conf"]  # (S, H, W)
        world_points = world_points_map
        conf = conf_map


    init_threshold_val = np.percentile(conf, 40)
    pts_world_points_map = world_points[conf>init_threshold_val].reshape(-1,3)
    np.save('./pts/predict_pts.npy',pts_world_points_map)

    # extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    # intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # # Compute world points from depth if not using the precomputed point map
    # if not use_point_map:
    #     world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
    #     conf = depth_conf
    # else:

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server

# Helper functions for sky segmentation


def apply_sky_segmentation(conf: np.ndarray, image_folder: str | list[str]) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    try:
        from visual_util import segment_sky, download_file_from_url
    except ImportError:
        from models.streamvggt.src.visual_util import segment_sky, download_file_from_url

    if isinstance(image_folder, list):
        image_files = image_folder[:S]
        common_root = os.path.commonpath(image_files) if image_files else "."
        sky_masks_dir = os.path.join(common_root, "_sky_masks")
    else:
        sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
        image_files = sorted(glob.glob(os.path.join(image_folder, "*")))[:S]
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files)):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder",
    type=str,
    default=None,
    help="Optional image folder for sky masking. Leave unset to use the selected NuScenes images.",
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument(
    "--dataroot",
    type=Path,
    default=DEFAULT_NUSC_ROOT,
    help="NuScenes dataroot containing samples/ and v1.0-* metadata.",
)
parser.add_argument("--version", default="v1.0-trainval", help="NuScenes metadata version.")
parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT, help="Checkpoint path.")
parser.add_argument("--scene-index", type=int, default=101, help="NuScenes scene index.")
parser.add_argument(
    "--start-index",
    type=int,
    default=5,
    help="Starting sample offset inside the selected scene.",
)
parser.add_argument(
    "--img-per-seq",
    type=int,
    default=15,
    help="Number of sequential samples to load from the scene.",
)


def main():
    """
    Main function for the VGGT demo with viser for 3D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --mask_sky: Apply sky segmentation to filter out sky points
    """
    
    args = parser.parse_args()
    os.makedirs("./pts", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    nusc = NuscenesDatasetEval(dataroot=args.dataroot, version=args.version)
    image_names, extris, intris = nusc.get_data(
        seq_index=args.scene_index,
        img_per_seq=args.img_per_seq,
        start_index=args.start_index,
    )
    nusc_input = load_and_preprocess_images_pos(image_names,extris,intris)
    # print("extrinsics",nusc_input["extrinsics"].size())# 1 30 3 4
    # Use the provided image folder path
    # nusc_data = NuscenesDataset_MultiView_eval(**common_conf)
    # data = nusc_data.get_data()
    # print(f"Preprocessed images shape: {data['images'][0].shape}")
    print(f"Found {len(image_names)} images")

    print("Initializing and loading VGGT model...")
    model = VGGT_decoder_global(enable_track=False, enable_point=False, enable_depth=True).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
     
    print("Running inference...")
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # , 'pose_enc_list'
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device == "cuda", dtype=dtype):

            # print(nusc_input["images"].size()) # 6*seq 3 H W?
            S,_,H,W = nusc_input["images"].size()
            # nusc_input norm
            real_world_extrinics = nusc_input["extrinsics"] # B S 3 4
            N_cam = 6
            N_seq = S//N_cam
            extrinsics_homog = torch.cat(
                    [
                        real_world_extrinics,
                        torch.zeros((1, S, 1, 4), device=device),
                    ],
                    dim=-2,
                )
            extrinsics_homog[:, :, -1, -1] = 1.0
            first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
            # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
            new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))[...,:3,:]  # (B,N,3,4)
            nusc_input["extrinsics"] = new_extrinsics
            print(new_extrinsics.size())
            
            start = time.time()          # CPU记录开始时间
            predictions = model(images= nusc_input["images"],others=nusc_input)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time()            # CPU记录结束时间
            
            N_cam = 6
            N_seq = S//N_cam
            nusc_input["images"] = nusc_input["images"].reshape(N_cam,N_seq,3,H,W)


            print("inference_times_is:  ", (end-start)*1000,'ms')
            
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    if ("mv_enc_list" in predictions.keys()) is False:
        with torch.cuda.amp.autocast(enabled=device == "cuda", dtype=torch.float64):
    
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], nusc_input["images"].shape[-2:])
    else:
        frame_extrinsic, _ = pose_encoding_to_extri_intri(predictions["seq_enc_list"][-1], nusc_input["images"].shape[-2:]) # 1 9 3 4
        relative_extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["mv_enc_list"][-1], nusc_input["images"].shape[-2:])# 1 6 3 4

        a = torch.zeros([1,S,1,3], device=device)
        b = torch.ones([1,S,1,1], device=device)
        c = torch.cat([a,b],dim=-1) # 1 S 1 4
        relative_pose = relative_extrinsic.unsqueeze(2).expand(1,N_cam,N_seq,3,4).reshape(1,-1,3,4)
        relative_pose = torch.cat([relative_pose,c],dim=-2)
        print("in",intrinsic.size())
        intrinsic = intrinsic.unsqueeze(2).expand(1,N_cam,N_seq,3,3).reshape(1,-1,3,3)
        frame_pose = frame_extrinsic.unsqueeze(1).expand(1,N_cam,N_seq,3,4).reshape(1,-1,3,4)
        frame_pose = torch.cat([frame_pose,c],dim=-2)
        # if predictions["scale"] is not None:
        #     frame_pose[...,:3,3:] = frame_pose[...,:3,3:]*predictions["scale"]
        #     relative_pose[...,:3,3:] = relative_pose[...,:3,3:]*predictions["scale"]
        #     predictions["depth"] = predictions["depth"]*predictions["scale"]
            
        extrinsic = relative_pose.matmul(frame_pose)[...,:3,:]


    # first_cam_extrinsic_inv = closed_form_inverse_se3(relative_pose[:, 0])
    # relative_pose = torch.matmul(relative_pose, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)
    # extrinsic = relative_pose.matmul(extrinsic)[:,:,:3,:]
    # extrinsic[...,-1] /=  20
    # print(relative_pose[:,0,...],extrinsic[:,0,...])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].detach().to(torch.float).cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    print("Starting viser visualization...")

    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder or image_names,
    )
    print("Visualization complete")


if __name__ == "__main__":
    main()
