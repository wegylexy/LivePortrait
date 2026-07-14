# coding: utf-8

"""
Pipeline of LivePortrait (Human)
"""

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
import ffmpegcv
import threading
from queue import Queue
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream
from .utils.crop import prepare_paste_back, paste_back
from .utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, is_square_video, calc_motion_multiplier
from .utils.filter import smooth
from .utils.rprint import rlog as log
# from .utils.viz import viz_lmk
from .live_portrait_wrapper import LivePortraitWrapper


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)

def read_video_chunks(video_path, chunk_size=128, max_dim=1280, division=2):
    def read_loop(q, path, c_size, m_dim, div):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize_to_limit(frame, m_dim, div)
            frames.append(frame)
            if len(frames) >= c_size:
                q.put(frames)
                frames = []
        if frames:
            q.put(frames)
        q.put(None)
        cap.release()

    q = Queue(maxsize=3)
    t = threading.Thread(target=read_loop, args=(q, video_path, chunk_size, max_dim, division))
    t.daemon = True
    t.start()

    while True:
        chunk = q.get()
        if chunk is None:
            break
        yield chunk


class CustomVideoWriterNV(ffmpegcv.ffmpeg_writer.FFmpegWriterNV):
    def _init_video_stream(self):
        # Default average bitrate to 1M if not specified
        bitrate_str = f"-b:v {self.bitrate} " if self.bitrate else "-b:v 1M "
        rtsp_str = f"-f rtsp" if self.filename.startswith("rtsp://") else ""
        filter_str = (
            ""
            if self.resize == self.size
            else f"-vf scale={self.resize[0]}:{self.resize[1]}"
        )
        # Detailed NVENC parameters for animated portrait on green screen:
        # - -rc vbr -cq 28: Use Variable Bitrate with Constant Quality target of 28.
        #   A CQ of 28 offers a great sweet spot for clean talking-head/portrait visuals at a low size.
        #   Since green screen backgrounds are flat/monochrome, they compress extremely well.
        # - -b:v 1M -maxrate 1.5M -bufsize 2M: Target average 1Mbps, capping spikes at 1.5Mbps 
        #   with a 2MB buffer. This is ideal for small facial expressions and lipsync.
        # - -spatial-aq 1: Spatial Adaptive Quantization adjusts QP within a frame, shifting bits
        #   away from flat areas (like the green screen) to high-detail areas (eyes, mouth).
        # - -temporal-aq 1: Temporal Adaptive Quantization adjusts QP across frames based on motion,
        #   maintaining high fidelity for talking head animations over time.
        # - -movflags +faststart: Moves the MOOV atom to the beginning of the file so the video
        #   can start playing immediately before it is fully downloaded (great for web streaming).
        self.ffmpeg_cmd = (
            f"ffmpeg -y -loglevel error "
            f"-f rawvideo -pix_fmt {self.pix_fmt} -s {self.width}x{self.height} -r {self.fps} -i pipe: "
            f"-preset {self.preset} -rc vbr -cq 28 {bitrate_str}-maxrate 1.5M -bufsize 2M "
            f"-spatial-aq 1 -temporal-aq 1 "
            f"-r {self.fps} -gpu {self.gpu} -c:v {self.codec} "
            f"{filter_str} {rtsp_str} "
            f'-pix_fmt yuv420p -movflags +faststart "{self.filename}"'
        )
        from ffmpegcv.ffmpeg_writer import run_async
        self.process = run_async(self.ffmpeg_cmd)


def create_custom_video_writer_nv(filename, codec='h264', fps=30, pix_fmt='bgr24', gpu=0, bitrate=None, resize=None, preset=None):
    from ffmpegcv.ffmpeg_writer import get_num_NVIDIA_GPUs, IN_COLAB
    numGPU = get_num_NVIDIA_GPUs()
    assert numGPU
    gpu = int(gpu) % numGPU if gpu is not None else 0
    if codec is None:
        codec = "h264_nvenc"
    elif not isinstance(codec, str):
        codec = "h264_nvenc"
    elif codec.endswith("_nvenc"):
        codec = codec
    else:
        codec = codec + "_nvenc"
    assert codec in ["hevc_nvenc", "h264_nvenc"], "codec should be `hevc_nvenc` or `h264_nvenc`"
    assert resize is None or len(resize) == 2

    vid = CustomVideoWriterNV()
    vid.fps = fps
    vid.codec, vid.pix_fmt, vid.filename = codec, pix_fmt, filename
    vid.gpu = gpu
    vid.bitrate = bitrate
    vid.resize = resize
    vid.preset = preset if preset is not None else ("default" if IN_COLAB else "p2")
    return vid


class LivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)

    def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs):
        n_frames = I_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
        }

        # Disable track if called in chunks to avoid multiple progress bars, or handle externally
        for i in range(n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            x_i_info = self.live_portrait_wrapper.get_kp_info(I_i)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_i_info)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
                'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
                'x_s': x_s.cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

            c_eyes = c_eyes_lst[i].astype(np.float32)
            template_dct['c_eyes_lst'].append(c_eyes)

            c_lip = c_lip_lst[i].astype(np.float32)
            template_dct['c_lip_lst'].append(c_lip)

        return template_dct

    def execute(self, args: ArgumentConfig):
        # for convenience
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device = self.live_portrait_wrapper.device
        crop_cfg = self.cropper.crop_cfg

        ######## load source input ########
        flag_is_source_video = False
        source_fps = 25
        if is_image(args.source):
            flag_is_source_video = False
            img_rgb = load_image_rgb(args.source)
            img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
            log(f"Load source image from {args.source}")
            source_rgb_lst = [img_rgb]
        elif is_video(args.source):
            flag_is_source_video = True
            source_fps = int(get_fps(args.source))
            log(f"Load source video from {args.source}, FPS is {source_fps}")
            # Don't load all frames here to avoid OOM
            source_rgb_lst = []
        else:  # source input is an unknown format
            raise Exception(f"Unknown source format: {args.source}")

        ######## process driving info ########
        flag_load_from_template = is_template(args.driving)
        driving_rgb_crop_256x256_lst = None
        wfp_template = None

        if flag_load_from_template:
            # NOTE: load from template, it is fast, but the cropping video is None
            log(f"Load from template: {args.driving}, NOT the video, so the cropping video and audio are both NULL.", style='bold green')
            driving_template_dct = load(args.driving)
            c_d_eyes_lst = driving_template_dct['c_eyes_lst'] if 'c_eyes_lst' in driving_template_dct.keys() else driving_template_dct['c_d_eyes_lst'] # compatible with previous keys
            c_d_lip_lst = driving_template_dct['c_lip_lst'] if 'c_lip_lst' in driving_template_dct.keys() else driving_template_dct['c_d_lip_lst']
            driving_n_frames = driving_template_dct['n_frames']
            flag_is_driving_video = True if driving_n_frames > 1 else False
            if flag_is_source_video and flag_is_driving_video:
                # We don't know source length yet if we don't load it, but we can estimate or handle it in loop
                n_frames = driving_n_frames # Placeholder, will be adjusted
            elif flag_is_source_video and not flag_is_driving_video:
                n_frames = 999999 # Placeholder
            else:
                n_frames = driving_n_frames

            # set output_fps
            output_fps = driving_template_dct.get('output_fps', inf_cfg.output_fps)
            log(f'The FPS of template: {output_fps}')

            if args.flag_crop_driving_video:
                log("Warning: flag_crop_driving_video is True, but the driving info is a template, so it is ignored.")

        elif osp.exists(args.driving):
            if is_video(args.driving):
                flag_is_driving_video = True
                # load from video file, AND make motion template
                output_fps = int(get_fps(args.driving))
                log(f"Load driving video from: {args.driving}, FPS is {output_fps}")

                log("Start making driving motion template...")
                driving_template_dct = { 'n_frames': 0, 'output_fps': output_fps, 'motion': [], 'c_eyes_lst': [], 'c_lip_lst': [] }

                prev_lmk = None
                for chunk_idx, driving_chunk in enumerate(read_video_chunks(args.driving, chunk_size=args.video_chunk_size, max_dim=1280, division=2)):
                    if inf_cfg.flag_crop_driving_video or (not is_square_video(args.driving)):
                        ret_d = self.cropper.crop_driving_video_chunk(driving_chunk, crop_cfg, prev_lmk=prev_lmk)
                        prev_lmk = ret_d['last_lmk']
                        driving_rgb_crop_lst = ret_d['frame_crop_lst']
                        driving_lmk_crop_lst = ret_d['lmk_crop_lst']
                        driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]
                    else:
                        # If not cropping, we still need landmarks for ratio calculation
                        # We can use crop_driving_video_chunk but ignore the crop result if we want,
                        # or just resize. But we need landmarks.
                        # For simplicity, let's assume we crop or at least detect landmarks.
                        # If is_square_video and not flag_crop, we usually just resize.
                        # But we need landmarks for `calc_ratio`.
                        # So we use the cropper to get landmarks.
                        ret_d = self.cropper.crop_driving_video_chunk(driving_chunk, crop_cfg, prev_lmk=prev_lmk)
                        prev_lmk = ret_d['last_lmk']
                        driving_lmk_crop_lst = ret_d['lmk_crop_lst']
                        driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_chunk]

                    c_d_eyes_lst_chunk, c_d_lip_lst_chunk = self.live_portrait_wrapper.calc_ratio(driving_lmk_crop_lst)
                    I_d_lst = self.live_portrait_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
                    chunk_motion_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst_chunk, c_d_lip_lst_chunk, output_fps=output_fps)

                    driving_template_dct['motion'].extend(chunk_motion_dct['motion'])
                    driving_template_dct['c_eyes_lst'].extend(chunk_motion_dct['c_eyes_lst'])
                    driving_template_dct['c_lip_lst'].extend(chunk_motion_dct['c_lip_lst'])
                    driving_template_dct['n_frames'] += len(driving_chunk)
                    log(f"Processed driving chunk {chunk_idx}, total frames: {driving_template_dct['n_frames']}")

                wfp_template = remove_suffix(args.driving) + '.pkl'
                dump(wfp_template, driving_template_dct)
                log(f"Dump motion template to {wfp_template}")

                driving_n_frames = driving_template_dct['n_frames']
                n_frames = driving_n_frames # Update n_frames

            elif is_image(args.driving):
                flag_is_driving_video = False
                driving_img_rgb = load_image_rgb(args.driving)
                output_fps = 25
                log(f"Load driving image from {args.driving}")
                driving_rgb_lst = [driving_img_rgb]

                # Process single image driving
                driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
                c_d_eyes_lst, c_d_lip_lst = self.live_portrait_wrapper.calc_ratio(driving_lmk_crop_lst)
                I_d_lst = self.live_portrait_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
                driving_template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)
                n_frames = 1
            else:
                raise Exception(f"{args.driving} is not a supported type!")

        else:
            raise Exception(f"{args.driving} does not exist!")

        if not flag_is_driving_video:
            c_d_eyes_lst = c_d_eyes_lst*n_frames
            c_d_lip_lst = c_d_lip_lst*n_frames

        ######## prepare for pasteback ########
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            I_p_pstbk_lst = []
            log("Prepared pasteback mask done.")

        I_p_lst = []
        R_d_0, x_d_0_info = None, None
        flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
        flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite
        lip_delta_before_animation, eye_delta_before_animation = None, None

        ######## process source info ########
        if flag_is_source_video:
            log(f"Start processing source video (Pass 1: Motion Template)...")
            # Pass 1: Calculate source motion template (needed for relative motion and smoothing)
            source_template_dct = {'motion': [], 'c_eyes_lst': [], 'c_lip_lst': []}
            source_lmk_lst_all = [] # Store original landmarks for Pass 2
            source_M_c2o_lst_all = [] # Store M_c2o for Pass 2

            prev_lmk = None
            for chunk_idx, source_chunk in enumerate(read_video_chunks(args.source, chunk_size=args.video_chunk_size, max_dim=inf_cfg.source_max_dim, division=inf_cfg.source_division)):
                if inf_cfg.flag_do_crop:
                    ret_s = self.cropper.crop_source_video_chunk(source_chunk, crop_cfg, prev_lmk=prev_lmk)
                    prev_lmk = ret_s['last_lmk']
                    img_crop_256x256_lst = ret_s['frame_crop_lst']
                    source_lmk_crop_lst = ret_s['lmk_crop_lst']
                    source_M_c2o_lst = ret_s['M_c2o_lst']
                    source_lmk_lst_all.extend(ret_s['lmk_lst'])
                    source_M_c2o_lst_all.extend(ret_s['M_c2o_lst'])
                else:
                    # Fallback if no crop (not recommended for OOM fix but kept for logic)
                    source_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(source_chunk)
                    img_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in source_chunk]
                    source_M_c2o_lst = [None] * len(source_chunk)
                    source_M_c2o_lst_all.extend(source_M_c2o_lst)

                c_s_eyes_lst, c_s_lip_lst = self.live_portrait_wrapper.calc_ratio(source_lmk_crop_lst)
                I_s_lst = self.live_portrait_wrapper.prepare_videos(img_crop_256x256_lst)
                chunk_motion_dct = self.make_motion_template(I_lst=I_s_lst, c_eyes_lst=c_s_eyes_lst, c_lip_lst=c_s_lip_lst, output_fps=source_fps)

                source_template_dct['motion'].extend(chunk_motion_dct['motion'])
                source_template_dct['c_eyes_lst'].extend(chunk_motion_dct['c_eyes_lst'])
                source_template_dct['c_lip_lst'].extend(chunk_motion_dct['c_lip_lst'])
                log(f"Processed source pass 1 chunk {chunk_idx}")

            source_n_frames = len(source_template_dct['motion'])
            n_frames = driving_n_frames if flag_is_driving_video else source_n_frames
            c_s_eyes_lst = source_template_dct['c_eyes_lst'] # For eye retargeting

            key_r = 'R' if 'R' in driving_template_dct['motion'][0].keys() else 'R_d'  # compatible with previous keys
            if inf_cfg.flag_relative_motion:
                if flag_is_driving_video:
                    x_d_exp_lst = [source_template_dct['motion'][i % source_n_frames]['exp'] + driving_template_dct['motion'][i]['exp'] - driving_template_dct['motion'][0]['exp'] for i in range(n_frames)]
                    x_d_exp_lst_smooth = smooth(x_d_exp_lst, source_template_dct['motion'][0]['exp'].shape, device, inf_cfg.driving_smooth_observation_variance)
                else:
                    x_d_exp_lst = [source_template_dct['motion'][i % source_n_frames]['exp'] + (driving_template_dct['motion'][0]['exp'] - inf_cfg.lip_array) for i in range(n_frames)]
                    x_d_exp_lst_smooth = [torch.tensor(x_d_exp[0], dtype=torch.float32, device=device) for x_d_exp in x_d_exp_lst]
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    if flag_is_driving_video:
                        x_d_r_lst = [(np.dot(driving_template_dct['motion'][i][key_r], driving_template_dct['motion'][0][key_r].transpose(0, 2, 1))) @ source_template_dct['motion'][i % source_n_frames]['R'] for i in range(n_frames)]
                        x_d_r_lst_smooth = smooth(x_d_r_lst, source_template_dct['motion'][0]['R'].shape, device, inf_cfg.driving_smooth_observation_variance)
                    else:
                        x_d_r_lst = [source_template_dct['motion'][i % source_n_frames]['R'] for i in range(n_frames)]
                        x_d_r_lst_smooth = [torch.tensor(x_d_r[0], dtype=torch.float32, device=device) for x_d_r in x_d_r_lst]
            else:
                if flag_is_driving_video:
                    x_d_exp_lst = [driving_template_dct['motion'][i]['exp'] for i in range(n_frames)]
                    x_d_exp_lst_smooth = smooth(x_d_exp_lst, source_template_dct['motion'][0]['exp'].shape, device, inf_cfg.driving_smooth_observation_variance)
                else:
                    x_d_exp_lst = [driving_template_dct['motion'][0]['exp']]
                    x_d_exp_lst_smooth = [torch.tensor(x_d_exp[0], dtype=torch.float32, device=device) for x_d_exp in x_d_exp_lst]*n_frames
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    if flag_is_driving_video:
                        x_d_r_lst = [driving_template_dct['motion'][i][key_r] for i in range(n_frames)]
                        x_d_r_lst_smooth = smooth(x_d_r_lst, source_template_dct['motion'][0]['R'].shape, device, inf_cfg.driving_smooth_observation_variance)
                    else:
                        x_d_r_lst = [driving_template_dct['motion'][0][key_r]]
                        x_d_r_lst_smooth = [torch.tensor(x_d_r[0], dtype=torch.float32, device=device) for x_d_r in x_d_r_lst]*n_frames

        else:  # if the input is a source image, process it only once
            if inf_cfg.flag_do_crop:
                crop_info = self.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)
                if crop_info is None:
                    raise Exception("No face detected in the source image!")
                source_lmk = crop_info['lmk_crop']
                img_crop_256x256 = crop_info['img_crop_256x256']
            else:
                source_lmk = self.cropper.calc_lmk_from_cropped_image(source_rgb_lst[0])
                img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))  # force to resize to 256x256
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_c_s = x_s_info['kp']
            R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

            # let lip-open scalar to be 0 at first
            if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
                c_d_lip_before_animation = [0.]
                combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                    lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))

        mkdir(args.output_dir)
        wfp = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}.mp4')
        wfp_concat = None

        ######## animate ########
        if flag_is_driving_video or (flag_is_source_video and not flag_is_driving_video):
            log(f"The animated video consists of {n_frames} frames.")
            # Setup CustomVideoWriterNV targeting 1Mbps, using CQ VBR with spatial/temporal AQ and faststart
            writer = create_custom_video_writer_nv(wfp, codec='h264', fps=output_fps)
        else:
            log(f"The output of image-driven portrait animation is an image.")

        # Loop for animation
        global_i = 0

        # If source is video, we iterate chunks again (Pass 2)
        # If source is image, we iterate range(n_frames)

        def frame_generator():
            if flag_is_source_video:
                while True:
                    for chunk in read_video_chunks(args.source, chunk_size=args.video_chunk_size, max_dim=inf_cfg.source_max_dim, division=inf_cfg.source_division):
                        for frame in chunk:
                            yield frame
            else:
                for _ in range(n_frames):
                    yield source_rgb_lst[0]

        for frame_idx, frame_rgb in enumerate(track(frame_generator(), description='🚀Animating...', total=n_frames)):
            if frame_idx >= n_frames: break
            i = frame_idx

            if flag_is_source_video:  # source video
                source_idx = i % source_n_frames
                x_s_info = source_template_dct['motion'][source_idx]
                x_s_info = dct2device(x_s_info, device)

                # We need to crop again using saved landmarks
                # We have source_lmk_lst_all from Pass 1
                lmk = source_lmk_lst_all[source_idx]

                # Crop
                from .utils.crop import crop_image
                ret_dct = crop_image(frame_rgb, lmk, dsize=crop_cfg.dsize, scale=crop_cfg.scale, vx_ratio=crop_cfg.vx_ratio, vy_ratio=crop_cfg.vy_ratio, flag_do_rot=crop_cfg.flag_do_rot)
                img_crop_256x256 = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)

                source_lmk = ret_dct['pt_crop']
                I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
                f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)

                x_c_s = x_s_info['kp']
                R_s = x_s_info['R']
                x_s =x_s_info['x_s']

                # let lip-open scalar to be 0 at first if the input is a video
                if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
                    c_d_lip_before_animation = [0.]
                    combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                    if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                        lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)
                    else:
                        lip_delta_before_animation = None

                # let eye-open scalar to be the same as the first frame if the latter is eye-open state
                if flag_source_video_eye_retargeting and source_lmk is not None:
                    if i == 0:
                        combined_eye_ratio_tensor_frame_zero = c_s_eyes_lst[0]
                        c_d_eye_before_animation_frame_zero = [[combined_eye_ratio_tensor_frame_zero[0][:2].mean()]]
                        if c_d_eye_before_animation_frame_zero[0][0] < inf_cfg.source_video_eye_retargeting_threshold:
                            c_d_eye_before_animation_frame_zero = [[0.39]]
                    combined_eye_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eye_before_animation_frame_zero, source_lmk)
                    eye_delta_before_animation = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor_before_animation)

                if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:  # prepare for paste back
                    mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, ret_dct['M_c2o'], dsize=(frame_rgb.shape[1], frame_rgb.shape[0]))
            if flag_is_source_video and not flag_is_driving_video:
                x_d_i_info = driving_template_dct['motion'][0]
            else:
                x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, device)
            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys

            if i == 0:  # cache the first frame
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info.copy()

            delta_new = x_s_info['exp'].clone()
            if inf_cfg.flag_relative_motion:
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    R_new = x_d_r_lst_smooth[i] if flag_is_source_video else (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                else:
                    R_new = R_s
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                    if flag_is_source_video:
                        for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                            delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :]
                        delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1]
                        delta_new[:, 5, 2] = x_d_exp_lst_smooth[i][5, 2]
                        delta_new[:, 8, 2] = x_d_exp_lst_smooth[i][8, 2]
                        delta_new[:, 9, 1:] = x_d_exp_lst_smooth[i][9, 1:]
                    else:
                        if flag_is_driving_video:
                            delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                        else:
                            delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device))
                elif inf_cfg.animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        if flag_is_source_video:
                            delta_new[:, lip_idx, :] = x_d_exp_lst_smooth[i][lip_idx, :]
                        elif flag_is_driving_video:
                            delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, lip_idx, :]
                        else:
                            delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device)))[:, lip_idx, :]
                elif inf_cfg.animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        if flag_is_source_video:
                            delta_new[:, eyes_idx, :] = x_d_exp_lst_smooth[i][eyes_idx, :]
                        elif flag_is_driving_video:
                            delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, eyes_idx, :]
                        else:
                            delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
                if inf_cfg.animation_region == "all":
                    scale_new = x_s_info['scale'] if flag_is_source_video else x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                else:
                    scale_new = x_s_info['scale']
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    t_new = x_s_info['t'] if flag_is_source_video else x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                else:
                    t_new = x_s_info['t']
            else:
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    R_new = x_d_r_lst_smooth[i] if flag_is_source_video else R_d_i
                else:
                    R_new = R_s
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                    for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                        delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :] if flag_is_source_video else x_d_i_info['exp'][:, idx, :]
                    delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1] if flag_is_source_video else x_d_i_info['exp'][:, 3:5, 1]
                    delta_new[:, 5, 2] = x_d_exp_lst_smooth[i][5, 2] if flag_is_source_video else x_d_i_info['exp'][:, 5, 2]
                    delta_new[:, 8, 2] = x_d_exp_lst_smooth[i][8, 2] if flag_is_source_video else x_d_i_info['exp'][:, 8, 2]
                    delta_new[:, 9, 1:] = x_d_exp_lst_smooth[i][9, 1:] if flag_is_source_video else x_d_i_info['exp'][:, 9, 1:]
                elif inf_cfg.animation_region == "lip":
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_exp_lst_smooth[i][lip_idx, :] if flag_is_source_video else x_d_i_info['exp'][:, lip_idx, :]
                elif inf_cfg.animation_region == "eyes":
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_exp_lst_smooth[i][eyes_idx, :] if flag_is_source_video else x_d_i_info['exp'][:, eyes_idx, :]
                scale_new = x_s_info['scale']
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    t_new = x_d_i_info['t']
                else:
                    t_new = x_s_info['t']

            t_new[..., 2].fill_(0)  # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            if inf_cfg.flag_relative_motion and inf_cfg.driving_option == "expression-friendly" and not flag_is_source_video and flag_is_driving_video:
                if i == 0:
                    x_d_0_new = x_d_i_new
                    motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
                    # motion_multiplier *= inf_cfg.driving_multiplier
                x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
                x_d_i_new = x_d_diff + x_s

            # Algorithm 1:
            if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # without stitching or retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new += lip_delta_before_animation
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
                else:
                    pass
            elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
                # with stitching and without retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation
                else:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
            else:
                eyes_delta, lip_delta = None, None
                if inf_cfg.flag_eye_retargeting and source_lmk is not None:
                    c_d_eyes_i = c_d_eyes_lst[i]
                    combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
                if inf_cfg.flag_lip_retargeting and source_lmk is not None:
                    c_d_lip_i = c_d_lip_lst[i]
                    combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

                if inf_cfg.flag_relative_motion:  # use x_s
                    x_d_i_new = x_s + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                        (eyes_delta if eyes_delta is not None else 0) + \
                        (lip_delta if lip_delta is not None else 0)

                if inf_cfg.flag_stitching:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU
                if flag_is_source_video:
                    I_p_pstbk = paste_back(I_p_i, ret_dct['M_c2o'], frame_rgb, mask_ori_float)
                else:
                    I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source_rgb_lst[0], mask_ori_float)

                if flag_is_driving_video or flag_is_source_video:
                    writer.write(cv2.cvtColor(I_p_pstbk, cv2.COLOR_RGB2BGR))
                else:
                    I_p_pstbk_lst = [I_p_pstbk] # For image output
            else:
                if flag_is_driving_video or flag_is_source_video:
                    writer.write(cv2.cvtColor(I_p_i, cv2.COLOR_RGB2BGR))

        if flag_is_driving_video or (flag_is_source_video and not flag_is_driving_video):
            writer.release()
            flag_source_has_audio = flag_is_source_video and has_audio_stream(args.source)
            flag_driving_has_audio = (not flag_load_from_template) and has_audio_stream(args.driving)

            # NOTE: update output fps
            output_fps = source_fps if flag_is_source_video else output_fps

            ######### build the final result #########
            if flag_source_has_audio or flag_driving_has_audio:
                wfp_with_audio = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_with_audio.mp4')
                audio_from_which_video = args.driving if ((flag_driving_has_audio and args.audio_priority == 'driving') or (not flag_source_has_audio)) else args.source
                log(f"Audio is selected from {audio_from_which_video}")
                add_audio_to_video(wfp, audio_from_which_video, wfp_with_audio)
                os.replace(wfp_with_audio, wfp)
                log(f"Replace {wfp_with_audio} with {wfp}")

            # final log
            if wfp_template not in (None, ''):
                log(f'Animated template: {wfp_template}, you can specify `-d` argument with this template path next time to avoid cropping video, motion making and protecting privacy.', style='bold green')
            log(f'Animated video: {wfp}')
        else:
            wfp = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}.jpg')
            if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
                cv2.imwrite(wfp, I_p_pstbk_lst[0][..., ::-1])
            else:
                cv2.imwrite(wfp, I_p_lst[0][..., ::-1])
            # final log
            log(f'Animated image: {wfp}')

        return wfp, wfp_concat
