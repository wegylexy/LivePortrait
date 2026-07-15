FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime AS base
# Install ffmpeg and build tools for compiling C-extensions (like stringzilla)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM base AS build
# Install dependencies
COPY requirements_base.txt requirements.txt ./
RUN pip install -r requirements.txt
COPY src src
COPY inference.py .
# Download pretrained weights
ADD https://huggingface.co/KlingTeam/LivePortrait/resolve/main/insightface/models/buffalo_l/2d106det.onnx pretrained_weights/insightface/models/buffalo_l/2d106det.onnx
ADD https://huggingface.co/KlingTeam/LivePortrait/resolve/main/insightface/models/buffalo_l/det_10g.onnx pretrained_weights/insightface/models/buffalo_l/det_10g.onnx
ADD https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/landmark.onnx pretrained_weights/liveportrait/landmark.onnx
ADD https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/appearance_feature_extractor.pth pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth
ADD https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/motion_extractor.pth pretrained_weights/liveportrait/base_models/motion_extractor.pth
ADD https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/spade_generator.pth pretrained_weights/liveportrait/base_models/spade_generator.pth
ADD https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/warping_module.pth pretrained_weights/liveportrait/base_models/warping_module.pth
ADD https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/retargeting_models/stitching_retargeting_module.pth pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth

FROM build AS publish
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENTRYPOINT [ "python", "inference.py" ]
