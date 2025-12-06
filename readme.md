[readme.md](https://github.com/user-attachments/files/23973093/readme.md)
# WanKeyframeBuilder

A ComfyUI custom node for building keyframe timelines for Wan video generation workflows.

## Overview

WanKeyframeBuilder places keyframe images at specific positions along a timeline with controllable influence strength. It's designed to work with the `WanVideoImageToVideoEncode` node from ComfyUI-WanVideoWrapper.

## Installation

Clone or copy this folder into your ComfyUI custom nodes directory:
```
ComfyUI/custom_nodes/WanKeyframeBuilder/
```

Restart ComfyUI.

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| image1 | IMAGE | First keyframe (required) |
| image2-8 | IMAGE | Additional keyframes (optional) |
| num_frames | INT | Total frames in output timeline. Use 4n+1 values (81, 121, 161) for Wan models |
| spacing_mode | ENUM | `even`: auto-distribute keyframes. `manual`: use frame position inputs |
| first_last_strength | FLOAT | Mask strength for first and last keyframes (default 1.0) |
| middle_strength | FLOAT | Mask strength for middle keyframes (default 0.8) |
| frame_1-8 | INT | Manual positions for each keyframe |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| images | IMAGE | Full timeline with keyframes placed, gray elsewhere |
| masks | MASK | Strength mask for each frame |
| keyframes | IMAGE | Batch of just the keyframe images |

## Usage

1. Connect your keyframe images (minimum 1, up to 8)
2. Set `num_frames` to match your video length
3. Choose `even` spacing or set manual frame positions
4. Connect `images` to `start_image` on WanVideoImageToVideoEncode
5. Connect `masks` to `temporal_mask` on WanVideoImageToVideoEncode

## Strength Values

- **1.0**: Fully locked - model must match this frame exactly
- **0.8**: Strong influence with some flexibility - good for middle keyframes
- **0.0**: No influence - model generates freely

## License

MIT
