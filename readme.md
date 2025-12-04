# WanKeyframeBuilder

A ComfyUI custom node for building keyframe timelines for Wan video generation workflows.

## Overview

WanKeyframeBuilder places keyframe images at specific positions along a timeline with controllable influence strength. It's designed to work with the `WanVideoImageToVideoEncode` node from [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper).

The node produces smooth keyframe-to-keyframe motion by:
- Locking first and last keyframes (default strength 1.0)
- Giving middle keyframes some flexibility (default strength 0.8)
- Leaving non-keyframe positions as gray (0.5) for the model to generate freely

## Installation

Clone or copy this folder into your ComfyUI custom nodes directory:

```
ComfyUI/custom_nodes/WanKeyframeBuilder/
```

Restart ComfyUI.

## Nodes

### Wan Keyframe Builder

Basic keyframe timeline builder.

| Input | Type | Description |
|-------|------|-------------|
| image1 | IMAGE | First keyframe (required) |
| image2-8 | IMAGE | Additional keyframes (optional) |
| num_frames | INT | Total frames in output timeline. Use 4n+1 values (81, 121, 161) for Wan models |
| spacing_mode | ENUM | `even`: auto-distribute keyframes. `manual`: use frame position inputs |
| first_last_strength | FLOAT | Mask strength for first and last keyframes (default 1.0) |
| middle_strength | FLOAT | Mask strength for middle keyframes (default 0.8) |
| frame_1-8 | INT | Manual positions for each keyframe |

### Wan Keyframe Builder (Continuation)

Extended version with video continuation support for smoother multi-segment video generation.

| Input | Type | Description |
|-------|------|-------------|
| *(all inputs from base node)* | | |
| continuation_image | IMAGE | Last frames from previous video generation (optional). Uses up to 3 frames - if more provided, takes the last 3. |
| continuation_strength | FLOAT | Mask strength for continuation frames (default 0.6) |
| continuation_gap | INT | Frames between continuation context and first keyframe (default 4, minimum 3) |

The continuation node places up to 3 frames from your previous generation at positions 0, 1, 2, giving the model motion context for smoother transitions between video segments. The first keyframe is offset by `continuation_gap` frames.

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| images | IMAGE | Full timeline with keyframes placed, gray elsewhere |
| masks | MASK | Strength mask for each frame |
| keyframes | IMAGE | Batch of just the keyframe images (does not include continuation frames) |

## Usage

### Basic Keyframe Workflow

1. Connect your keyframe images (minimum 1, up to 8)
2. Set `num_frames` to match your video length
3. Choose `even` spacing or set manual frame positions
4. Connect `images` to `start_image` on WanVideoImageToVideoEncode
5. Connect `masks` to `temporal_mask` on WanVideoImageToVideoEncode

### Multi-Segment Video Workflow

1. Generate your first video segment using the base node
2. For subsequent segments, use the Continuation node
3. Connect the last frames from your previous video to `continuation_image`
4. The continuation frames provide motion context to reduce jarring transitions between segments

## Strength Values

- **1.0**: Fully locked - model must match this frame exactly
- **0.8**: Strong influence with some flexibility - good for middle keyframes
- **0.6**: Moderate influence - good for continuation context
- **0.0**: No influence - model generates freely

The default 0.8 for middle keyframes allows the model to interpret the pose rather than pixel-match it, resulting in smoother motion transitions.

## License

MIT
