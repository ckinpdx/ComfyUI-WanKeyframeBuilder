import torch


class WanKeyframeBuilder:
    """
    Keyframe Timeline Builder for Wan Video Workflows
    
    Places keyframe images at specific positions along a timeline with controllable
    influence strength. Designed to work with WanVideoImageToVideoEncode node.
    
    - First and last keyframes get independent strengths
    - Middle keyframes get adjustable influence (default 0.8) for smoother motion
    - Non-keyframe positions are gray (0.5) for the model to generate freely
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "num_frames": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 4096,
                        "step": 4,
                        "tooltip": "Total frames in the output timeline. Should be 4n+1 (e.g., 81, 121, 161) for Wan models.",
                    },
                ),
                "spacing_mode": (
                    ["even", "manual"],
                    {
                        "tooltip": "Even: auto-distribute keyframes. Manual: use frame position inputs.",
                    },
                ),
                "first_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Mask strength for the FIRST keyframe. 1.0 = fully locked.",
                    },
                ),
                "last_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Mask strength for the LAST keyframe. 1.0 = fully locked.",
                    },
                ),
                "middle_strength": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Mask strength for middle keyframes. 0.8 gives the model some freedom to interpret the pose.",
                    },
                ),
                "frame_1": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 1 (used in manual mode)",
                    },
                ),
                "frame_2": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 2 (used in manual mode)",
                    },
                ),
                "frame_3": (
                    "INT",
                    {
                        "default": 40,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 3 (used in manual mode)",
                    },
                ),
                "frame_4": (
                    "INT",
                    {
                        "default": 60,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 4 (used in manual mode)",
                    },
                ),
                "frame_5": (
                    "INT",
                    {
                        "default": 80,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 5 (used in manual mode)",
                    },
                ),
                "frame_6": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 6 (used in manual mode)",
                    },
                ),
                "frame_7": (
                    "INT",
                    {
                        "default": 120,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 7 (used in manual mode)",
                    },
                ),
                "frame_8": (
                    "INT",
                    {
                        "default": 140,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 8 (used in manual mode)",
                    },
                ),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("images", "masks", "keyframes")
    FUNCTION = "build_sequence"
    CATEGORY = "WanKeyframeBuilder"

    def _prepare_single_frame(self, img):
        """Ensure image shape = [1, H, W, C]."""
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.shape[0] > 1:
            img = img[:1]
        return img

    def build_sequence(
        self,
        image1,
        num_frames,
        spacing_mode,
        first_strength,
        last_strength,
        middle_strength,
        frame_1,
        frame_2,
        frame_3,
        frame_4,
        frame_5,
        frame_6,
        frame_7,
        frame_8,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
    ):
        # Collect all possible images
        all_imgs = [
            image1,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
        ]

        # Keep only connected (non-None) images
        available_imgs = [img for img in all_imgs if img is not None]

        # Number of keyframes = number of connected images
        N = len(available_imgs)

        # Normalize all used images to [1, H, W, C]
        used_imgs = [self._prepare_single_frame(img) for img in available_imgs]

        # Validate resolution - ComfyUI IMAGE is [B, H, W, C]
        h, w = used_imgs[0].shape[1], used_imgs[0].shape[2]
        C = used_imgs[0].shape[3]
        for idx, im in enumerate(used_imgs[1:], start=2):
            if im.shape[1] != h or im.shape[2] != w:
                raise ValueError(
                    f"All keyframe images must match resolution. First is {w}x{h}, "
                    f"but image {idx} is {im.shape[2]}x{im.shape[1]}."
                )

        T = max(int(num_frames), 1)

        device = used_imgs[0].device
        dtype = used_imgs[0].dtype

        # Fixed mid-gray background (0.5)
        gray = torch.full((1, h, w, C), 0.5, device=device, dtype=dtype)

        # Allocate outputs
        images_out = gray.repeat(T, 1, 1, 1)  # [T, H, W, C]
        masks_out = torch.zeros((T, h, w), device=device, dtype=dtype)  # [T, H, W]

        # Manual frame indices (we only use first N)
        manual_frames = [
            frame_1,
            frame_2,
            frame_3,
            frame_4,
            frame_5,
            frame_6,
            frame_7,
            frame_8,
        ][:N]

        # Determine keyframe anchor indices
        if spacing_mode == "manual":
            key_idx = [max(0, min(T - 1, int(f))) for f in manual_frames]
            key_idx = sorted(key_idx)
        else:
            # Even spacing across full timeline
            if N == 1:
                key_idx = [0]  # Single image goes at start
            else:
                key_idx = [round(i * (T - 1) / (N - 1)) for i in range(N)]

        # Apply keyframes
        for anchor, img_tensor in zip(key_idx, used_imgs):
            img = img_tensor[0]  # [H, W, C]

            # Place keyframe image
            images_out[anchor] = img

            # Set mask strength based on position
            if anchor == key_idx[0] and anchor == key_idx[-1]:
                # Only one keyframe
                masks_out[anchor] = first_strength
            elif anchor == key_idx[0]:
                masks_out[anchor] = first_strength
            elif anchor == key_idx[-1]:
                masks_out[anchor] = last_strength
            else:
                masks_out[anchor] = middle_strength

        # Build keyframes-only batch [N, H, W, C]
        keyframes_batch = torch.cat(used_imgs, dim=0)

        return images_out, masks_out, keyframes_batch


class WanKeyframeBuilderContinuation:
    """
    Keyframe Timeline Builder with Continuation Support + SVI Outputs
    
    Designed for multi-pass video generation where HuMo handles quality via
    reference images. Continuation frames provide motion context without a
    first keyframe on the timeline.
    
    - Continuation frames (from previous generation) fill positions 0 to N-1
    - image1 is placed on the timeline (with adjustable strength)
    - Keyframes (image2+) are distributed across remaining timeline
    - Strength can be uniform or decay from start to end
    
    SVI-Specific Outputs:
    - svi_reference_only: Timeline filled entirely with image1 (for SVI-Shot reference padding),
      overridden at continuation/keyframe positions.
    - svi_keyframe_segments: Timeline where each segment is filled with its corresponding keyframe
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": (
                    "IMAGE",
                    {
                        "tooltip": "First reference image. Placed on the timeline and used for SVI/reference.",
                    },
                ),
                "num_frames": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 4096,
                        "step": 4,
                        "tooltip": "Total frames in the output timeline. Should be 4n+1 (e.g., 81, 121, 161) for Wan models.",
                    },
                ),
                "spacing_mode": (
                    ["even", "manual"],
                    {
                        "tooltip": "Even: auto-distribute keyframes. Manual: use frame position inputs.",
                    },
                ),
                "last_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Mask strength for the LAST keyframe. 1.0 = fully locked.",
                    },
                ),
                "middle_strength": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Mask strength for middle keyframes.",
                    },
                ),
                "continuation_frames": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Number of frames to use from continuation input (taken from END of previous video).",
                    },
                ),
                "continuation_strength_mode": (
                    ["uniform", "decay"],
                    {
                        "tooltip": "Uniform: all continuation frames use same strength. Decay: exponential decay from start to end.",
                    },
                ),
                "continuation_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Strength for continuation frames (uniform mode).",
                    },
                ),
                "continuation_strength_start": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Starting strength for first continuation frame (decay mode).",
                    },
                ),
                "continuation_strength_end": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Ending strength for last continuation frame (decay mode).",
                    },
                ),
                "continuation_decay_rate": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Exponential decay rate. Higher = sharper drop. 3.0 = moderate, 5.0+ = aggressive.",
                    },
                ),
                "place_first_keyframe": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Place image1 on the timeline (after continuation if present, otherwise at frame 0).",
                    },
                ),
                "first_keyframe_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Mask strength for image1 wherever it is placed.",
                    },
                ),
                "frame_2": (
                    "INT",
                    {
                        "default": 40,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 2 (used in manual mode)",
                    },
                ),
                "frame_3": (
                    "INT",
                    {
                        "default": 80,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 3 (used in manual mode)",
                    },
                ),
                "frame_4": (
                    "INT",
                    {
                        "default": 60,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 4 (used in manual mode)",
                    },
                ),
                "frame_5": (
                    "INT",
                    {
                        "default": 80,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 5 (used in manual mode)",
                    },
                ),
                "frame_6": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 6 (used in manual mode)",
                    },
                ),
                "frame_7": (
                    "INT",
                    {
                        "default": 120,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 7 (used in manual mode)",
                    },
                ),
                "frame_8": (
                    "INT",
                    {
                        "default": 140,
                        "min": 0,
                        "max": 4095,
                        "step": 1,
                        "tooltip": "Position for keyframe 8 (used in manual mode)",
                    },
                ),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "continuation_image": (
                    "IMAGE",
                    {
                        "tooltip": "Frames from previous generation. Last N frames will be used for motion context.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "masks", "keyframes", "svi_reference_only", "svi_keyframe_segments")
    FUNCTION = "build_sequence"
    CATEGORY = "WanKeyframeBuilder"

    def _prepare_single_frame(self, img):
        """Ensure image shape = [1, H, W, C]."""
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.shape[0] > 1:
            img = img[:1]
        return img

    def build_sequence(
        self,
        image1,
        num_frames,
        spacing_mode,
        last_strength,
        middle_strength,
        continuation_frames,
        continuation_strength_mode,
        continuation_strength,
        continuation_strength_start,
        continuation_strength_end,
        continuation_decay_rate,
        place_first_keyframe,
        first_keyframe_strength,
        frame_2,
        frame_3,
        frame_4,
        frame_5,
        frame_6,
        frame_7,
        frame_8,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        continuation_image=None,
    ):
        # image1 is required and is placed on the timeline (with adjustable strength)
        # Collect timeline keyframes (image2+)
        timeline_imgs = [
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
        ]

        # All images for keyframes output (image1 + any connected image2-8)
        all_keyframe_imgs = [image1] + [img for img in timeline_imgs if img is not None]

        # Timeline images only (image2+ that are connected)
        available_timeline_imgs = [img for img in timeline_imgs if img is not None]

        # Number of keyframes on timeline (excluding image1, which is special)
        N = len(available_timeline_imgs)

        # Normalize image1 for reference
        img1_prepared = self._prepare_single_frame(image1)
        h, w = img1_prepared.shape[1], img1_prepared.shape[2]
        C = img1_prepared.shape[3]

        # Normalize timeline images
        used_timeline_imgs = [self._prepare_single_frame(img) for img in available_timeline_imgs]

        # Validate resolution against image1
        for idx, im in enumerate(used_timeline_imgs, start=2):
            if im.shape[1] != h or im.shape[2] != w:
                raise ValueError(
                    f"All keyframe images must match resolution. image1 is {w}x{h}, "
                    f"but image{idx} is {im.shape[2]}x{im.shape[1]}."
                )

        T = max(int(num_frames), 1)
        device = img1_prepared.device
        dtype = img1_prepared.dtype

        # Fixed mid-gray background (0.5) for standard output
        gray = torch.full((1, h, w, C), 0.5, device=device, dtype=dtype)

        # Allocate standard outputs
        images_out = gray.repeat(T, 1, 1, 1)  # [T, H, W, C]
        masks_out = torch.zeros((T, h, w), device=device, dtype=dtype)  # [T, H, W]

        # === SVI OUTPUT 1: Reference Only ===
        # Timeline filled entirely with image1 (for SVI-Shot reference padding)
        svi_reference_only = img1_prepared[0].unsqueeze(0).repeat(T, 1, 1, 1)

        # Process continuation frames
        cont_frames_list = []
        if continuation_image is not None:
            cont_img = continuation_image
            if cont_img.ndim == 3:
                cont_img = cont_img.unsqueeze(0)

            # Take last N frames from previous video (in chronological order)
            num_cont = min(continuation_frames, cont_img.shape[0])
            cont_img = cont_img[-num_cont:]  # Take last N frames

            # DO NOT reverse - keep chronological order
            # First frame in cont_img is oldest, last frame is newest

            # Resize each frame if needed
            for i in range(cont_img.shape[0]):
                frame = cont_img[i : i + 1]  # Keep as [1, H, W, C]
                if frame.shape[1] != h or frame.shape[2] != w:
                    frame = frame.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                    frame = torch.nn.functional.interpolate(
                        frame,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    frame = frame.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
                cont_frames_list.append(frame[0])  # Store as [H, W, C]

        # Apply continuation frames at positions 0 to N-1
        num_cont_applied = len(cont_frames_list)
        for i, cont_frame in enumerate(cont_frames_list):
            if i < T:  # Safety check
                images_out[i] = cont_frame
                svi_reference_only[i] = cont_frame  # Also apply to SVI reference output

                # Calculate strength based on mode
                if continuation_strength_mode == "uniform":
                    strength = continuation_strength
                else:  # decay - from start strength to end strength
                    if num_cont_applied > 1:
                        t = i / (num_cont_applied - 1)  # 0 to 1 (0=first/oldest, 1=last/newest)
                        import math
                        # Exponential decay from start to end
                        decay = math.exp(-continuation_decay_rate * t)
                        strength = continuation_strength_end + (continuation_strength_start - continuation_strength_end) * decay
                    else:
                        strength = continuation_strength_start

                masks_out[i] = strength

        # === Place image1 on the timeline with adjustable strength ===
        # - If continuation frames exist: directly after them
        # - If none: at frame 0
        if place_first_keyframe:
            if num_cont_applied > 0:
                first_keyframe_position = num_cont_applied
            else:
                first_keyframe_position = 0

            if first_keyframe_position < T:
                img1_frame = img1_prepared[0]  # [H, W, C]
                images_out[first_keyframe_position] = img1_frame
                svi_reference_only[first_keyframe_position] = img1_frame
                masks_out[first_keyframe_position] = first_keyframe_strength

        # Manual frame indices for timeline keyframes (image2+)
        manual_frames = [
            frame_2,
            frame_3,
            frame_4,
            frame_5,
            frame_6,
            frame_7,
            frame_8,
        ][:N]

        # Determine keyframe anchor indices for remaining keyframes (image2+)
        key_idx = []
        if N > 0:
            if spacing_mode == "manual":
                key_idx = [max(0, min(T - 1, int(f))) for f in manual_frames]
                key_idx = sorted(key_idx)
            else:
                # Even spacing as if image1 was at a phantom position 0
                total_for_spacing = N + 1
                all_positions = [round(i * (T - 1) / (total_for_spacing - 1)) for i in range(total_for_spacing)]
                key_idx = all_positions[1:]  # Skip position 0 (reserved conceptually for image1)

            # Apply timeline keyframes (image2+)
            for idx in range(len(key_idx)):
                anchor = key_idx[idx]
                img_tensor = used_timeline_imgs[idx]
                img = img_tensor[0]  # [H, W, C]
                images_out[anchor] = img

                # Set mask strength
                if anchor == key_idx[-1]:
                    masks_out[anchor] = last_strength
                else:
                    masks_out[anchor] = middle_strength

        # === SVI OUTPUT 2: Keyframe Segments ===
        # Each keyframe fills its segment equally
        svi_keyframe_segments = img1_prepared[0].unsqueeze(0).repeat(T, 1, 1, 1)

        # Apply continuation frames first (same as standard output)
        for i, cont_frame in enumerate(cont_frames_list):
            if i < T:
                svi_keyframe_segments[i] = cont_frame

        # Divide remaining timeline into equal segments for each keyframe (image2+)
        if N > 0:
            cont_end = num_cont_applied  # Where continuation frames end
            remaining_frames = T - cont_end
            
            if remaining_frames > 0:
                # Calculate segment boundaries
                segment_size = remaining_frames / N
                
                for idx, img_tensor in enumerate(used_timeline_imgs):
                    img = img_tensor[0]  # [H, W, C]
                    
                    # Calculate this keyframe's segment range
                    start_pos = cont_end + int(idx * segment_size)
                    end_pos = cont_end + int((idx + 1) * segment_size)
                    
                    # Last segment extends to end
                    if idx == N - 1:
                        end_pos = T
                    
                    # Fill this segment with the keyframe
                    for pos in range(start_pos, end_pos):
                        if 0 <= pos < T:
                            svi_keyframe_segments[pos] = img

        # Build keyframes-only batch [N+1, H, W, C] (image1 + timeline images)
        keyframes_list = [img1_prepared] + used_timeline_imgs
        keyframes_batch = torch.cat(keyframes_list, dim=0)

        return images_out, masks_out, keyframes_batch, svi_reference_only, svi_keyframe_segments


NODE_CLASS_MAPPINGS = {
    "WanKeyframeBuilder": WanKeyframeBuilder,
    "WanKeyframeBuilderContinuation": WanKeyframeBuilderContinuation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanKeyframeBuilder": "Wan Keyframe Builder",
    "WanKeyframeBuilderContinuation": "Wan Keyframe Builder (Continuation)",
}
