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
    Keyframe Timeline Builder with Continuation Support
    
    Designed for multi-pass video generation where HuMo handles quality via
    reference images. Continuation frames provide motion context without a
    first keyframe on the timeline.
    
    - Continuation frames (from previous generation) fill positions 0 to N-1
    - Keyframes (image2+) are distributed across remaining timeline
    - image1 is included in keyframes output for HuMo but NOT placed on timeline
    - Strength can be uniform or decay from start to end
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": (
                    "IMAGE",
                    {
                        "tooltip": "First reference image for HuMo. Included in keyframes output but NOT placed on timeline.",
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
                        "tooltip": "Number of frames to use from continuation input.",
                    },
                ),
                "continuation_strength_mode": (
                    ["uniform", "decay"],
                    {
                        "tooltip": "Uniform: all continuation frames use same strength. Decay: exponential decay from ceiling to floor.",
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
                "continuation_strength_ceiling": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Starting strength for newest continuation frame (decay mode).",
                    },
                ),
                "continuation_strength_floor": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Minimum strength for oldest continuation frames (decay mode).",
                    },
                ),
                "continuation_decay_rate": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Exponential decay rate. Higher = sharper drop to floor. 3.0 = moderate, 5.0+ = aggressive.",
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
        last_strength,
        middle_strength,
        continuation_frames,
        continuation_strength_mode,
        continuation_strength,
        continuation_strength_ceiling,
        continuation_strength_floor,
        continuation_decay_rate,
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
        # image1 is required but only for keyframes output, not timeline
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

        # Number of keyframes on timeline
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

        # Fixed mid-gray background (0.5)
        gray = torch.full((1, h, w, C), 0.5, device=device, dtype=dtype)

        # Allocate outputs
        images_out = gray.repeat(T, 1, 1, 1)  # [T, H, W, C]
        masks_out = torch.zeros((T, h, w), device=device, dtype=dtype)  # [T, H, W]

        # Process continuation frames
        cont_frames_list = []
        if continuation_image is not None:
            cont_img = continuation_image
            if cont_img.ndim == 3:
                cont_img = cont_img.unsqueeze(0)

            # Take last N frames (or fewer if less provided)
            num_cont = min(continuation_frames, cont_img.shape[0])
            cont_img = cont_img[-num_cont:]  # Take last N frames

            # Reverse order: newest frame first (position 0), oldest last
            cont_img = cont_img.flip(0)

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
        # Frame 0 = newest (highest strength), Frame N-1 = oldest (lowest strength)
        num_cont_applied = len(cont_frames_list)
        for i, cont_frame in enumerate(cont_frames_list):
            if i < T:  # Safety check
                images_out[i] = cont_frame

                # Calculate strength based on mode
                if continuation_strength_mode == "uniform":
                    strength = continuation_strength
                else:  # decay - exponential from ceiling to floor
                    if num_cont_applied > 1:
                        t = i / (num_cont_applied - 1)  # 0 to 1
                        # Exponential decay: ceiling at t=0, approaches floor as t->1
                        import math
                        decay = math.exp(-continuation_decay_rate * t)
                        strength = continuation_strength_floor + (continuation_strength_ceiling - continuation_strength_floor) * decay
                    else:
                        strength = continuation_strength_ceiling

                masks_out[i] = strength

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

        # Determine keyframe anchor indices
        # Calculate positions as if image1 was at position 0, but don't place it
        # image2+ keep their original spacing
        
        if N > 0:
            if spacing_mode == "manual":
                key_idx = [max(0, min(T - 1, int(f))) for f in manual_frames]
                key_idx = sorted(key_idx)
            else:
                # Even spacing as if image1 was at 0
                # Total keyframes for spacing calculation = N + 1 (including phantom image1)
                total_for_spacing = N + 1
                # Generate positions for all keyframes including phantom image1
                all_positions = [round(i * (T - 1) / (total_for_spacing - 1)) for i in range(total_for_spacing)]
                # Skip position 0 (image1's spot), take the rest
                key_idx = all_positions[1:]

            # Apply timeline keyframes
            for anchor, img_tensor in zip(key_idx, used_timeline_imgs):
                img = img_tensor[0]  # [H, W, C]

                # Place keyframe image
                images_out[anchor] = img

                # Set mask strength based on position
                if anchor == key_idx[-1]:
                    masks_out[anchor] = last_strength
                else:
                    masks_out[anchor] = middle_strength

        # Build keyframes-only batch [N, H, W, C] (image1 + all connected timeline images)
        keyframes_list = [img1_prepared] + used_timeline_imgs
        keyframes_batch = torch.cat(keyframes_list, dim=0)

        return images_out, masks_out, keyframes_batch


NODE_CLASS_MAPPINGS = {
    "WanKeyframeBuilder": WanKeyframeBuilder,
    "WanKeyframeBuilderContinuation": WanKeyframeBuilderContinuation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanKeyframeBuilder": "Wan Keyframe Builder",
    "WanKeyframeBuilderContinuation": "Wan Keyframe Builder (Continuation)",
}
