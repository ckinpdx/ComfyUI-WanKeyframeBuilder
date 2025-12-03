import torch

class WanKeyframeBuilder:
    """
    Keyframe Timeline Builder for Wan Video Workflows
    
    Places keyframe images at specific positions along a timeline with controllable
    influence strength. Designed to work with WanVideoImageToVideoEncode node.
    
    - First and last keyframes get full lock (default 1.0)
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
                "first_last_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Mask strength for first and last keyframes. 1.0 = fully locked.",
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
                "frame_1": ("INT", {"default": 0, "min": 0, "max": 4095, "step": 1, "tooltip": "Position for keyframe 1 (used in manual mode)"}),
                "frame_2": ("INT", {"default": 20, "min": 0, "max": 4095, "step": 1, "tooltip": "Position for keyframe 2 (used in manual mode)"}),
                "frame_3": ("INT", {"default": 40, "min": 0, "max": 4095, "step": 1, "tooltip": "Position for keyframe 3 (used in manual mode)"}),
                "frame_4": ("INT", {"default": 60, "min": 0, "max": 4095, "step": 1, "tooltip": "Position for keyframe 4 (used in manual mode)"}),
                "frame_5": ("INT", {"default": 80, "min": 0, "max": 4095, "step": 1, "tooltip": "Position for keyframe 5 (used in manual mode)"}),
                "frame_6": ("INT", {"default": 100, "min": 0, "max": 4095, "step": 1, "tooltip": "Position for keyframe 6 (used in manual mode)"}),
                "frame_7": ("INT", {"default": 120, "min": 0, "max": 4095, "step": 1, "tooltip": "Position for keyframe 7 (used in manual mode)"}),
                "frame_8": ("INT", {"default": 140, "min": 0, "max": 4095, "step": 1, "tooltip": "Position for keyframe 8 (used in manual mode)"}),
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
    CATEGORY = "WanVideoWrapper"

    def _prepare_single_frame(self, img):
        """Ensure image shape = [1, C, H, W]."""
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
        first_last_strength,
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

        # Normalize all used images to [1, C, H, W]
        used_imgs = [self._prepare_single_frame(img) for img in available_imgs]

        # Validate resolution
        h, w = used_imgs[0].shape[2], used_imgs[0].shape[3]
        for idx, im in enumerate(used_imgs[1:], start=2):
            if im.shape[2] != h or im.shape[3] != w:
                raise ValueError(
                    f"All keyframe images must match resolution. First is {h}x{w}, "
                    f"but image {idx} is {im.shape[2]}x{im.shape[3]}."
                )

        T = max(int(num_frames), 1)

        device = used_imgs[0].device
        dtype = used_imgs[0].dtype
        C = used_imgs[0].shape[1]

        # Fixed mid-gray background (0.5)
        gray = torch.full((1, C, h, w), 0.5, device=device, dtype=dtype)

        # Allocate outputs
        images_out = gray.repeat(T, 1, 1, 1)  # [T, C, H, W]
        masks_out = torch.zeros((T, h, w), device=device, dtype=dtype)  # [T, H, W]

        # Manual frame indices (we only use first N)
        manual_frames = [
            frame_1, frame_2, frame_3, frame_4,
            frame_5, frame_6, frame_7, frame_8
        ][:N]

        # Determine keyframe anchor indices
        if spacing_mode == "manual":
            key_idx = [
                max(0, min(T - 1, int(f)))
                for f in manual_frames
            ]
            key_idx = sorted(key_idx)
        else:
            # Even spacing
            if N == 1:
                key_idx = [0]  # Single image goes at start
            else:
                key_idx = [
                    round(i * (T - 1) / (N - 1))
                    for i in range(N)
                ]

        # Apply keyframes
        for i, (anchor, img_tensor) in enumerate(zip(key_idx, used_imgs)):
            img = img_tensor[0]  # [C, H, W]

            # Place keyframe image
            images_out[anchor] = img

            # Set mask strength based on position
            if anchor == key_idx[0] or anchor == key_idx[-1]:
                masks_out[anchor] = first_last_strength
            else:
                masks_out[anchor] = middle_strength

        # Build keyframes-only batch [N, C, H, W]
        keyframes_batch = torch.cat(used_imgs, dim=0)

        return images_out, masks_out, keyframes_batch


NODE_CLASS_MAPPINGS = {
    "WanKeyframeBuilder": WanKeyframeBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanKeyframeBuilder": "Wan Keyframe Builder",
}
