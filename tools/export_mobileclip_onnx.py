"""Export a MobileCLIP v1/v2 image encoder to a static OpenCV ONNX.

MobileCLIP2 additionally requires Apple's OpenCLIP patch described in the
official ml-mobileclip repository. The script verifies OpenCV-DNN parity before
the model is considered usable by the experimental profile.

Requires the separate .venv-mobileclip-win environment and Apple's checkpoint.
The resulting model is intentionally not distributed by this project.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mobileclip
import torch


class ImageEncoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embedding = self.model.encode_image(image)
        return embedding / embedding.norm(dim=-1, keepdim=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("mobileclip_s0", "MobileCLIP2-S0"), default="mobileclip_s0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    is_v2 = args.architecture.startswith("MobileCLIP2")
    args.checkpoint = args.checkpoint or Path("models/mobileclip2_s0.pt" if is_v2 else "models/mobileclip_s0.pt")
    args.output = args.output or Path("models/experimental/mobileclip2_s0_image.onnx" if is_v2 else "models/stable/mobileclip_s0_image.onnx")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if is_v2:
        try:
            import open_clip
        except ImportError as exc:
            raise SystemExit("MobileCLIP2 export requires OpenCLIP with Apple's MobileCLIP2 patch.") from exc
        model, _, _ = open_clip.create_model_and_transforms(
            args.architecture,
            pretrained=str(args.checkpoint),
            image_mean=(0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0),
            device="cpu",
        )
    else:
        model, _, _ = mobileclip.create_model_and_transforms(args.architecture, pretrained=str(args.checkpoint), device="cpu")
    if is_v2:
        # Apple's V2 instructions explicitly require this before exporting.
        # The released V1 S0 checkpoint loaded by `mobileclip` is already in
        # inference form and reparameterizing it a second time is invalid.
        from mobileclip.modules.common.mobileone import reparameterize_model

        model = reparameterize_model(model)
    model = model.eval()
    wrapper = ImageEncoder(model).eval()
    generator = torch.Generator().manual_seed(20260715)
    example = torch.rand((1, 3, 256, 256), dtype=torch.float32, generator=generator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (example,),
        str(args.output),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=18,
        dynamo=False,
    )
    if args.verify:
        import cv2
        import numpy as np

        with torch.inference_mode():
            expected = wrapper(example).cpu().numpy().reshape(-1)
        net = cv2.dnn.readNetFromONNX(str(args.output))
        net.setInput(example.numpy())
        actual = net.forward().reshape(-1)
        cosine = float(expected @ actual / max(1e-12, np.linalg.norm(expected) * np.linalg.norm(actual)))
        max_abs = float(np.max(np.abs(expected - actual)))
        if cosine < 0.9999:
            args.output.unlink(missing_ok=True)
            raise SystemExit(f"OpenCV parity failed: cosine={cosine:.8f}, max_abs={max_abs:.8g}")
        print(f"OpenCV parity: cosine={cosine:.8f}, max_abs={max_abs:.8g}")
    print(f"Exported {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
