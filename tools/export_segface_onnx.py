"""Export and parity-check the official SegFace MobileNet checkpoint.

This is an isolated spike. Its output is not selected by a model profile until
mask quality, OpenCV compatibility and CPU speed are reviewed on real photos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True, help="Checkout of Kartik-3004/SegFace.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="CelebAMask-HQ MobileNet checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("models/experimental/segface_mobilenet_celeb_512.onnx"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not (args.repo / "network" / "__init__.py").is_file():
        raise SystemExit(f"Not a SegFace checkout: {args.repo}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from network import get_model

    model = get_model("segface_celeb", args.resolution, "mobilenet")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict_backbone", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    class ImageOnly(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            dataset = torch.zeros((image.shape[0],), dtype=torch.long, device=image.device)
            return self.wrapped(image, {}, dataset)

    wrapper = ImageOnly(model).eval()
    generator = torch.Generator().manual_seed(20260715)
    example = torch.rand((1, 3, args.resolution, args.resolution), generator=generator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (example,),
        str(args.output),
        input_names=["image"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
    )
    if args.verify:
        import cv2
        import numpy as np

        with torch.inference_mode():
            expected = wrapper(example).cpu().numpy()
        net = cv2.dnn.readNetFromONNX(str(args.output))
        net.setInput(example.numpy())
        actual = net.forward()
        pixel_agreement = float(np.mean(expected.argmax(axis=1) == actual.argmax(axis=1)))
        max_abs = float(np.max(np.abs(expected - actual)))
        if pixel_agreement < 0.999:
            args.output.unlink(missing_ok=True)
            raise SystemExit(f"OpenCV parity failed: pixel_agreement={pixel_agreement:.6f}, max_abs={max_abs:.8g}")
        print(f"OpenCV parity: pixel_agreement={pixel_agreement:.6f}, max_abs={max_abs:.8g}")
    print(f"Exported spike model: {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
