#!/usr/bin/env python3
"""Local Gradio UI for FASHN VTON v1.5 virtual try-on."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from fashn_vton import TryOnPipeline

PIPELINE: Optional[TryOnPipeline] = None


def get_pipeline(weights_dir: str, device: Optional[str]) -> TryOnPipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = TryOnPipeline(weights_dir=weights_dir, device=device)
    return PIPELINE


def run_tryon(
    person_image: Optional[Image.Image],
    garment_image: Optional[Image.Image],
    category: str,
    garment_photo_type: str,
    num_timesteps: int,
    guidance_scale: float,
    seed: int,
    segmentation_free: bool,
) -> Image.Image:
    if person_image is None:
        raise gr.Error("Please upload a person/customer photo.")
    if garment_image is None:
        raise gr.Error("Please upload a garment photo.")

    pipeline = get_pipeline(os.environ["FASHN_WEIGHTS_DIR"], os.environ.get("FASHN_DEVICE") or None)
    result = pipeline(
        person_image=person_image.convert("RGB"),
        garment_image=garment_image.convert("RGB"),
        category=category,
        garment_photo_type=garment_photo_type,
        num_samples=1,
        num_timesteps=int(num_timesteps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        segmentation_free=bool(segmentation_free),
    )
    return result.images[0]


def build_ui() -> gr.Blocks:
    examples_dir = Path(__file__).resolve().parent / "data"
    example_person = str(examples_dir / "model.webp") if (examples_dir / "model.webp").exists() else None
    example_garment = str(examples_dir / "garment.webp") if (examples_dir / "garment.webp").exists() else None

    with gr.Blocks(title="FASHN VTON v1.5") as demo:
        gr.Markdown(
            """
            # FASHN VTON v1.5 — Local Virtual Try-On
            Upload a **customer photo** and a **garment photo**, then generate the try-on result.

            **Note:** On CPU this can take a long time per image. Prefer lower timesteps (e.g. 20) for faster runs.
            """
        )

        with gr.Row():
            with gr.Column():
                person_image = gr.Image(label="Customer / person photo", type="pil", height=360)
                garment_image = gr.Image(label="Garment photo", type="pil", height=360)
            with gr.Column():
                output_image = gr.Image(label="Try-on result", type="pil", height=720)

        with gr.Row():
            category = gr.Dropdown(
                choices=["tops", "bottoms", "one-pieces"],
                value="tops",
                label="Garment category",
            )
            garment_photo_type = gr.Dropdown(
                choices=["model", "flat-lay"],
                value="model",
                label="Garment photo type",
                info="'model' if worn by a person, 'flat-lay' for product shots",
            )

        with gr.Accordion("Advanced settings", open=False):
            num_timesteps = gr.Slider(10, 50, value=20, step=1, label="Diffusion steps")
            guidance_scale = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Guidance scale")
            seed = gr.Number(value=42, precision=0, label="Seed")
            segmentation_free = gr.Checkbox(
                value=True,
                label="Segmentation-free mode",
                info="Usually better body preservation and garment volume",
            )

        run_btn = gr.Button("Generate try-on", variant="primary")
        run_btn.click(
            fn=run_tryon,
            inputs=[
                person_image,
                garment_image,
                category,
                garment_photo_type,
                num_timesteps,
                guidance_scale,
                seed,
                segmentation_free,
            ],
            outputs=output_image,
        )

        if example_person and example_garment:
            gr.Examples(
                examples=[[example_person, example_garment, "tops", "model"]],
                inputs=[person_image, garment_image, category, garment_photo_type],
                label="Sample images",
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="FASHN VTON v1.5 local web UI")
    parser.add_argument("--weights-dir", type=str, default="./weights", help="Directory with model weights")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        raise SystemExit(
            f"Weights directory not found: {weights_dir}\n"
            f"Run: python scripts/download_weights.py --weights-dir {args.weights_dir}"
        )

    os.environ["FASHN_WEIGHTS_DIR"] = str(weights_dir.resolve())
    if args.device:
        os.environ["FASHN_DEVICE"] = args.device

    # Load models once at startup so the first click is faster to start generating.
    print(f"Loading pipeline from {weights_dir} (device={args.device or 'auto'})...", flush=True)
    get_pipeline(os.environ["FASHN_WEIGHTS_DIR"], args.device)
    print(f"UI ready at http://{args.host}:{args.port}", flush=True)

    demo = build_ui()
    demo.queue(max_size=2).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
