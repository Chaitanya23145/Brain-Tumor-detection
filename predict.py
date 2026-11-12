"""
Clinical Brain Tumor MRI Inference CLI
Accepts an image path, executes skull-stripping preprocessing, predicts malignancy,
and saves the Grad-CAM activation overlay.
"""
import os
import argparse
import cv2
from PIL import Image

from src.model import predict_mri


def main():
    parser = argparse.ArgumentParser(description="Clinical Brain Tumor Detection & Grad-CAM CLI")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to brain MRI image (JPG/PNG)")
    parser.add_argument("--output-dir", "-o", type=str, default="results", help="Directory to save visual explanation")
    parser.add_argument("--no-gradcam", action="store_true", help="Disable Grad-CAM heatmap generation")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        return

    print(f"[INFO] Analyzing MRI scan: {args.image}")
    result = predict_mri(args.image, include_gradcam=not args.no_gradcam)

    print("\n" + "="*50)
    print("           DIAGNOSTIC REPORT")
    print("="*50)
    print(f"Classification:         {result['label']}")
    print(f"Confidence Score:       {result['confidence_percentage']}%")
    print(f"Posterior Probabilities:")
    print(f"  • Healthy / No Tumor: {result['probabilities']['No Tumor'] * 100:.2f}%")
    print(f"  • Neoplasm / Tumor:   {result['probabilities']['Tumor Present'] * 100:.2f}%")
    print(f"Model Checkpoint State: {result['status']}")
    print("="*50)

    if not args.no_gradcam and result.get("gradcam_image") is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        gradcam_path = os.path.join(args.output_dir, f"{base_name}_gradcam.png")
        cropped_path = os.path.join(args.output_dir, f"{base_name}_roi_cropped.png")

        cv2.imwrite(gradcam_path, cv2.cvtColor(result["gradcam_image"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(cropped_path, cv2.cvtColor(result["cropped_image"], cv2.COLOR_RGB2BGR))

        print(f"\n[INFO] Saved ROI Cropped image: {cropped_path}")
        print(f"[INFO] Saved Grad-CAM overlay:   {gradcam_path}")


if __name__ == "__main__":
    main()

