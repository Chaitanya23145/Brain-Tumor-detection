"""
Automated One-Click Deployment Script for Hugging Face Spaces.
Uses huggingface_hub API to upload only production assets, handling large files natively.
"""
import os
import sys
import argparse
from huggingface_hub import HfApi


def deploy(token: str, repo_id: str = "chaitanya231450/brain-tumor-detection"):
    print(f"[INFO] Deploying to Hugging Face Space: https://huggingface.co/spaces/{repo_id}")
    api = HfApi(token=token)

    allow_patterns = [
        "app.py",
        "requirements.txt",
        "packages.txt",
        "README.md",
        "src/**",
        "model_weights/vgg19_model_03.weights.h5",
        "model_weights/metrics.json",
        "uploads/**"
    ]

    print("[INFO] Uploading production files (app.py, model weights, src, sample scans)...")
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=allow_patterns,
        commit_message="Deploy production Clinical Decision Support System"
    )
    print("\n" + "="*60)
    print(" [SUCCESS] Deployment Completed!")
    print(f" View your live web app at: https://huggingface.co/spaces/{repo_id}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload to Hugging Face Space")
    parser.add_argument("--token", "-t", type=str, required=True, help="Hugging Face Access Token (Write permissions)")
    parser.add_argument("--repo-id", "-r", type=str, default="chaitanya231450/brain-tumor-detection", help="Target Space ID")
    args = parser.parse_args()

    deploy(args.token, args.repo_id)

