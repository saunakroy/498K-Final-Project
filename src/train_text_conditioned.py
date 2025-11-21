import torch
import torch.nn as nn
import torch.optim as optim

from fusion_model import TextGaussianFusionModel
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from datasets import ImageDataset


def train_text_conditioned(
        data_path,
        text_prompt="a red shiny object",
        iterations=3000,
        lr=0.001,
        device="cuda"
):
    # Load dataset
    dataset = ImageDataset(data_path)
    
    # Load Gaussian Splatting model
    gs_model = GaussianModel(device=device)

    # Load text fusion module
    fusion = TextGaussianFusionModel(
        gaussian_dim=gs_model.feature_dim,
        device=device
    ).to(device)

    optimizer = optim.Adam(
        list(gs_model.parameters()) + list(fusion.parameters()), 
        lr=lr
    )

    for it in range(iterations):
        batch = dataset.sample_random_image()

        # Extract Gaussian features (N, F)
        gaussian_feats = gs_model.get_features() 

        # Fuse with text
        fused_feats = fusion(text_prompt, gaussian_feats)

        # Put fused features back into GS model
        gs_model.set_features(fused_feats)

        # Render image
        pred_img = render(gs_model, batch.camera)

        # Compute loss
        loss = ((pred_img - batch.image)**2).mean()

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 100 == 0:
            print(f"Iter {it} | Loss: {loss.item():.4f}")

    print("Training complete. Saving...")
    gs_model.save("output/text_conditioned_model")

if __name__ == "__main__":
    train_text_conditioned(
        data_path="data/nerf_synthetic/hotdog",
        text_prompt="a plastic red toy",
        iterations=1000,
        lr=0.0005
    )
