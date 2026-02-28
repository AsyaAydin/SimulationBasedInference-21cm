# src/train_2C_snpe.py
import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from sbi import utils as U
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
        )
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(x + self.conv(x))


class ShellResNet(nn.Module):
    def __init__(self, emb_dim: int = 60):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.block1 = ResBlock(64)
        self.block2 = ResBlock(64)
        self.block3 = ResBlock(64)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(64 * 4 * 4, emb_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class SNPE_with_dataloader(SNPE):
    def get_dataloaders(self, *args, **kwargs):
        loaders = super().get_dataloaders(*args, **kwargs)
        if isinstance(loaders, tuple) and len(loaders) == 2:
            self.train_loader, self.val_loader = loaders
        return loaders




def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_dataset(dataset_dir: str, device: torch.device):
    X = np.load(os.path.join(dataset_dir, "X.npy"))
    Y = np.load(os.path.join(dataset_dir, "Y.npy"))
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)
    return X, Y


def build_prior(device: torch.device):
    low = torch.tensor([0.1, 0.03, 0.6], dtype=torch.float32)
    high = torch.tensor([0.5, 0.07, 1.0], dtype=torch.float32)
    return U.BoxUniform(low=low, high=high)


def run_snpe_once(X, Y, device, seed: int, epochs: int, batch_size: int, val_frac: float):
    seed_everything(seed)

    embedding_net = ShellResNet().to(device)

    density_estimator_build = posterior_nn(
        model="nsf",
        embedding_net=embedding_net,
    )

    inference = SNPE_with_dataloader(
        prior=build_prior(device),
        density_estimator=density_estimator_build,
        device=device,
    )

    inference.append_simulations(Y, X)
    density_estimator = inference.train(
        max_num_epochs=epochs,
        training_batch_size=batch_size,
        validation_fraction=val_frac,
    )

    posterior = inference.build_posterior(density_estimator)

    val_xs, val_thetas = None, None
    if getattr(inference, "val_loader", None) is not None:
        xs_list, theta_list = [], []
        for batch in inference.val_loader:
            if len(batch) == 2:
                theta_batch, x_batch = batch
            else:  
                theta_batch, x_batch = batch[0], batch[1]
            xs_list.append(x_batch.detach().cpu())
            theta_list.append(theta_batch.detach().cpu())
        val_xs = torch.cat(xs_list, dim=0)
        val_thetas = torch.cat(theta_list, dim=0)

    return posterior, val_xs, val_thetas


def main():
    ap = argparse.ArgumentParser(description="Train SNPE (NSF) with a 2-channel CNN embedding.")
    ap.add_argument("--dataset-dir", required=True, help="Directory containing X.npy and Y.npy")
    ap.add_argument("--out-dir", required=True, help="Output directory for posteriors and samples (ignored by git)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--x-obs-index", type=int, default=8, help="Which simulation index to treat as x_obs")
    ap.add_argument("--num-samples", type=int, default=100000, help="Posterior samples per seed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = load_dataset(args.dataset_dir, device=device)

    posteriors = {}
    val_xs_global, val_thetas_global = None, None

    for i, s in enumerate(args.seeds):
        print(f"[run] seed={s} device={device}", flush=True)
        posterior, val_xs, val_thetas = run_snpe_once(
            X=X, Y=Y, device=device, seed=s,
            epochs=args.epochs, batch_size=args.batch_size, val_frac=args.val_frac,
        )
        posteriors[s] = posterior
        if i == 0:
            val_xs_global, val_thetas_global = val_xs, val_thetas

    torch.save(posteriors, os.path.join(args.out_dir, "posteriors.pt"))

    # sample from ensemble
    x_obs = X[args.x_obs_index : args.x_obs_index + 1]
    all_samp = []
    for s in args.seeds:
        samp = posteriors[s].sample((args.num_samples,), x=x_obs).cpu()
        all_samp.append(samp)
    ensemble_samples = torch.cat(all_samp, dim=0)
    torch.save(ensemble_samples, os.path.join(args.out_dir, "ensemble_samples.pt"))

    # save data
    run_meta = dict(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        x_obs_index=args.x_obs_index,
        num_samples=args.num_samples,
        device=str(device),
        X_shape=list(X.shape),
        Y_shape=list(Y.shape),
        prior_low=[0.1, 0.03, 0.6],
        prior_high=[0.5, 0.07, 1.0],
        model="SNPE + NSF + ShellResNet (2-channel)",
    )
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    print("[ok] training complete")

if __name__ == "__main__":
    main()
