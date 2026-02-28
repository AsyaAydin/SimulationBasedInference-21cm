# src/train_field_snpe.py
import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from sbi import utils as U
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn


from deepsphere.utils.laplacian_funcs import get_healpix_laplacians
from deepsphere.models.spherical_unet.utils import SphericalChebBN, SphericalChebBNPool
from deepsphere.layers.samplings.healpix_pool_unpool import Healpix


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FieldHealpixDataset(Dataset):
    def __init__(self, files, use_meanTb: bool = True):
        self.files = list(files)
        self.use_meanTb = use_meanTb

        d0 = np.load(self.files[0])
        self.nside = int(d0["nside"])
        self.npix = 12 * self.nside**2

        X0 = d0["X"]
        self.n_shells = int(X0.shape[0])

        # check meanTb existence
        self.has_meanTb = ("meanTb_global" in d0.files) and use_meanTb

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        X = d["X"].astype(np.float32)   # (S, Npix)
        y = d["y"].astype(np.float32)   # (3,)

        # DeepSphere expects [B, Npix, channels]
        X = torch.from_numpy(X).T       # (Npix, S)

        if self.has_meanTb:
            meanTb = torch.from_numpy(d["meanTb_global"].astype(np.float32))  # (1,)
        else:
            meanTb = torch.zeros(1, dtype=torch.float32)

        return X, torch.from_numpy(y), meanTb


class DeepSphereEmbedding(nn.Module):
    def __init__(self, nside: int, in_channels: int, emb_dim: int = 64,
                 depth: int = 3, kernel_size: int = 5, laplacian_type: str = "normalized"):
        super().__init__()
        self.npix = 12 * nside**2

        lap_list = get_healpix_laplacians(self.npix, depth, laplacian_type)
        lap_coarse, lap_mid, lap_fine = lap_list[0], lap_list[1], lap_list[2]

        healpix_pool = Healpix().pooling

        self.conv1 = SphericalChebBN(
            in_channels=in_channels, out_channels=32,
            lap=lap_fine, kernel_size=kernel_size,
        )
        self.block2 = SphericalChebBNPool(
            in_channels=32, out_channels=64,
            lap=lap_mid, pooling=healpix_pool, kernel_size=kernel_size,
        )
        self.block3 = SphericalChebBNPool(
            in_channels=64, out_channels=128,
            lap=lap_coarse, pooling=healpix_pool, kernel_size=kernel_size,
        )
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        # x: [B, Npix, S_shells]
        x = self.conv1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=1)   # global mean over pixels -> [B, 128]
        return self.fc(x)   # [B, emb_dim]


class MeanTbConcat(nn.Module):
    """
    Wrap embedding: output z = [emb, meanTb] so sbi sees one vector.
    """
    def __init__(self, embedding: nn.Module, emb_dim: int, use_meanTb: bool = True):
        super().__init__()
        self.embedding = embedding
        self.use_meanTb = use_meanTb
        self.emb_dim = emb_dim

    def forward(self, x):
        # x will be provided by sbi; we pack meanTb into x if we want
        # So: we require meanTb to be appended as an extra "channel" OR we disable meanTb here.
        raise RuntimeError("Use the packed-input version below.")


def main():
    ap = argparse.ArgumentParser(description="Field-level SNPE with DeepSphere embedding.")
    ap.add_argument("--data-dir", required=True, help="Directory with sim_XXXX_dtb_healpix.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--use-meanTb", action="store_true", help="If dataset contains meanTb_global, append it to embedding.")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".npz")])
    dataset = FieldHealpixDataset(files, use_meanTb=args.use_meanTb)
    N = len(dataset)
    n_val = int(args.val_frac * N)
    n_train = N - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Build embedding
    embedding = DeepSphereEmbedding(
        nside=dataset.nside,
        in_channels=dataset.n_shells,
        emb_dim=args.emb_dim,
        depth=3,
        kernel_size=5,
    ).to(device)

    # IMPORTANT: SBI expects x to be a tensor. We cannot pass meanTb separately without custom wrapper.
    # So: if you want meanTb, easiest is to append it as an extra constant channel to x before embedding.
    # We'll do that here.
    def pack_x(X, meanTb):
        # X: (B, Npix, S)
        if args.use_meanTb:
            # broadcast meanTb as an extra channel across pixels
            B, Npix, S = X.shape
            tb_chan = meanTb.view(B, 1, 1).expand(B, Npix, 1)
            return torch.cat([X, tb_chan], dim=2)  # (B, Npix, S+1)
        return X

    # embedding in_channels update if meanTb is used
    if args.use_meanTb:
        embedding = DeepSphereEmbedding(
            nside=dataset.nside,
            in_channels=dataset.n_shells + 1,
            emb_dim=args.emb_dim,
            depth=3,
            kernel_size=5,
        ).to(device)

    # Prepare tensors for SBI
    X_list, Y_list = [], []
    for X, y, meanTb in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
        X = X.to(device)             # (B, Npix, S)
        meanTb = meanTb.to(device)   # (B, 1)
        x_packed = pack_x(X, meanTb)
        X_list.append(x_packed.detach().cpu())
        Y_list.append(y.detach().cpu())

    X_all = torch.cat(X_list, dim=0).to(device)
    Y_all = torch.cat(Y_list, dim=0).to(device)

    # prior
    prior = U.BoxUniform(
        low=torch.tensor([0.1, 0.03, 0.6], dtype=torch.float32),
        high=torch.tensor([0.5, 0.07, 1.0], dtype=torch.float32),
    )

    density_estimator_build = posterior_nn(
        model="nsf",
        embedding_net=embedding,
    )

    inference = SNPE(prior=prior, density_estimator=density_estimator_build, device=device)
    inference.append_simulations(Y_all, X_all)
    density_estimator = inference.train(max_num_epochs=args.epochs, training_batch_size=args.batch_size)

    posterior = inference.build_posterior(density_estimator)
    torch.save(posterior, os.path.join(args.out_dir, "posterior.pt"))

    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump(
            dict(
                seed=args.seed,
                batch_size=args.batch_size,
                epochs=args.epochs,
                emb_dim=args.emb_dim,
                nside=dataset.nside,
                n_shells=dataset.n_shells,
                use_meanTb=args.use_meanTb,
                device=str(device),
                X_shape=list(X_all.shape),
                Y_shape=list(Y_all.shape),
            ),
            f,
            indent=2,
        )

if __name__ == "__main__":
    main()
