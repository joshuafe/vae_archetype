"""
Two-Stage Lymphocyte Archetype Trainer
======================================

Stage 1:
    - Simple gating to retain viable lymphocyte-like cells
    - Rejects myeloid-biased events using composite channels

Stage 2:
    - Lightweight archetypal VAE trained only on the surviving cells
    - Supports single-run execution and random search sweeps
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Panel configuration and Stage 1 heuristics
# ---------------------------------------------------------------------------

CHANNEL_NAMES = [
    'CD16+TIGIT', 'CD8+CD14', 'CD56', 'CD45RA', 'CD366', 'CD69', 'CD25',
    'CD4+CD33', 'CD62L', 'CD152', 'Viability', 'CD45', 'CD45RO',
    'CD279+CD24', 'CD95', 'CD34+CD223', 'CD197', 'CD3+CD19'
]

CHANNEL_ALIASES = {
    'CD25': ['IL2RA', 'IL-2RA', 'CD25BB515', 'CD25-BB515'],
    'CD152': ['CTLA4', 'CTLA-4', 'CD152R718', 'CD152-R718'],
    'Viability': ['LIVEDEAD', 'VIABILITYDYE']
}


def _idx(name: str) -> int:
    try:
        return CHANNEL_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"Channel '{name}' not found in panel") from exc


@dataclass
class StageOneVAEParams:
    latent_dim: int = 8
    hidden_dim: int = 64
    archetype_dim: int = 8
    tau_init: float = 0.8
    tau_final: float = 0.3
    tau_decay_epochs: int = 20
    lr: float = 1e-3
    epochs: int = 40
    batch_size: int = 2048
    kl_weight: float = 1e-3
    recon_weight: float = 1.0
    label_weight: float = 1.0
    entropy_weight: float = 1e-2
    lymph_percentile: float = 65.0
    myeloid_percentile: float = 60.0
    gating_threshold: float = 0.4
    keep_fraction: Optional[float] = 0.20
    max_train_cells: Optional[int] = 200000


def _approx_split(composite: np.ndarray, bias: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    bias = np.clip(bias, 0.05, 0.95)
    a = composite * bias
    b = composite * (1.0 - bias)
    return a, b


def compute_marker_dict(data: np.ndarray) -> Dict[str, np.ndarray]:
    cd3_raw = data[:, _idx('CD3+CD19')]
    cd3, cd19 = _approx_split(cd3_raw, 0.55)
    cd8cd14 = data[:, _idx('CD8+CD14')]
    _, cd14 = _approx_split(cd8cd14, 0.4)
    cd4cd33 = data[:, _idx('CD4+CD33')]
    _, cd33 = _approx_split(cd4cd33, 0.35)
    cd34cd223 = data[:, _idx('CD34+CD223')]
    cd34, _ = _approx_split(cd34cd223, 0.6)
    cd16tigit = data[:, _idx('CD16+TIGIT')]
    cd16, tigit = _approx_split(cd16tigit, 0.6)
    return {
        'viability': data[:, _idx('Viability')],
        'cd3': cd3,
        'cd19': cd19,
        'cd56': data[:, _idx('CD56')],
        'cd14': cd14,
        'cd33': cd33,
        'cd34': cd34,
        'cd16': cd16,
        'tigit': tigit
    }


class StageOneVAENet(nn.Module):
    def __init__(self, input_dim: int, params: StageOneVAEParams):
        super().__init__()
        hidden = params.hidden_dim
        latent = params.latent_dim
        arch_dim = params.archetype_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.archetype_layer = ArchetypeLayer(hidden, params.latent_dim, arch_dim, tau=params.tau_init)
        self.decoder = nn.Sequential(
            nn.Linear(arch_dim, max(hidden // 2, arch_dim)),
            nn.LeakyReLU(0.2),
            nn.Linear(max(hidden // 2, arch_dim), input_dim)
        )
        self.gating_head = nn.Linear(arch_dim, 1)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, h

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, h = self.encode(x)
        z = self.reparam(mu, logvar)
        arch_latent, alpha = self.archetype_layer(h)
        recon = self.decoder(arch_latent)
        logits = self.gating_head(arch_latent)
        return recon, mu, logvar, alpha, logits


class SimplePanelConfig:
    def __init__(self):
        self.input_channels = CHANNEL_NAMES
        self.input_idx = {ch: i for i, ch in enumerate(self.input_channels)}
        self.viability_idx = self.input_idx['Viability']


class SimpleFCSLoader:
    def __init__(self, config, arcsinh_cofactor=5.0):
        self.config = config
        self.arcsinh_cofactor = arcsinh_cofactor

    def load_fcs(self, filepath: str, sample_name: Optional[str] = None):
        try:
            from fcsparser import parse
        except ImportError as exc:
            raise ImportError("Please install fcsparser to load FCS files.") from exc

        meta, df = parse(filepath, reformat_meta=True)

        def normalize(name: str) -> str:
            return ''.join(ch for ch in name.upper() if ch.isalnum())

        normalized_cols = {}
        for col in df.columns:
            key = normalize(col)
            if key not in normalized_cols:
                normalized_cols[key] = col

        columns_in_order: List[str] = []
        missing: List[str] = []
        for channel in self.config.input_channels:
            candidates = [channel] + CHANNEL_ALIASES.get(channel, [])
            matched = None
            for cand in candidates:
                key = normalize(cand)
                if key in normalized_cols:
                    matched = normalized_cols[key]
                    break
            if matched is None:
                missing.append(channel)
            else:
                columns_in_order.append(matched)

        if missing:
            raise ValueError(
                f"File {filepath} missing channels: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        data = df[columns_in_order].to_numpy(dtype=np.float32)
        return data, meta.get('$FIL', sample_name or Path(filepath).stem)

    def preprocess(self, data: np.ndarray, apply_arcsinh: bool = True, qc_viability: bool = True):
        processed = data.copy()
        if qc_viability:
            viability = processed[:, self.config.viability_idx]
            v_min, v_max = viability.min(), viability.max()
            if v_max > v_min:
                norm = (viability - v_min) / (v_max - v_min)
                mask = norm > 0.1
                processed = processed[mask]
        if apply_arcsinh:
            processed = np.arcsinh(processed / self.arcsinh_cofactor)
        return processed


# ---------------------------------------------------------------------------
# Stage 2 dataset and model
# ---------------------------------------------------------------------------

class LymphocyteDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StageOneDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ArchetypeLayer(nn.Module):
    def __init__(self, input_dim: int, n_archetypes: int, archetype_dim: int, tau: float = 0.5):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_archetypes)
        self.tau = tau
        self.register_parameter(
            "archetypes",
            nn.Parameter(torch.randn(n_archetypes, archetype_dim) * 0.1)
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.encoder(h)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        alpha = F.softmax(logits / self.tau, dim=-1)
        archetypes = F.normalize(self.archetypes, p=2, dim=1)
        z = alpha @ archetypes
        return z, alpha


class TwoStageArchetypeVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 128,
                 n_archetypes: int = 12, archetype_dim: int = 16, tau: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.archetype_layer = ArchetypeLayer(hidden_dim, n_archetypes, archetype_dim, tau=tau)

        decoder_dim = max(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(archetype_dim, decoder_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(decoder_dim, input_dim)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, h

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar, h = self.encode(x)
        z = self.reparameterize(mu, logvar)
        archetype_latent, alpha = self.archetype_layer(h)
        recon = self.decoder(archetype_latent)
        return recon, mu, logvar, alpha


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

@dataclass
class StageTwoParams:
    latent_dim: int = 12
    hidden_dim: int = 128
    n_archetypes: int = 12
    archetype_dim: int = 12
    tau_init: float = 0.6
    tau_final: float = 0.05
    tau_decay_epochs: int = 100
    lr: float = 5e-4
    kl_weight: float = 5e-3
    entropy_weight: float = 0.03
    purity_margin: float = 0.6
    purity_weight: float = 0.4
    epochs: int = 150
    batch_size: int = 512
    stage_epoch: int = 100
    stage_entropy_scale: float = 1.5
    stage_purity_scale: float = 1.8


def _kl_gaussian(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def _to_serialisable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    return obj


def train_stage_one(data: np.ndarray, params: StageOneVAEParams, device: str = "cpu"):
    markers = compute_marker_dict(data)
    viability = markers['viability']

    def pct(x, p):
        return np.percentile(x, p)

    cd3_thr = pct(markers['cd3'], params.lymph_percentile)
    cd19_thr = pct(markers['cd19'], params.lymph_percentile)
    cd56_thr = pct(markers['cd56'], params.lymph_percentile)

    lymph_score = (
        (markers['cd3'] > cd3_thr).astype(float) +
        (markers['cd19'] > cd19_thr).astype(float) +
        (markers['cd56'] > cd56_thr).astype(float)
    ) / 3.0

    cd14_thr = pct(markers['cd14'], params.myeloid_percentile)
    cd33_thr = pct(markers['cd33'], params.myeloid_percentile)
    cd34_thr = pct(markers['cd34'], params.myeloid_percentile)

    myeloid_score = (
        (markers['cd14'] > cd14_thr).astype(float) +
        (markers['cd33'] > cd33_thr).astype(float) +
        (markers['cd34'] > cd34_thr).astype(float)
    ) / 3.0

    myeloid_mask = myeloid_score > 0.5

    viability_norm = (viability - viability.min()) / (viability.max() - viability.min() + 1e-6)
    soft_labels = np.clip(lymph_score * viability_norm - 0.5 * myeloid_score, 0.0, 1.0)

    train_mask = np.ones(len(data), dtype=bool)
    if params.max_train_cells is not None and len(data) > params.max_train_cells:
        idx = np.random.choice(len(data), params.max_train_cells, replace=False)
        train_data = data[idx]
        train_labels = soft_labels[idx]
    else:
        train_data = data
        train_labels = soft_labels

    dataset = StageOneDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, drop_last=False)

    model = StageOneVAENet(data.shape[1], params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=1e-4)

    def tau_schedule(epoch: int):
        if epoch >= params.tau_decay_epochs:
            return params.tau_final
        ratio = epoch / max(1, params.tau_decay_epochs)
        return params.tau_init + (params.tau_final - params.tau_init) * ratio

    for epoch in tqdm(range(params.epochs), desc="Stage 1 VAE", leave=False):
        model.train()
        tau = tau_schedule(epoch)
        model.archetype_layer.tau = max(tau, 1e-3)
        for batch, labels in loader:
            batch = batch.to(device)
            labels = labels.to(device)
            recon, mu, logvar, alpha, logits = model(batch)
            recon_loss = F.mse_loss(recon, batch, reduction='mean')
            kl_loss = _kl_gaussian(mu, logvar).mean()
            entropy = -(alpha * torch.log(alpha + 1e-10)).sum(dim=1).mean()
            label_loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = (
                params.recon_weight * recon_loss +
                params.kl_weight * kl_loss +
                params.label_weight * label_loss +
                params.entropy_weight * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        torch_data = torch.tensor(data, dtype=torch.float32, device=device)
        recon, mu, logvar, alpha, logits = model(torch_data)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    thresholds = [params.gating_threshold]
    if soft_labels.mean() > 0:
        quant = np.clip(1 - soft_labels.mean(), 0.0, 1.0)
        thresholds.append(np.quantile(probs, quant))
    if params.keep_fraction is not None and 0.0 < params.keep_fraction < 1.0:
        keep_thr = np.quantile(probs, 1 - params.keep_fraction)
        thresholds.append(keep_thr)
    threshold = max(thresholds)
    pass_mask = probs >= threshold

    info = {
        'probabilities': probs,
        'soft_labels': soft_labels,
        'threshold': threshold,
        'tau_final': model.archetype_layer.tau,
        'viability': markers['viability'],
        'cd3': markers['cd3'],
        'cd19': markers['cd19'],
        'cd56': markers['cd56'],
        'cd14': markers['cd14'],
        'cd33': markers['cd33'],
        'cd34': markers['cd34'],
        'myeloid_mask': myeloid_mask.astype(bool)
    }

    return data[pass_mask], pass_mask, info, model.state_dict()


def train_stage_two(model: TwoStageArchetypeVAE, data: np.ndarray, params: StageTwoParams,
                    device: str = "cpu") -> Dict[str, float]:
    dataset = LymphocyteDataset(data)
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, drop_last=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=1e-4)

    def tau_schedule(epoch: int):
        if epoch >= params.tau_decay_epochs:
            return params.tau_final
        ratio = epoch / max(1, params.tau_decay_epochs)
        return params.tau_init + (params.tau_final - params.tau_init) * ratio

    history = {
        'loss': [], 'recon': [], 'kl': [], 'entropy': [], 'purity_loss': [], 'tau': []
    }

    epoch_iter = tqdm(range(params.epochs), desc="Stage 2 Training", leave=False)
    for epoch in epoch_iter:
        model.train()
        epoch_losses = {'loss': 0.0, 'recon': 0.0, 'kl': 0.0, 'entropy': 0.0, 'purity': 0.0}
        tau = tau_schedule(epoch)
        model.archetype_layer.tau = max(tau, 1e-3)

        for batch in loader:
            batch = batch.to(device)
            recon, mu, logvar, alpha = model(batch)

            recon_loss = F.mse_loss(recon, batch, reduction='mean')
            kl_loss = _kl_gaussian(mu, logvar).mean()
            entropy = -(alpha * torch.log(alpha + 1e-10)).sum(dim=1).mean()

            max_alpha = alpha.max(dim=1).values
            purity_hinge = torch.clamp(params.purity_margin - max_alpha, min=0.0)
            purity_loss = purity_hinge.mean()

            entropy_weight = params.entropy_weight
            purity_weight = params.purity_weight
            if epoch >= params.stage_epoch:
                entropy_weight *= params.stage_entropy_scale
                purity_weight *= params.stage_purity_scale

            loss = (
                recon_loss +
                params.kl_weight * kl_loss +
                entropy_weight * entropy +
                purity_weight * purity_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_losses['loss'] += loss.item() * len(batch)
            epoch_losses['recon'] += recon_loss.item() * len(batch)
            epoch_losses['kl'] += kl_loss.item() * len(batch)
            epoch_losses['entropy'] += entropy.item() * len(batch)
            epoch_losses['purity'] += purity_loss.item() * len(batch)

        n = len(dataset)
        history['loss'].append(epoch_losses['loss'] / n)
        history['recon'].append(epoch_losses['recon'] / n)
        history['kl'].append(epoch_losses['kl'] / n)
        history['entropy'].append(epoch_losses['entropy'] / n)
        history['purity_loss'].append(epoch_losses['purity'] / n)
        history['tau'].append(model.archetype_layer.tau)

        if (epoch + 1) % 20 == 0 or epoch == params.epochs - 1:
            epoch_iter.set_postfix({
                'loss': f"{history['loss'][-1]:.4f}",
                'purity': f"{1 - history['purity_loss'][-1]:.4f}",
                'tau': f"{model.archetype_layer.tau:.3f}"
            })

    model.eval()
    with torch.no_grad():
        all_batch = torch.tensor(data, dtype=torch.float32).to(device)
        recon, mu, logvar, alpha = model(all_batch)
        recon_loss = F.mse_loss(recon, all_batch, reduction='mean').item()
        kl_loss = _kl_gaussian(mu, logvar).mean().item()
        max_alpha = alpha.max(dim=1).values.cpu().numpy()
        archetype_weights = model.archetype_layer.archetypes.detach().cpu()
        archetype_profiles = model.decoder(archetype_weights.to(device)).cpu().numpy()
        alpha_np = alpha.cpu().numpy()

    metrics = {
        'final_recon': recon_loss,
        'final_kl': kl_loss,
        'mean_purity': float(max_alpha.mean()),
        'median_purity': float(np.median(max_alpha)),
        'history': history,
        'alpha': alpha_np,
        'archetype_profiles': archetype_profiles,
        'alpha_mean_usage': alpha_np.mean(axis=0)
    }
    return metrics


# ---------------------------------------------------------------------------
# Driver functions
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    stage1: StageOneVAEParams
    stage2: StageTwoParams


def run_single_experiment(data: np.ndarray, config: ExperimentConfig,
                          device: str = "cpu") -> Dict[str, float]:
    stage1_params = config.stage1
    filtered, mask, info, stage1_state = train_stage_one(data, stage1_params, device=device)
    total_cells = len(data)
    passed_cells = len(filtered)
    myeloid_rejected = int(info['myeloid_mask'].sum())
    print(f"Stage 1 summary: {passed_cells}/{total_cells} cells passed ({passed_cells / max(total_cells, 1):.2%}).")
    print(f"  Rejected myeloid-like cells: {myeloid_rejected}")
    print(f"  Gating threshold: {info['threshold']:.4f}, keep fraction: {stage1_params.keep_fraction if stage1_params.keep_fraction else 'auto'}")
    if passed_cells == 0:
        raise RuntimeError("Stage 1 filtered every cell; relax thresholds.")

    model = TwoStageArchetypeVAE(
        input_dim=data.shape[1],
        latent_dim=config.stage2.latent_dim,
        hidden_dim=config.stage2.hidden_dim,
        n_archetypes=config.stage2.n_archetypes,
        archetype_dim=config.stage2.archetype_dim,
        tau=config.stage2.tau_init
    )
    metrics = train_stage_two(model, filtered, config.stage2, device=device)
    metrics['stage1_pass_rate'] = float(len(filtered)) / float(len(data))
    metrics['stage1_mask'] = mask
    metrics['stage1_info'] = info
    metrics['stage1_model_state_dict'] = stage1_state
    metrics['config'] = {
        'stage1': asdict(config.stage1),
        'stage2': asdict(config.stage2)
    }
    metrics['model_state_dict'] = model.state_dict()
    print(
        f"Stage 2 final metrics: recon={metrics['final_recon']:.4f}, "
        f"KL={metrics['final_kl']:.4f}, mean_purity={metrics['mean_purity']:.4f}"
    )

    return metrics


def save_visual_summary(metrics: Dict[str, float], output_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)

    history = metrics['history']
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['loss'], label='Total loss')
    plt.plot(epochs, history['recon'], label='Reconstruction')
    plt.plot(epochs, history['kl'], label='KL')
    plt.plot(epochs, history['purity_loss'], label='Purity loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Stage 2 Training Curves')
    plt.tight_layout()
    plt.savefig(output_dir / 'stage2_training_curves.png', dpi=150)
    plt.close()

    alpha_usage = metrics['alpha_mean_usage']
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(range(len(alpha_usage))), y=alpha_usage)
    plt.xlabel('Archetype')
    plt.ylabel('Mean usage')
    plt.title('Archetype Usage')
    plt.tight_layout()
    plt.savefig(output_dir / 'stage2_archetype_usage.png', dpi=150)
    plt.close()

    archetypes = metrics['archetype_profiles']
    plt.figure(figsize=(12, 6))
    sns.heatmap(archetypes, cmap='vlag', center=0, annot=False)
    plt.xlabel('Marker')
    plt.ylabel('Archetype')
    plt.title('Decoded Archetype Profiles')
    plt.tight_layout()
    plt.savefig(output_dir / 'stage2_archetype_profiles.png', dpi=150)
    plt.close()

    return metrics


def run_random_sweep(data: np.ndarray,
                     stage1_grid: Dict[str, Iterable[float]],
                     stage2_grid: Dict[str, Iterable[float]],
                     n_trials: int = 10,
                     device: str = "cpu",
                     seed: int = 42,
                     output_dir: Optional[Path] = None) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)

    def sample_params(grid: Dict[str, Iterable[float]], base: dataclass):
        params_dict = asdict(base)
        for key, values in grid.items():
            values = list(values)
            if not values:
                continue
            params_dict[key] = values[int(rng.integers(len(values)))]
        return params_dict

    results = []
    for trial in range(1, n_trials + 1):
        print(f"\n=== Random Trial {trial}/{n_trials} ===")
        stage1_params = StageOneVAEParams(**sample_params(stage1_grid, StageOneVAEParams()))
        stage2_params = StageTwoParams(**sample_params(stage2_grid, StageTwoParams()))
        config = ExperimentConfig(stage1=stage1_params, stage2=stage2_params)
        try:
            metrics = run_single_experiment(data, config, device=device)
            metrics['trial'] = trial
            results.append(metrics)
        except RuntimeError as exc:
            print(f"  Trial skipped: {exc}")
            continue

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            trial_path = output_dir / f"trial_{trial:02d}.json"
            serialisable = {
                k: _to_serialisable(v)
                for k, v in metrics.items()
                if k not in {'model_state_dict', 'stage1_model_state_dict', 'alpha', 'history'}
            }
            with open(trial_path, 'w') as f:
                json.dump(serialisable, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage lymphocyte archetype trainer.")
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to numpy .npy array (cells × channels). Optional if --fcs is provided."
    )
    parser.add_argument(
        "--fcs",
        nargs="+",
        help="Paths or glob patterns to FCS files. Optional if --data is provided."
    )
    parser.add_argument(
        "--sample-name",
        default=None,
        help="Sample name to assign when loading FCS files (defaults to filename)."
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda:0).")
    parser.add_argument("--mode", choices=["single", "sweep"], default="single")
    parser.add_argument("--output", type=Path, default=Path("./two_stage_output"),
                        help="Directory to save results.")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials for sweep mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.data is None and not args.fcs:
        raise ValueError("Provide either --data or --fcs input.")

    if args.data is not None:
        data = np.load(args.data)
    else:
        import glob
        config = SimplePanelConfig()
        loader = SimpleFCSLoader(config, arcsinh_cofactor=5.0)
        all_arrays = []
        for pattern in args.fcs:
            if any(ch in pattern for ch in "*?[]"):
                matches = sorted(Path(p) for p in glob.glob(pattern))
            else:
                matches = [Path(pattern)]
            if not matches:
                raise FileNotFoundError(f"No files match '{pattern}'")
            for path in matches:
                data_arr, _ = loader.load_fcs(str(path), sample_name=args.sample_name or path.stem)
                data_arr = loader.preprocess(data_arr, apply_arcsinh=True, qc_viability=False)
                all_arrays.append(data_arr)
        data = np.concatenate(all_arrays, axis=0)
    device = args.device
    output_dir = args.output

    if args.mode == "single":
        config = ExperimentConfig(stage1=StageOneVAEParams(), stage2=StageTwoParams())
        metrics = run_single_experiment(data, config, device=device)
        output_dir.mkdir(parents=True, exist_ok=True)
        serialisable = {
            k: _to_serialisable(v)
            for k, v in metrics.items()
            if k not in {'model_state_dict', 'stage1_model_state_dict', 'alpha', 'history'}
        }
        with open(output_dir / "single_result.json", 'w') as f:
            json.dump(serialisable, f, indent=2)
        torch.save(metrics['model_state_dict'], output_dir / "stage2_model.pt")
        np.save(output_dir / "stage1_mask.npy", metrics['stage1_mask'])
        np.save(output_dir / "alpha.npy", metrics['alpha'])
        np.save(output_dir / "archetype_profiles.npy", metrics['archetype_profiles'])
        save_visual_summary(metrics, output_dir)
        print("\nSingle run complete. Results saved to", output_dir)
    else:
        stage1_grid = {
            'lymph_percentile': [60.0, 65.0, 70.0],
            'myeloid_percentile': [55.0, 60.0, 65.0],
            'gating_threshold': [0.4, 0.5, 0.6],
            'latent_dim': [6, 8, 10],
            'archetype_dim': [6, 8, 10]
        }
        stage2_grid = {
            'n_archetypes': [10, 12, 15],
            'tau_init': [0.6, 0.5],
            'tau_final': [0.2, 0.1],
            'entropy_weight': [0.01, 0.02, 0.03],
            'purity_margin': [0.6, 0.65, 0.7],
            'purity_weight': [0.4, 0.6, 0.8]
        }
        results = run_random_sweep(
            data,
            stage1_grid=stage1_grid,
            stage2_grid=stage2_grid,
            n_trials=args.n_trials,
            device=device,
            output_dir=output_dir
        )
        print(f"\nSweep complete. {len(results)} trials succeeded. Results saved to {output_dir}.")


if __name__ == "__main__":
    main()
