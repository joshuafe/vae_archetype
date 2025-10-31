# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import json
import pickle
import warnings
import time
from itertools import product

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

warnings.filterwarnings('ignore')

# FCS loading
try:
    import fcsparser
    print("✅ fcsparser available")
except ImportError:
    print("⚠️  fcsparser not installed. Install with: pip install fcsparser")
    print("    Will use dummy data for demonstration")

import glob

device = torch.device('mps' if torch.backends.mps.is_available() else 
                     'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")



CHANNEL_NAMES = [
    'CD16+TIGIT',
    'CD8+CD14',
    'CD56',
    'CD45RA',
    'CD366',      # Tim-3
    'CD69',
    'CD25',
    'CD4+CD33',
    'CD62L',
    'CD152',      # CTLA-4
    'Viability',
    'CD45',
    'CD45RO',
    'CD279+CD24', # PD-1 + CD24
    'CD95',
    'CD34+CD223', # CD34 + LAG-3
    'CD197',      # CCR7
    'CD3+CD19'    # PRIMARY LINEAGE
]

class PanelConfig:
    def __init__(self):
        self.input_channels = CHANNEL_NAMES
        self.input_idx = {ch: i for i, ch in enumerate(self.input_channels)}

        # Composite channels - ALL 6 need deconvolution
        self.composite_channels = {
            'CD3+CD19': {
                'markers': ['CD3', 'CD19'],
                'typical_lineage': {'CD3': 'T_cell', 'CD19': 'B_cell'},
                'mutual_exclusive': True,
            },
            'CD8+CD14': {
                'markers': ['CD8', 'CD14'],
                'typical_lineage': {'CD8': 'T_cell', 'CD14': 'Myeloid'},
                'mutual_exclusive': True,
            },
            'CD4+CD33': {
                'markers': ['CD4', 'CD33'],
                'typical_lineage': {'CD4': 'T_cell', 'CD33': 'Myeloid'},
                'mutual_exclusive': True,
            },
            'CD16+TIGIT': {
                'markers': ['CD16', 'TIGIT'],
                'typical_lineage': {'CD16': 'NK_Myeloid', 'TIGIT': 'T_NK'},
                'mutual_exclusive': False,
            },
            'CD279+CD24': {
                'markers': ['CD279', 'CD24'],
                'typical_lineage': {'CD279': 'T_cell', 'CD24': 'B_cell'},
                'mutual_exclusive': True,
            },
            'CD34+CD223': {
                'markers': ['CD34', 'CD223'],
                'typical_lineage': {'CD34': 'Stem', 'CD223': 'T_cell'},
                'mutual_exclusive': True,
            },
        }

        # Auxiliary signals to steer composite disambiguation
        self.composite_aux_map = {
            'CD3+CD19': ('lineage_is_tcell', 'lineage_is_bcell'),
            'CD8+CD14': ('subtype_is_cd8', 'lineage_is_myeloid'),
            'CD4+CD33': ('subtype_is_cd4', 'lineage_is_myeloid'),
            'CD16+TIGIT': ('lineage_is_nk', 'subtype_is_exhausted'),
            'CD279+CD24': ('subtype_is_exhausted', 'lineage_is_bcell'),
            'CD34+CD223': ('lineage_is_myeloid', 'subtype_is_naive'),
        }

        # Simple channels
        self.simple_channels = [
            'CD56', 'CD45RA', 'CD366', 'CD69', 'CD25', 'CD62L',
            'CD152', 'Viability', 'CD45', 'CD45RO', 'CD95', 'CD197'
        ]
        self.simple_indices = [self.input_idx[ch] for ch in self.simple_channels]

        # All output markers
        self.output_markers = self.simple_channels.copy()
        for comp_info in self.composite_channels.values():
            self.output_markers.extend(comp_info['markers'])
        self.output_idx = {ch: i for i, ch in enumerate(self.output_markers)}
        self.composite_marker_indices = {
            comp: [self.output_idx[m] for m in info['markers']]
            for comp, info in self.composite_channels.items()
        }
        self.viability_idx = self.input_idx['Viability']
        self.composite_min_fraction = 0.02

        self.n_input = len(self.input_channels)
        self.n_output = len(self.output_markers)
        self.lymphocyte_aux_keys = ('lineage_is_tcell', 'lineage_is_bcell', 'lineage_is_nk')
        self.rare_aux_keys = (
            'subtype_is_double_positive',
            'subtype_is_exhausted',
            'subtype_is_activated'
        )

    def build_deconvolved_targets(self, data, aux_labels):
        """Expand the 18 raw inputs into 24 disambiguated marker targets."""
        data_np = np.asarray(data, dtype=np.float32)
        n = data_np.shape[0]
        targets = np.zeros((n, self.n_output), dtype=np.float32)

        def _fetch_aux(key):
            if not key:
                return np.ones(n, dtype=np.float32)
            values = aux_labels.get(key)
            if values is None:
                return np.ones(n, dtype=np.float32)
            if isinstance(values, torch.Tensor):
                return values.detach().cpu().numpy().astype(np.float32)
            return np.asarray(values, dtype=np.float32)

        # Simple channels copy directly
        for marker in self.simple_channels:
            targets[:, self.output_idx[marker]] = data_np[:, self.input_idx[marker]]

        # Composite markers use auxiliary priors for disambiguation
        for comp_name, comp_info in self.composite_channels.items():
            markers = comp_info['markers']
            combo_signal = data_np[:, self.input_idx[comp_name]]
            aux_keys = self.composite_aux_map.get(comp_name, ())

            priors = [_fetch_aux(key) for key in aux_keys[:len(markers)]]
            if len(priors) < len(markers):
                priors.extend([np.ones(n, dtype=np.float32)] * (len(markers) - len(priors)))

            weights = np.stack(priors, axis=1)
            weights = weights + 1e-6  # avoid divide-by-zero
            denom = weights.sum(axis=1, keepdims=True)
            fractions = weights / denom

            if comp_info.get('mutual_exclusive', True):
                min_frac = getattr(self, 'composite_min_fraction', 0.02)
                min_frac = float(np.clip(min_frac, 0.0, 0.49))
                fractions = fractions.clip(min_frac, 1.0 - min_frac)

            split_values = combo_signal[:, None] * fractions
            for marker, values in zip(markers, split_values.T):
                targets[:, self.output_idx[marker]] = values

        return targets

    def print_summary(self):
        print()
        print('='*70)
        print('PANEL CONFIGURATION')
        print('='*70)
        print(f'Input channels: {self.n_input}')
        print(f'  - Simple: {len(self.simple_channels)}')
        print(f'  - Composite: {len(self.composite_channels)}')
        print(f'Output markers (deconvolved): {self.n_output}')
        print()
        print('-'*70)
        print('COMPOSITE CHANNELS:')
        for ch_name, ch_info in self.composite_channels.items():
            markers = ch_info['markers']
            print(f"  {ch_name:15s} → {markers[0]:6s} OR {markers[1]:6s}")
        print('='*70)

config = PanelConfig()
config.print_summary()



class FCSLoader:
    """
    Load and preprocess FCS files.
    """
    
    def __init__(self, config, arcsinh_cofactor=5.0):
        self.config = config
        self.arcsinh_cofactor = arcsinh_cofactor
        
    def load_fcs(self, filepath, sample_name=None):
        """Load single FCS file."""
        if sample_name is None:
            sample_name = Path(filepath).stem
            
        print(f"Loading: {sample_name}")
        
        try:
            meta, data = fcsparser.parse(filepath, reformat_meta=True)
            print(f"  → {data.shape[0]:,} events, {data.shape[1]} channels")
            
            extracted = self.extract_channels(data, meta)
            return extracted, sample_name
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None, sample_name
    
    def extract_channels(self, data, meta):
        """Extract the 18 required channels."""
        available = data.columns.tolist()
        extracted = np.zeros((len(data), self.config.n_input))
        
        for i, ch_name in enumerate(self.config.input_channels):
            if ch_name in available:
                extracted[:, i] = data[ch_name].values
            else:
                # Try fuzzy matching
                matched = self._fuzzy_match(ch_name, available)
                if matched:
                    extracted[:, i] = data[matched].values
                    print(f"  → Matched '{ch_name}' to '{matched}'")
                else:
                    print(f"  ⚠️  Missing: '{ch_name}'")
        
        return extracted
    
    def _fuzzy_match(self, target, available):
        """Fuzzy channel name matching."""
        target_clean = target.replace('+', '').replace('-', '').replace('/', '').lower()
        
        for col in available:
            col_clean = col.replace('+', '').replace('-', '').replace('/', '').lower()
            if target_clean in col_clean or col_clean in target_clean:
                return col
        return None
    
    def load_multiple(self, file_pattern, max_files=None, subsample=None):
        """
        Load multiple FCS files.
        
        Args:
            file_pattern: Glob pattern, e.g., 'data/normal*.fcs'
            max_files: Max number of files to load
            subsample: If int, randomly sample this many cells per file
        """
        files = sorted(glob.glob(file_pattern))
        
        if len(files) == 0:
            raise ValueError(f"No files found: {file_pattern}")
        
        if max_files:
            files = files[:max_files]
        
        print(f"\n{'='*70}")
        print(f"LOADING {len(files)} FCS FILES")
        print(f"{'='*70}\n")
        
        all_data = []
        sample_ids = []
        
        for filepath in tqdm(files):
            data, name = self.load_fcs(filepath)
            if data is not None:
                # Subsample if requested
                if subsample and len(data) > subsample:
                    idx = np.random.choice(len(data), subsample, replace=False)
                    data = data[idx]
                
                all_data.append(data)
                sample_ids.extend([name] * len(data))
        
        combined = np.vstack(all_data)
        
        print(f"\n{'='*70}")
        print(f"LOADING COMPLETE")
        print(f"{'='*70}")
        print(f"Total: {combined.shape[0]:,} cells from {len(all_data)} samples")
        
        return combined, sample_ids
    
    def preprocess(self, data, apply_arcsinh=True, qc_viability=True):
        """
        Preprocess data: arcsinh transform + QC.
        """
        data_processed = data.copy()
        
        # QC: Remove dead cells
        if qc_viability:
            viability = data[:, self.config.viability_idx]
            viability_norm = (viability - viability.min()) / (viability.max() - viability.min() + 1e-10)
            live_mask = viability_norm > 0.1
            
            print(f"\nQC: {live_mask.sum():,}/{len(data):,} live cells ({live_mask.sum()/len(data)*100:.1f}%)")
            data_processed = data_processed[live_mask]
        
        # Arcsinh transform
        if apply_arcsinh:
            print(f"Arcsinh transform (cofactor={self.arcsinh_cofactor})")
            data_processed = np.arcsinh(data_processed / self.arcsinh_cofactor)
        
        return data_processed

print("✅ FCS loader loaded")

from scipy.sparse import diags 
class HeatDiffusionFast:
    """Heat diffusion using PyNNDescent (crash-proof)."""
    
    def __init__(self, k=15, t=0.5, use_pynndescent=True):
        self.k = k
        self.t = t
        self.use_pynndescent = use_pynndescent
        self.L = None
        
    def fit(self, X):
        if self.k == 0:
            print("k=0: Skipping heat diffusion")
            return self
            
        t0 = time.time()
        print(f"\n🔥 Building {self.k}-NN graph for {X.shape[0]:,} cells...")
        
        # Use PyNNDescent instead of sklearn
        if self.use_pynndescent:
            from pynndescent import NNDescent
            from scipy.sparse import csr_matrix
            
            index = NNDescent(X, n_neighbors=self.k+1, metric='euclidean', n_jobs=1)
            indices, distances = index.neighbor_graph
            
            # Remove self
            indices = indices[:, 1:]
            distances = distances[:, 1:]
            
            # Build sparse matrix
            n = X.shape[0]
            row_ind = np.repeat(np.arange(n), self.k)
            col_ind = indices.flatten()
            data = distances.flatten()
            
            A = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
        else:
            # Fallback to sklearn (will crash)
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(X, self.k, mode='distance', n_jobs=1)
        
        t1 = time.time()
        print(f"   → k-NN graph built in {t1-t0:.1f}s")
        
        # Rest is same...
        A = A.tocsr()
        W = (A + A.T) / 2
        
        print(f"   → Computing Laplacian...")
        t2 = time.time()
        
        D = np.asarray(W.sum(axis=1)).flatten()
        D_inv_sqrt = diags(1.0 / np.sqrt(D + 1e-10))
        I = diags(np.ones(X.shape[0]))
        self.L = I - D_inv_sqrt @ W @ D_inv_sqrt
        
        t3 = time.time()
        print(f"   → Laplacian computed in {t3-t2:.1f}s")
        print(f"✅ Setup complete in {t3-t0:.1f}s")
        return self
    
    def transform(self, X):
        if self.k == 0 or self.L is None:
            return X
        
        t0 = time.time()
        print(f"   → Applying heat diffusion (t={self.t})...")
        X_smooth = expm_multiply(-self.t * self.L, X)
        t1 = time.time()
        print(f"✅ Diffusion applied in {t1-t0:.1f}s")
        return X_smooth
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

print("✅ PyNNDescent heat diffusion loaded")

# Cell 6 (Corrected)

class ArchetypalLayer(nn.Module):
    # ... (this class is unchanged, no need to copy it again) ...
    def __init__(self, input_dim, n_archetypes=8, archetype_dim=8):
        super().__init__()
        
        self.n_archetypes = n_archetypes
        self.archetype_dim = archetype_dim
        
        # Encoder: h1 -> archetype weights (alpha)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, n_archetypes),
        )
        
        # Learnable archetypes
        self.archetypes = nn.Parameter(
            torch.randn(n_archetypes, archetype_dim) * 0.1
        )
        
        nn.init.orthogonal_(self.archetypes.data)
        
        
    def forward(self, h):
        logits = self.encoder(h)                          # [B, K]
        tau = getattr(self, "tau", 0.5)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        alpha = F.softmax(logits / tau, dim=-1)           # [B, K], sum=1

        alpha_safe = alpha.clamp_min(1e-8)
        entropy = -(alpha_safe * alpha_safe.log()).sum(dim=-1).mean()

        A = self.archetypes.to(h.dtype).to(h.device)      # [K, D]
        z3 = alpha @ A                                    # [B, D]
        return z3, alpha, A, entropy

    
    def diversity_loss(self):
        # ... (this method is correct) ...
        archetypes_norm = F.normalize(self.archetypes, p=2, dim=1)
        similarity_matrix = archetypes_norm @ archetypes_norm.T
        mask = ~torch.eye(self.n_archetypes, device=similarity_matrix.device).bool()
        diversity_loss = similarity_matrix[mask].pow(2).mean()
        return diversity_loss

# --- The VAE Class ---
class HierarchicalVAE_WithArchetypes(nn.Module):
    def __init__(self, config,
                 latent_dims=[16, 12, 8],
                 n_archetypes=8,
                 hidden_dims=[128, 96, 64]):
        super().__init__()
        
        self.config = config
        self.output_dim = config.n_output
        self.latent_dims = latent_dims
        self.n_archetypes = n_archetypes
        self.hidden_dims = hidden_dims
        
        input_dim = config.n_input
        output_dim = config.n_output
        
        # === ENCODER Level 1: q(z1 | x) ===
        # This will produce 'h1' which we now feed to TWO places:
        # 1. The z2 encoder
        # 2. The ArchetypalLayer
        self.encoder_1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        self.fc_mu_1 = nn.Linear(hidden_dims[0], latent_dims[0])
        self.fc_logvar_1 = nn.Linear(hidden_dims[0], latent_dims[0])
        
        # === ENCODER Level 2: q(z2 | x, z1) ===
        self.encoder_2 = nn.Sequential(
            nn.Linear(hidden_dims[0] + latent_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        self.fc_mu_2 = nn.Linear(hidden_dims[1], latent_dims[1])
        self.fc_logvar_2 = nn.Linear(hidden_dims[1], latent_dims[1])
        
        # === PRIOR NETWORK: p(z2 | z1) ===
        self.prior_2 = nn.Sequential(
            nn.Linear(latent_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.2),
        )
        self.prior_mu_2 = nn.Linear(hidden_dims[1], latent_dims[1])
        self.prior_logvar_2 = nn.Linear(hidden_dims[1], latent_dims[1])
        
        
        # === ARCHETYPAL LAYER: h1 -> z3 ===
        self.archetypal_layer = ArchetypalLayer(
            input_dim=hidden_dims[0],  # <--- MODIFIED: Takes h1 (dim=128), not z2 (dim=12)
            n_archetypes=n_archetypes,
            archetype_dim=latent_dims[2] # This is the output dim (z3)
        )
        
        # === DECODER: z3 -> x_recon ===
        # (This is unchanged from your notebook, z3 is the only input)
        total_latent = latent_dims[2] 
        
        self.decoder = nn.Sequential(
            nn.Linear(total_latent, hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], output_dim)
        )
        
        # === AUXILIARY CLASSIFIERS ===
        # On z1 (Viability)
        self.aux_viability = nn.Linear(latent_dims[0], 1)
        
        # On z2 (Lineage)
        self.aux_lineage = nn.ModuleDict({
            'lineage_is_tcell': nn.Linear(latent_dims[1], 1),
            'lineage_is_bcell': nn.Linear(latent_dims[1], 1),
            'lineage_is_myeloid': nn.Linear(latent_dims[1], 1),
            'lineage_is_nk': nn.Linear(latent_dims[1], 1),
        })
        
        # <--- NEW: AUXILIARY SUBTYPE CLASSIFIERS (on z3) --->
        # This is the key to teaching the archetypes about T-cell states
        self.aux_subtypes = nn.ModuleDict({
            'subtype_is_cd4': nn.Linear(latent_dims[2], 1),
            'subtype_is_cd8': nn.Linear(latent_dims[2], 1),
            'subtype_is_double_positive': nn.Linear(latent_dims[2], 1),
            'subtype_is_naive': nn.Linear(latent_dims[2], 1),
            'subtype_is_central_memory': nn.Linear(latent_dims[2], 1),
            'subtype_is_effector_memory': nn.Linear(latent_dims[2], 1),
            'subtype_is_activated': nn.Linear(latent_dims[2], 1),
            'subtype_is_exhausted': nn.Linear(latent_dims[2], 1),
        })
        
        # Initialize
        self.apply(self._init_weights)
        nn.init.constant_(self.fc_logvar_1.bias, -2.0)
        nn.init.constant_(self.fc_logvar_2.bias, -2.0)
        nn.init.constant_(self.prior_logvar_2.bias, -2.0) 

    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        # q(z1 | x)
        h1 = self.encoder_1(x) # <--- h1 is the key hidden state
        mu1 = self.fc_mu_1(h1)
        logvar1 = self.fc_logvar_1(h1)
        z1 = self.reparameterize(mu1, logvar1)
        
        # q(z2 | x, z1)
        h2_input = torch.cat([h1, z1], dim=1)
        h2 = self.encoder_2(h2_input) 
        mu2 = self.fc_mu_2(h2)
        logvar2 = self.fc_logvar_2(h2)
        z2 = self.reparameterize(mu2, logvar2)
        
        # z3
        # <--- MODIFIED: z3 is now derived from h1, in parallel to z2
        z3, alpha, archetypes, entropy = self.archetypal_layer(h1) 
        
        return (mu1, logvar1, z1), (mu2, logvar2, z2), (z3, alpha, archetypes, entropy)
    
    def forward(self, x):
        # Encode (Inference)
        (mu1, logvar1, z1), (mu2, logvar2, z2), (z3, alpha, archetypes, entropy) = self.encode(x)
        
        # Get Prior Parameters for z2 (p(z2 | z1))
        prior_h2 = self.prior_2(z1)
        prior_mu2 = self.prior_mu_2(prior_h2)
        prior_logvar2 = self.prior_logvar_2(prior_h2)
        
        # Decode (Unchanged from notebook, still z3 -> recon)
        z_combined = z3
        recon = self.decoder(z_combined)
        
        # Auxiliary predictions
        aux_preds = {}
        aux_preds['viability'] = self.aux_viability(z1) # From z1
        
        for name, classifier in self.aux_lineage.items():
            aux_preds[name] = classifier(z2) # From z2
        
        # <--- NEW: Add subtype predictions from z3 --->
        for name, classifier in self.aux_subtypes.items():
            aux_preds[name] = classifier(z3) # From z3
        
        return recon, (mu1, logvar1, mu2, logvar2, prior_mu2, prior_logvar2), alpha, archetypes, entropy, aux_preds

# Cell 7 (Complete and Corrected)

from IPython.display import clear_output, display 

# Helper function for KL(q || p)
def kl_divergence_gaussians(q_mu, q_logvar, p_mu, p_logvar):
    # ... (this function is unchanged, no need to copy it again) ...
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar) + 1e-10 
    
    kl = 0.5 * (
        p_logvar - q_logvar + 
        (q_var + (q_mu - p_mu).pow(2)) / p_var - 
        1.0
    ).sum(dim=1)
    
    return kl


def train_hierarchical_vae_with_archetypes(
        model,
        train_loader,
        val_loader,
        n_epochs=50,
        lr=1e-3,
        device='cpu',
        save_dir='./checkpoints',
        experiment_name='exp',
        save_epochs=[10, 20, 30, 40, 50],
        entropy_cap=None,
        tau_init=0.5,
        tau_final=None,
        tau_decay_start=None,
        tau_decay_end=None,
        combo_weight=0.5,
        batch_entropy_weight=0.1,
        kl_warmup_epochs=10,
        kl_ramp_epochs=30,
        purity_warmup_epochs=20,
        purity_ramp_epochs=30,
        cell_entropy_max=0.1,
        aux_lineage_weight=0.2,
        aux_subtype_weight=1.0,
        purity_margin=0.65,
        purity_lambda=0.7,
        stage_epoch=None,
        stage_cell_entropy_scale=1.5,
        stage_purity_lambda_scale=1.5,
        live_registry=None,
        run_name=None,
        trial_epochs=None,
        early_stop_checker=None):
    model = model.to(device)
    if tau_init is not None:
        model.archetypal_layer.tau = tau_init
    max_epochs = n_epochs if trial_epochs is None else min(n_epochs, trial_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs if max_epochs > 0 else 1)
    
    save_dir = Path(save_dir) / experiment_name
    save_dir.mkdir(exist_ok=True, parents=True)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_recon': [], 'train_kl1': [], 'train_kl2': [],
        'train_entropy': [], 'train_diversity': [], 'train_batch_entropy': [],
        'train_combo': [], 'train_aux': [], 'train_purity_loss': [],
        'val_combo': [], 'val_purity': [], 'val_nk_usage': [], 'val_purity_loss': []
    }
    
    run_label = run_name or experiment_name
    best_val_loss = float('inf')
    completed_epochs = 0
    
    print(f"\nTraining: {experiment_name}")
    print(f"Save dir: {save_dir}")
    print(f"\n{'='*70}")
    print(f"Training Hierarchical VAE with Archetypal Layer")
    print(f"Archetypes: {model.n_archetypes}")
    print(f"{'='*70}\n")

    diversity_weight = 1.0 
    stage_epoch = stage_epoch if stage_epoch is not None else max(1, int(0.6 * max_epochs))
    stage_epoch = min(stage_epoch, max_epochs - 1)

    tau_decay_start = tau_decay_start if tau_decay_start is not None else 0
    tau_decay_end = tau_decay_end if tau_decay_end is not None else max_epochs
    tau_range = max(1, tau_decay_end - tau_decay_start)
    
    for epoch in range(max_epochs):
        # --- KL ANNEALING ---
        beta_max = 1.0   
        current_beta = max(0.0, min(beta_max, (epoch - kl_warmup_epochs) / max(1, kl_ramp_epochs)))
        
        beta1 = current_beta * 0.1 # z1 (viability) - less important KL
        beta2 = current_beta * 1.0 # z2 (lineage) - more important KL
        
        # --- DYNAMIC ENTROPY WEIGHTS ---
        if epoch <= purity_warmup_epochs:
            cell_entropy_weight = 0.0
        else:
            progress = min(1.0, (epoch - purity_warmup_epochs) / max(1, purity_ramp_epochs))
            cell_entropy_weight = cell_entropy_max * progress

        if epoch >= stage_epoch:
            cell_entropy_weight = max(cell_entropy_weight, cell_entropy_max)
            cell_entropy_weight = min(cell_entropy_weight * stage_cell_entropy_scale, cell_entropy_max * stage_cell_entropy_scale)
            purity_lambda_epoch = purity_lambda * stage_purity_lambda_scale
        else:
            purity_lambda_epoch = purity_lambda

        # --- TAU SCHEDULING ---
        if tau_final is not None and epoch >= tau_decay_start:
            progress = min(1.0, (epoch - tau_decay_start) / tau_range)
            target_tau = tau_init + (tau_final - tau_init) * progress
            model.archetypal_layer.tau = float(max(target_tau, 1e-3))
            if epoch >= stage_epoch:
                model.archetypal_layer.tau = float(max(min(model.archetypal_layer.tau, tau_final if tau_final is not None else model.archetypal_layer.tau), 1e-3))


        
        # --- Training Loop ---
        model.train()
        train_metrics = {
            'loss': 0.0, 'recon': 0.0, 'kl1': 0.0, 'kl2': 0.0, 'aux': 0.0,
            'entropy': 0.0, 'diversity': 0.0, 'batch_entropy': 0.0, 'combo': 0.0,
            'purity': 0.0
        }
        train_weight = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
        for batch in pbar:
            x = batch['data'].to(device)
            
            if torch.isnan(x).any() or torch.isinf(x).any():
                continue
            
            try:
                recon, (mu1, logvar1, mu2, logvar2, prior_mu2, prior_logvar2), alpha, archetypes, entropy, aux_preds = model(x)
                
                if torch.isnan(recon).any():
                    continue
                
                targets = batch.get('recon_target', batch['data']).to(device)
                sample_weight = batch.get('sample_weight')
                if sample_weight is None:
                    weights = torch.ones(len(x), device=device)
                else:
                    weights = sample_weight.to(device)
                weight_norm = weights.sum().clamp_min(1e-6)
                
                batch_weight = float(weight_norm.item())

                # 1. Reconstruction loss
                recon_error = (recon - targets) ** 2
                recon_loss = (recon_error * weights.unsqueeze(1)).sum() / weight_norm
                
                # 2. KL divergence
                logvar1_c = logvar1.clamp(-10, 10)
                kl1_per = -0.5 * (1 + logvar1_c - mu1.pow(2) - logvar1_c.exp()).sum(dim=1)
                kl1 = (kl1_per * weights).sum() / weight_norm

                mu2_c = mu2.clamp(-10, 10)
                logvar2_c = logvar2.clamp(-10, 10)
                prior_mu2_c = prior_mu2.clamp(-10, 10)
                prior_logvar2_c = prior_logvar2.clamp(-10, 10)
                kl2_per = kl_divergence_gaussians(mu2_c, logvar2_c, prior_mu2_c, prior_logvar2_c)
                kl2 = (kl2_per * weights).sum() / weight_norm
                kl_loss = beta1 * kl1 + beta2 * kl2
                
                # 3. Cell Purity Loss (MINIMIZE cell entropy)
                cell_entropy_per = -(alpha * torch.log(alpha + 1e-10)).sum(dim=1)
                cell_entropy = (cell_entropy_per * weights).sum() / weight_norm
                if entropy_cap is not None:
                    cell_entropy = torch.clamp(cell_entropy, max=entropy_cap)
                cell_entropy_loss = cell_entropy_weight * cell_entropy

                max_alpha = alpha.max(dim=1).values
                purity_hinge = torch.clamp(purity_margin - max_alpha, min=0.0)
                purity_penalty = (purity_hinge * weights).sum() / weight_norm
                purity_loss = purity_lambda_epoch * purity_penalty
                
                # 4. Diversity loss
                diversity_loss = model.archetypal_layer.diversity_loss()
                
                # 5. Batch Usage Loss (MAXIMIZE batch entropy)
                mean_alpha_over_batch = (alpha * weights.unsqueeze(1)).sum(dim=0) / weight_norm
                batch_entropy = -(mean_alpha_over_batch * torch.log(mean_alpha_over_batch + 1e-10)).sum()
                batch_entropy_loss = -batch_entropy_weight * batch_entropy
                
                # 6. Composite consistency (align summed markers with original combos)
                combo_losses = []
                for comp_name, marker_indices in config.composite_marker_indices.items():
                    idx_a, idx_b = marker_indices
                    combo_idx = config.input_idx[comp_name]
                    pred_combo = recon[:, idx_a] + recon[:, idx_b]
                    true_combo = x[:, combo_idx]
                    combo_losses.append((pred_combo - true_combo) ** 2)
                if combo_losses:
                    combo_tensor = torch.stack(combo_losses, dim=1)
                    combo_consistency_loss = (combo_tensor * weights.unsqueeze(1)).sum() / (weight_norm * len(combo_losses))
                else:
                    combo_consistency_loss = torch.tensor(0.0, device=device)
                
                # 7. Auxiliary losses (with masking)
                aux_loss = torch.tensor(0.0, device=device)
                tcell_mask = batch['lineage_is_tcell'].to(device).bool().squeeze()

                for task_name, aux_pred in aux_preds.items():
                    if task_name in batch:
                        aux_label = batch[task_name].to(device).unsqueeze(1)
                        
                        if 'subtype_' in task_name:
                            if tcell_mask.sum() > 0:
                                aux_loss = aux_loss + aux_subtype_weight * F.binary_cross_entropy_with_logits(
                                    aux_pred[tcell_mask], 
                                    aux_label[tcell_mask]
                                )
                        elif 'lineage_' in task_name or 'viability' in task_name:
                            aux_loss = aux_loss + aux_lineage_weight * F.binary_cross_entropy_with_logits(aux_pred, aux_label)
                
                # Total loss
                loss = (
                    recon_loss + 
                    kl_loss + 
                    cell_entropy_loss + 
                    purity_loss + 
                    batch_entropy_loss + 
                    diversity_weight * diversity_loss + 
                    combo_weight * combo_consistency_loss + 
                    aux_loss
                )
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_metrics['loss'] += loss.item() * batch_weight
                train_metrics['recon'] += recon_loss.item() * batch_weight
                train_metrics['kl1'] += kl1.item() * batch_weight
                train_metrics['kl2'] += kl2.item() * batch_weight
                train_metrics['aux'] += aux_loss.item() * batch_weight
                train_metrics['entropy'] += cell_entropy.item() * batch_weight
                train_metrics['diversity'] += diversity_loss.item() * batch_weight
                train_metrics['batch_entropy'] += batch_entropy.item() * batch_weight
                train_metrics['combo'] += combo_consistency_loss.item() * batch_weight
                train_metrics['purity'] += purity_loss.item() * batch_weight
                train_weight += batch_weight
                avg_aux = train_metrics['aux'] / max(train_weight, 1e-6)
                pbar.set_postfix(Loss=loss.item(), KL1=kl1.item(), KL2=kl2.item(), Combo=combo_consistency_loss.item(), Aux=avg_aux)

            except RuntimeError as e:
                print(f"⚠️  Runtime error: {e}, skipping batch")
                continue
        
        # Average metrics
        if train_weight > 0:
            for k in train_metrics:
                train_metrics[k] /= train_weight
        
        # --- Validation ---
        model.eval()
        val_metrics = {'loss': 0.0, 'recon': 0.0, 'kl1': 0.0, 'kl2': 0.0, 'aux': 0.0, 'combo': 0.0, 'purity': 0.0}
        val_weight = 0.0
        epoch_metrics = {'purity_sum': 0.0, 'combo_sum': 0.0, 'weight': 0.0, 'nk_sum': 0.0, 'nk_weight': 0.0}
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['data'].to(device)
                if torch.isnan(x).any():
                    continue
                
                try:
                    recon, (mu1, logvar1, mu2, logvar2, prior_mu2, prior_logvar2), alpha, archetypes, entropy, aux_preds = model(x)
                    
                    if torch.isnan(recon).any():
                        continue
                    
                    targets = batch.get('recon_target', batch['data']).to(device)
                    sample_weight = batch.get('sample_weight')
                    if sample_weight is None:
                        weights = torch.ones(len(x), device=device)
                    else:
                        weights = sample_weight.to(device)
                    weight_norm = weights.sum().clamp_min(1e-6)
                    
                    batch_weight = float(weight_norm.item())

                    recon_error = (recon - targets) ** 2
                    recon_loss = (recon_error * weights.unsqueeze(1)).sum() / weight_norm
                    
                    logvar1_c = logvar1.clamp(-10, 10)
                    kl1_per = -0.5 * (1 + logvar1_c - mu1.pow(2) - logvar1_c.exp()).sum(dim=1)
                    kl1 = (kl1_per * weights).sum() / weight_norm
                    
                    mu2_c = mu2.clamp(-10, 10)
                    logvar2_c = logvar2.clamp(-10, 10)
                    prior_mu2_c = prior_mu2.clamp(-10, 10)
                    prior_logvar2_c = prior_logvar2.clamp(-10, 10)
                    kl2_per = kl_divergence_gaussians(mu2_c, logvar2_c, prior_mu2_c, prior_logvar2_c)
                    kl2 = (kl2_per * weights).sum() / weight_norm
                    kl_loss = beta1 * kl1 + beta2 * kl2
                    
                    cell_entropy_per = -(alpha * torch.log(alpha + 1e-10)).sum(dim=1)
                    cell_entropy = (cell_entropy_per * weights).sum() / weight_norm
                    if entropy_cap is not None:
                        cell_entropy = torch.clamp(cell_entropy, max=entropy_cap)
                    cell_entropy_loss = cell_entropy_weight * cell_entropy
                    purity_hinge = torch.clamp(purity_margin - alpha.max(dim=1).values, min=0.0)
                    purity_penalty = (purity_hinge * weights).sum() / weight_norm
                    purity_loss = purity_lambda_epoch * purity_penalty

                    diversity_loss = model.archetypal_layer.diversity_loss()
                    
                    mean_alpha_over_batch = (alpha * weights.unsqueeze(1)).sum(dim=0) / weight_norm
                    batch_entropy = -(mean_alpha_over_batch * torch.log(mean_alpha_over_batch + 1e-10)).sum()
                    batch_entropy_loss = -batch_entropy_weight * batch_entropy
                    
                    combo_losses = []
                    for comp_name, marker_indices in config.composite_marker_indices.items():
                        idx_a, idx_b = marker_indices
                        combo_idx = config.input_idx[comp_name]
                        pred_combo = recon[:, idx_a] + recon[:, idx_b]
                        true_combo = x[:, combo_idx]
                        combo_losses.append((pred_combo - true_combo) ** 2)
                    if combo_losses:
                        combo_tensor = torch.stack(combo_losses, dim=1)
                        combo_consistency_loss = (combo_tensor * weights.unsqueeze(1)).sum() / (weight_norm * len(combo_losses))
                    else:
                        combo_consistency_loss = torch.tensor(0.0, device=device)
                    
                    aux_loss = torch.tensor(0.0, device=device)
                    tcell_mask = batch['lineage_is_tcell'].to(device).bool().squeeze()
                    for task_name, aux_pred in aux_preds.items():
                        if task_name in batch:
                            aux_label = batch[task_name].to(device).unsqueeze(1)
                            if 'subtype_' in task_name:
                                if tcell_mask.sum() > 0:
                                    aux_loss = aux_loss + aux_subtype_weight * F.binary_cross_entropy_with_logits(
                                        aux_pred[tcell_mask], 
                                        aux_label[tcell_mask]
                                    )
                            elif 'lineage_' in task_name or 'viability' in task_name:
                                aux_loss = aux_loss + aux_lineage_weight * F.binary_cross_entropy_with_logits(aux_pred, aux_label)
                    
                    loss = (
                        recon_loss + 
                        kl_loss + 
                        cell_entropy_loss + 
                        purity_loss + 
                        batch_entropy_loss + 
                        diversity_weight * diversity_loss + 
                        combo_weight * combo_consistency_loss + 
                        aux_loss
                    )
                    
                    val_metrics['loss'] += loss.item() * batch_weight
                    val_metrics['recon'] += recon_loss.item() * batch_weight
                    val_metrics['kl1'] += kl1.item() * batch_weight
                    val_metrics['kl2'] += kl2.item() * batch_weight
                    val_metrics['aux'] += aux_loss.item() * batch_weight
                    val_metrics['combo'] += combo_consistency_loss.item() * batch_weight
                    val_metrics['purity'] += purity_loss.item() * batch_weight
                    val_weight += batch_weight

                    purity_vals = alpha.max(dim=1).values
                    epoch_metrics['purity_sum'] += (purity_vals * weights).sum().item()
                    epoch_metrics['combo_sum'] += combo_consistency_loss.item() * batch_weight
                    epoch_metrics['weight'] += batch_weight

                    nk_labels = batch.get('lineage_is_nk')
                    if nk_labels is not None:
                        nk_mask = nk_labels.to(device).squeeze() > 0.5
                        if nk_mask.any():
                            nk_weights = weights[nk_mask]
                            epoch_metrics['nk_sum'] += (purity_vals[nk_mask] * nk_weights).sum().item()
                            epoch_metrics['nk_weight'] += nk_weights.sum().item()
                    
                except RuntimeError:
                    continue
            
        if val_weight > 0:
            for k in val_metrics:
                val_metrics[k] /= val_weight
        else:
            for k in val_metrics:
                val_metrics[k] = 0.0
        
        epoch_summary = {
            'purity': epoch_metrics['purity_sum'] / max(epoch_metrics['weight'], 1e-6),
            'combo': epoch_metrics['combo_sum'] / max(epoch_metrics['weight'], 1e-6) if epoch_metrics['weight'] > 0 else 0.0,
            'nk_usage': epoch_metrics['nk_sum'] / max(epoch_metrics['nk_weight'], 1e-6) if epoch_metrics['nk_weight'] > 0 else 0.0
        }
        val_metrics['combo'] = epoch_summary['combo']

        # Record
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_recon'].append(train_metrics['recon'])
        history['train_kl1'].append(train_metrics['kl1'])
        history['train_kl2'].append(train_metrics['kl2'])
        history['train_aux'].append(train_metrics['aux'])
        history['train_entropy'].append(train_metrics['entropy'])
        history['train_combo'].append(train_metrics['combo'])
        history['train_purity_loss'].append(train_metrics['purity'])
        history['val_combo'].append(epoch_summary['combo'])
        history['val_purity'].append(epoch_summary['purity'])
        history['val_nk_usage'].append(epoch_summary['nk_usage'])
        history['val_purity_loss'].append(val_metrics['purity'])
        history['train_diversity'].append(train_metrics['diversity'])
        history['train_batch_entropy'].append(train_metrics['batch_entropy'])

        if live_registry is not None:
            live_registry.update(run_label, {
                'epoch': epoch + 1,
                'loss': float(history['train_loss'][-1]),
                'val_loss': float(history['val_loss'][-1]),
                'purity': float(epoch_summary['purity']),
                'nk_usage': float(epoch_summary['nk_usage']),
                'combo': float(epoch_summary['combo']),
                'purity_loss': float(val_metrics['purity']),
                'tau': float(getattr(model.archetypal_layer, 'tau', tau_init if tau_init is not None else 0.0))
            })

        completed_epochs = epoch + 1

        # --- Print ---
        if (epoch + 1) % 5 == 0 or epoch == 0:
            clear_output(wait=True) 
            
            print(f"Epoch {epoch+1:2d}/{max_epochs} | "
                  f"Loss: {train_metrics['loss']:.4f} → {val_metrics['loss']:.4f} | "
                  f"β=[{beta1:.2f}, {beta2:.2f}] | "
                  f"KL=[{train_metrics['kl1']:.3f}, {train_metrics['kl2']:.3f}] | "
                  f"Combo={epoch_summary['combo']:.3f} | "
                  f"Purity={epoch_summary['purity']:.3f} | "
                  f"PurLoss={val_metrics['purity']:.3f} | "
                  f"NK={epoch_summary['nk_usage']:.3f} | "
                  f"Tau={getattr(model.archetypal_layer, 'tau', 0.0):.3f} | "
                  f"Entropy={train_metrics['entropy']:.3f} (cap={entropy_cap if entropy_cap is not None else '∞'})")
            
            # Call the diagnostic plot function
            generate_diagnostic_plots(
                model, 
                val_loader, 
                config,
                device, 
                epoch + 1, 
                save_dir,
                history,
                live=None if live_registry is None else True
            )
        
        stop_reason = None
        if early_stop_checker is not None:
            try:
                decision = early_stop_checker(epoch + 1, epoch_summary, history)
            except TypeError:
                decision = early_stop_checker(epoch + 1, epoch_summary)
            if decision:
                stop_reason = decision if isinstance(decision, str) else "early-stop criterion met"
        
        # Save checkpoints
        if (epoch + 1) in save_epochs and not np.isnan(val_metrics['loss']):
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'val_loss': val_metrics['loss'],
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
            if (epoch + 1) % 5 != 0: 
                print(f"  ✓ Saved")
        
        # Save best
        if not np.isnan(val_metrics['loss']) and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, save_dir / 'best_model.pt')
        
        if stop_reason:
            print(f"⚠️ Early stop triggered at epoch {epoch+1}: {stop_reason}")
            break
        
        scheduler.step()

    # Save final
    torch.save({
        'epoch': completed_epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, save_dir / 'final_model.pt')
    
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete after {completed_epochs} epoch(s)! Best val: {best_val_loss:.4f}")

    return model, history

from scipy.spatial.distance import cdist
from scipy.optimize import nnls

def find_archetypes(X, n_archetypes=5, max_iter=50):
    """
    Find archetypes (extreme phenotypes) using alternating optimization.
    
    Returns:
        archetypes: (n_archetypes, n_features)
        weights: (n_cells, n_archetypes) - convex combination weights
    """
    from sklearn.preprocessing import StandardScaler
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize with furthest points
    archetypes_idx = [0]
    for _ in range(n_archetypes - 1):
        dists = cdist(X_scaled, X_scaled[archetypes_idx], metric='euclidean')
        furthest = dists.min(axis=1).argmax()
        archetypes_idx.append(furthest)
    
    archetypes = X_scaled[archetypes_idx]
    
    # Refine
    for iteration in range(max_iter):
        # Find weights for each cell
        weights = np.zeros((len(X_scaled), n_archetypes))
        for i in range(len(X_scaled)):
            w, _ = nnls(archetypes.T, X_scaled[i])
            w = w / (w.sum() + 1e-10)
            weights[i] = w
        
        # Update archetypes
        archetypes_new = weights.T @ X_scaled
        archetypes_new = archetypes_new / (np.linalg.norm(archetypes_new, axis=1, keepdims=True) + 1e-10)
        
        if np.linalg.norm(archetypes_new - archetypes) < 1e-4:
            break
        archetypes = archetypes_new
    
    archetypes_original = scaler.inverse_transform(archetypes)
    
    return archetypes_original, weights


def plot_archetypes(results, n_archetypes=5):
    """
    Visualize archetypes in UMAP space.
    """
    # Find archetypes in z2 (lineage)
    archetypes_z2, weights_z2 = find_archetypes(results['z2'], n_archetypes=n_archetypes)
    
    # Assign labels based on dominant archetype
    labels = weights_z2.argmax(axis=1)
    
    # Also track mixed phenotypes (cells with >1 strong archetype)
    n_strong = (weights_z2 > 0.3).sum(axis=1)
    mixed_mask = n_strong > 1
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot pure archetypes
    for i in range(n_archetypes):
        mask = (labels == i) & (~mixed_mask)
        ax.scatter(results['umap2'][mask, 0], results['umap2'][mask, 1],
                  s=1, alpha=0.3, label=f'Archetype {i}')
    
    # Plot mixed phenotypes
    ax.scatter(results['umap2'][mixed_mask, 0], results['umap2'][mixed_mask, 1],
              c='gray', s=1, alpha=0.5, label='Mixed')
    
    # Mark archetype locations (find nearest cells)
    archetype_coords = []
    for arch in archetypes_z2:
        dists = cdist([arch], results['z2'])[0]
        nearest = dists.argmin()
        archetype_coords.append(results['umap2'][nearest])
    archetype_coords = np.array(archetype_coords)
    
    ax.scatter(archetype_coords[:, 0], archetype_coords[:, 1],
              marker='*', s=500, c='black', edgecolors='white', linewidths=2,
              label='Archetypes', zorder=10)
    
    ax.set_title('Archetypal Analysis - Lineage Space (z2)', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('archetypes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print archetype statistics
    print("\nArchetype Statistics:")
    print("="*50)
    for i in range(n_archetypes):
        pure = ((labels == i) & (~mixed_mask)).sum()
        total = (labels == i).sum()  
        print(f"Archetype {i}: {pure:,} pure ({pure/len(labels)*100:.1f}%), "
              f"{total:,} total ({total/len(labels)*100:.1f}%)")
    print(f"\nMixed phenotypes: {mixed_mask.sum():,} ({mixed_mask.sum()/len(labels)*100:.1f}%)")
    
    return archetypes_z2, weights_z2

# Use it:
# archetypes, weights = plot_archetypes(results, n_archetypes=5)

def analyze_archetypes(model, data_loader, device='cpu', n_cells=10000):
    """
    Analyze learned archetypes.
    
    Returns:
        - archetypes: The learned archetype vectors
        - alpha_matrix: Archetype weights for each cell
        - dominant_archetype: Which archetype each cell is closest to
        - purity: How "pure" each cell is (max weight)
    """
    model.eval()
    
    all_alpha = []
    all_z1, all_z2, all_z3 = [], [], []
    all_data = []
    
    n_collected = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if n_collected >= n_cells:
                break
            
            x = batch['data'].to(device)
            (mu1, logvar1, z1), (mu2, logvar2, z2), (z3, alpha, archetypes, entropy) = model.encode(x)
            
            all_z1.append(z1.cpu().numpy())
            all_z2.append(z2.cpu().numpy())
            all_z3.append(z3.cpu().numpy())
            all_alpha.append(alpha.cpu().numpy())
            all_data.append(x.cpu().numpy())
            
            n_collected += len(x)
    
    z1 = np.vstack(all_z1)[:n_cells]
    z2 = np.vstack(all_z2)[:n_cells]
    z3 = np.vstack(all_z3)[:n_cells]
    alpha_matrix = np.vstack(all_alpha)[:n_cells]
    data = np.vstack(all_data)[:n_cells]
    
    # Get learned archetypes
    archetypes = model.archetypal_layer.archetypes.detach().cpu().numpy()
    
    # Dominant archetype for each cell
    dominant_archetype = alpha_matrix.argmax(axis=1)
    
    # Purity (how strongly cell commits to single archetype)
    purity = alpha_matrix.max(axis=1)
    
    # Mixed cells (use multiple archetypes)
    n_active_archetypes = (alpha_matrix > 0.2).sum(axis=1)
    mixed_mask = n_active_archetypes > 1
    
    print(f"\n{'='*70}")
    print("ARCHETYPE ANALYSIS")
    print(f"{'='*70}")
    print(f"Number of archetypes: {model.n_archetypes}")
    print(f"Total cells analyzed: {len(alpha_matrix):,}")
    print(f"\nArchetype usage:")
    for i in range(model.n_archetypes):
        n_dominant = (dominant_archetype == i).sum()
        n_active = (alpha_matrix[:, i] > 0.2).sum()
        print(f"  Archetype {i}: {n_dominant:,} dominant ({n_dominant/len(alpha_matrix)*100:.1f}%), "
              f"{n_active:,} active ({n_active/len(alpha_matrix)*100:.1f}%)")
    
    print(f"\nCell purity:")
    print(f"  Mean purity: {purity.mean():.3f}")
    print(f"  Pure cells (>0.8): {(purity > 0.8).sum():,} ({(purity > 0.8).sum()/len(purity)*100:.1f}%)")
    print(f"  Mixed cells (using >1 archetype): {mixed_mask.sum():,} ({mixed_mask.sum()/len(mixed_mask)*100:.1f}%)")
    
    return {
        'archetypes': archetypes,
        'alpha_matrix': alpha_matrix,
        'dominant_archetype': dominant_archetype,
        'purity': purity,
        'mixed_mask': mixed_mask,
        'z1': z1, 'z2': z2, 'z3': z3,
        'data': data
    }


if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║     HIERARCHICAL VAE WITH INTEGRATED ARCHETYPAL ANALYSIS           ║
    ║                                                                    ║
    ║  ✅ z1: Viability/debris (standard VAE, β=3.0)                    ║
    ║  ✅ z2: Lineage (standard VAE, β=0.8)                             ║
    ║  ✅ z3: ARCHETYPES (AANet-style, learned directly)                ║
    ║                                                                    ║
    ║  Key advantages:                                                   ║
    ║  • Archetypes learned end-to-end with model                        ║
    ║  • No post-hoc archetypal analysis needed                          ║
    ║  • Each cell = convex combination of archetypes                    ║
    ║  • More interpretable by design                                    ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    Usage:
    ------
    1. Create model:
       >>> model = HierarchicalVAE_WithArchetypes(
       ...     config, 
       ...     latent_dims=[16, 12, 8],
       ...     n_archetypes=8
       ... )
    
    2. Train:
       >>> history = train_hierarchical_vae_with_archetypes(
       ...     model, train_loader, val_loader, n_epochs=50
       ... )
    
    3. Analyze archetypes:
       >>> results = analyze_archetypes(model, data_loader)
       >>> print(f"Learned {model.n_archetypes} archetypes")
       >>> print(f"Mixed phenotypes: {results['mixed_mask'].sum()} cells")
    """)


class ExperimentSweepAdvanced:
    """
    Advanced sweep orchestrator that explores diffusion, architecture, entropy caps,
    and temperature initialisation while streaming metrics to the live registry.
    """

    def __init__(self, config, base_save_dir='./experiments',
                 k_neighbors=(0, 5, 10, 15, 20, 30),
                 latent_grid=None,
                 entropy_caps=(None,),
                 tau_inits=(0.5,),
                 tau_finals=(None,),
                 tau_decay_starts=(0,),
                 tau_decay_ends=(None,),
                 archetype_counts=None,
                 combo_weights=(0.5,),
                 batch_entropy_weights=(0.1,),
                 kl_warmups=(10,),
                 kl_ramps=(30,),
                 purity_warmups=(20,),
                 purity_ramps=(30,),
                 cell_entropy_maxes=(0.1,),
                 learning_rates=(1e-3,),
                 batch_sizes=(512,),
                 diffusion_times=(0.5,),
                 sample_weight_scales=(1.0,),
                 composite_min_fractions=(0.02,),
                 aux_lineage_weights=(0.2,),
                 aux_subtype_weights=(1.0,),
                 val_fraction=0.2,
                 trial_epochs=None,
                 purity_min=0.85,
                 nk_usage_min=0.35,
                 combo_max=0.05,
                 check_epoch=40,
                 live_registry=None,
                 random_seed=42):
        self.config = config
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(exist_ok=True, parents=True)

        if latent_grid is None:
            latent_grid = {
                'small': [8, 6, 4],
                'medium': [16, 12, 8],
                'large': [24, 18, 12],
                'xlarge': [32, 24, 16],
            }

        self.latent_grid = latent_grid
        self.k_neighbors = list(k_neighbors)
        self.entropy_caps = list(entropy_caps)
        self.tau_inits = list(tau_inits)
        self.tau_finals = list(tau_finals)
        self.tau_decay_starts = list(tau_decay_starts)
        self.tau_decay_ends = list(tau_decay_ends)
        if archetype_counts is None:
            archetype_counts = list(range(6, 25, 3))
        self.archetype_counts = sorted(archetype_counts)
        self.combo_weights = list(combo_weights)
        self.batch_entropy_weights = list(batch_entropy_weights)
        self.kl_warmups = list(kl_warmups)
        self.kl_ramps = list(kl_ramps)
        self.purity_warmups = list(purity_warmups)
        self.purity_ramps = list(purity_ramps)
        self.cell_entropy_maxes = list(cell_entropy_maxes)
        self.learning_rates = list(learning_rates)
        self.batch_sizes = list(batch_sizes)
        self.diffusion_times = list(diffusion_times)
        self.sample_weight_scales = list(sample_weight_scales)
        self.composite_min_fractions = list(composite_min_fractions)
        self.aux_lineage_weights = list(aux_lineage_weights)
        self.aux_subtype_weights = list(aux_subtype_weights)
        self.val_fraction = val_fraction
        self.trial_epochs = trial_epochs
        self.early_stop_rules = {
            'purity_min': purity_min,
            'nk_usage_min': nk_usage_min,
            'combo_max': combo_max,
            'check_epoch': check_epoch,
        }
        self.live_trainer = LiveTrainer(config, live_registry)
        self.results = []
        self._rng = np.random.default_rng(random_seed)
        self._current_sample_weight_scale = 1.0

    def _build_experiment_name(self, arch_name, k_neighbors, entropy_cap, tau_init, n_archetypes):
        entropy_tag = 'free' if entropy_cap is None else f"ec{entropy_cap:.2f}"
        tau_tag = f"tau{tau_init:.2f}"
        return f"k{k_neighbors}_{arch_name}_a{n_archetypes}_{entropy_tag}_{tau_tag}"

    def _split_data(self, train_data, val_data=None):
        train_array = np.asarray(train_data, dtype=np.float32)
        if val_data is not None:
            val_array = np.asarray(val_data, dtype=np.float32)
            return train_array.copy(), val_array.copy()

        n_total = len(train_array)
        if n_total == 0:
            raise ValueError("Training data must contain at least one cell")
        if self.val_fraction <= 0 or n_total == 1:
            return train_array.copy(), train_array.copy()

        n_val = int(n_total * self.val_fraction)
        n_val = min(max(n_val, 1), n_total - 1)
        indices = self._rng.permutation(n_total)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return train_array[train_idx], train_array[val_idx]

    def _apply_diffusion(self, train_data, val_data, k_neighbors, diffusion_time):
        if k_neighbors <= 0 or diffusion_time <= 0:
            return train_data, val_data

        diffusion = HeatDiffusionFast(k=k_neighbors, t=diffusion_time)
        if len(val_data):
            merged = np.concatenate([train_data, val_data], axis=0)
            diffused = diffusion.fit_transform(merged)
            train_len = len(train_data)
            return diffused[:train_len], diffused[train_len:]

        diffused = diffusion.fit_transform(train_data)
        return diffused, val_data

    def _make_early_stop_checker(self, stop_state):
        rules = self.early_stop_rules

        def checker(epoch, metrics, history=None):
            if metrics is None or epoch < rules['check_epoch']:
                return None
            purity = metrics.get('purity', 0.0)
            nk_usage = metrics.get('nk_usage', 0.0)
            combo = metrics.get('combo', 0.0)

            if purity < rules['purity_min']:
                reason = f"purity {purity:.3f} < {rules['purity_min']}"
            elif nk_usage < rules['nk_usage_min']:
                reason = f"nk {nk_usage:.3f} < {rules['nk_usage_min']}"
            elif combo > rules['combo_max']:
                reason = f"combo {combo:.3f} > {rules['combo_max']}"
            else:
                return None

            stop_state['reason'] = reason
            return reason

        return checker

    def run_experiment(self, train_data, *, k_neighbors, latent_dims, arch_name,
                       entropy_cap=None, tau_init=0.5, tau_final=None,
                       tau_decay_start=0, tau_decay_end=None, n_archetypes=None,
                       combo_weight=0.5, batch_entropy_weight=0.1,
                       kl_warmup=10, kl_ramp=30, purity_warmup=20, purity_ramp=30,
                       cell_entropy_max=0.1, learning_rate=1e-3, batch_size=512,
                       diffusion_time=0.5, sample_weight_scale=1.0,
                       composite_min_fraction=0.02, aux_lineage_weight=0.2,
                       aux_subtype_weight=1.0, purity_margin=0.65, purity_lambda=0.7,
                       stage_epoch=None, stage_cell_entropy_scale=1.5,
                       stage_purity_lambda_scale=1.5, use_early_stop=True,
                       val_data=None,
                       n_epochs=50, device='cpu'):
        n_archetypes = n_archetypes or self.archetype_counts[0]
        exp_name = self._build_experiment_name(arch_name, k_neighbors, entropy_cap, tau_init, n_archetypes)
        stage_epoch_effective = stage_epoch if stage_epoch is not None else int(0.6 * n_epochs)
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"  k_neighbors : {k_neighbors}")
        print(f"  latent_dims : {latent_dims}")
        print(f"  entropy_cap : {entropy_cap}")
        print(f"  tau_init    : {tau_init}")
        print(f"  tau_final   : {tau_final}")
        print(f"  archetypes  : {n_archetypes}")
        print(f"  lr          : {learning_rate}")
        print(f"  batch_size  : {batch_size}")
        print(f"  combo_w     : {combo_weight}")
        print(f"  batch_ent_w : {batch_entropy_weight}")
        print(f"  diffusion_t : {diffusion_time}")
        print(f"  stage_epoch : {stage_epoch_effective}")
        print(f"  purity_margin / λ: {purity_margin} / {purity_lambda}")
        print(f"{'='*70}")

        X_train_raw, X_val_raw = self._split_data(train_data, val_data)
        X_train_processed, X_val_processed = self._apply_diffusion(X_train_raw, X_val_raw, k_neighbors, diffusion_time)

        print(f"  Train cells : {len(X_train_processed):,}")
        print(f"  Val cells   : {len(X_val_processed):,}")

        original_composite_min = getattr(self.config, 'composite_min_fraction', 0.02)
        self.config.composite_min_fraction = composite_min_fraction
        self._current_sample_weight_scale = sample_weight_scale
        aux_train = self._create_biology_based_aux_labels(X_train_processed)
        aux_val = self._create_biology_based_aux_labels(X_val_processed)

        train_targets = self.config.build_deconvolved_targets(X_train_processed, aux_train)
        val_targets = self.config.build_deconvolved_targets(X_val_processed, aux_val)

        train_dataset = FlowDataset(X_train_processed, aux_train, targets=train_targets)
        val_dataset = FlowDataset(X_val_processed, aux_val, targets=val_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        try:
            model = HierarchicalVAE_WithArchetypes(
                self.config,
                latent_dims=latent_dims,
                n_archetypes=n_archetypes,
                hidden_dims=[128, 96, 64]
            )

            print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

            stop_state = {'reason': None}
            early_stop_checker = self._make_early_stop_checker(stop_state) if use_early_stop else None

            start = time.time()
            model, history = self.live_trainer.fit(
                model,
                train_loader,
                val_loader,
                n_epochs=n_epochs,
                lr=learning_rate,
                device=device,
                save_dir=self.base_save_dir,
                experiment_name=exp_name,
                entropy_cap=entropy_cap,
                tau_init=tau_init,
                tau_final=tau_final,
                tau_decay_start=tau_decay_start,
                tau_decay_end=tau_decay_end,
                combo_weight=combo_weight,
                batch_entropy_weight=batch_entropy_weight,
                kl_warmup_epochs=kl_warmup,
                kl_ramp_epochs=kl_ramp,
                purity_warmup_epochs=purity_warmup,
                purity_ramp_epochs=purity_ramp,
                cell_entropy_max=cell_entropy_max,
                aux_lineage_weight=aux_lineage_weight,
                aux_subtype_weight=aux_subtype_weight,
                purity_margin=purity_margin,
                purity_lambda=purity_lambda,
                stage_epoch=stage_epoch_effective,
                stage_cell_entropy_scale=stage_cell_entropy_scale,
                stage_purity_lambda_scale=stage_purity_lambda_scale,
                trial_epochs=self.trial_epochs,
                early_stop_checker=early_stop_checker
            )
            elapsed = time.time() - start

            val_loss_hist = history.get('val_loss', [])
            best_idx = int(np.argmin(val_loss_hist)) if val_loss_hist else -1
            best_val_loss = float(val_loss_hist[best_idx]) if best_idx >= 0 else None
            final_val_loss = float(val_loss_hist[-1]) if val_loss_hist else None

            best_combo = float(history['val_combo'][best_idx]) if best_idx >= 0 and history.get('val_combo') else None
            best_purity = float(history['val_purity'][best_idx]) if best_idx >= 0 and history.get('val_purity') else None
            best_nk = float(history['val_nk_usage'][best_idx]) if best_idx >= 0 and history.get('val_nk_usage') else None

            result = {
                'experiment_name': exp_name,
                'k_neighbors': k_neighbors,
                'latent_dims': list(latent_dims),
                'arch_name': arch_name,
                'entropy_cap': float(entropy_cap) if entropy_cap is not None else None,
                'tau_init': float(tau_init),
                'tau_final': None if tau_final is None else float(tau_final),
                'tau_decay_start': int(tau_decay_start) if tau_decay_start is not None else None,
                'tau_decay_end': int(tau_decay_end) if tau_decay_end is not None else None,
                'n_params': int(sum(p.numel() for p in model.parameters())),
                'n_archetypes': int(n_archetypes),
                'n_train': int(len(X_train_processed)),
                'n_val': int(len(X_val_processed)),
                'best_val_loss': best_val_loss,
                'final_val_loss': final_val_loss,
                'best_epoch': best_idx + 1 if best_idx >= 0 else None,
                'completed_epochs': len(val_loss_hist),
                'best_val_combo': best_combo,
                'best_val_purity': best_purity,
                'best_val_nk_usage': best_nk,
                'stop_reason': stop_state['reason'],
                'elapsed_sec': elapsed,
                'trial_epochs': self.trial_epochs,
                'learning_rate': float(learning_rate),
                'batch_size': int(batch_size),
                'diffusion_time': float(diffusion_time),
                'combo_weight': float(combo_weight),
                'batch_entropy_weight': float(batch_entropy_weight),
                'kl_warmup': int(kl_warmup),
                'kl_ramp': int(kl_ramp),
                'purity_warmup': int(purity_warmup),
                'purity_ramp': int(purity_ramp),
            'cell_entropy_max': float(cell_entropy_max),
            'sample_weight_scale': float(sample_weight_scale),
            'composite_min_fraction': float(composite_min_fraction),
            'aux_lineage_weight': float(aux_lineage_weight),
            'aux_subtype_weight': float(aux_subtype_weight),
            'purity_margin': float(purity_margin),
            'purity_lambda': float(purity_lambda),
            'stage_epoch': int(stage_epoch_effective),
            'stage_cell_entropy_scale': float(stage_cell_entropy_scale),
            'stage_purity_lambda_scale': float(stage_purity_lambda_scale),
        }
            self.results.append(result)

            if best_val_loss is not None:
                print(f"  → Best val loss: {best_val_loss:.4f} at epoch {result['best_epoch']}")
            if stop_state['reason']:
                print(f"  ⚠️ Early stop: {stop_state['reason']}")

            return model, history
        finally:
            self.config.composite_min_fraction = original_composite_min
            self._current_sample_weight_scale = 1.0

    def run_sweep(self, train_data, val_data=None, n_epochs=50, device='cpu'):
        combos = list(product(
            self.k_neighbors,
            self.latent_grid.items(),
            self.entropy_caps,
            self.tau_inits,
            self.tau_finals,
            self.tau_decay_starts,
            self.tau_decay_ends,
            self.archetype_counts,
            self.combo_weights,
            self.batch_entropy_weights,
            self.kl_warmups,
            self.kl_ramps,
            self.purity_warmups,
            self.purity_ramps,
            self.cell_entropy_maxes,
            self.learning_rates,
            self.batch_sizes,
            self.diffusion_times,
            self.sample_weight_scales,
            self.composite_min_fractions,
            self.aux_lineage_weights,
            self.aux_subtype_weights
        ))

        total = len(combos)
        print(f"\n{'#'*70}")
        print("STARTING ADVANCED EXPERIMENT SWEEP")
        print(f"{'#'*70}")
        print(f"Total experiments: {total}")
        print(f"k_neighbors: {self.k_neighbors}")
        print(f"Architectures: {list(self.latent_grid.keys())}")
        print(f"Entropy caps: {self.entropy_caps}")
        print(f"Tau init: {self.tau_inits}")
        print(f"Archetype counts: {self.archetype_counts}")
        print(f"Tau finals: {self.tau_finals}")
        print(f"Combo weights: {self.combo_weights}")
        print(f"Batch entropy weights: {self.batch_entropy_weights}")
        print(f"KL warmups: {self.kl_warmups}")
        print(f"KL ramps: {self.kl_ramps}")
        print(f"Purity warmups: {self.purity_warmups}")
        print(f"Purity ramps: {self.purity_ramps}")
        print(f"Cell entropy max: {self.cell_entropy_maxes}")
        print(f"Learning rates: {self.learning_rates}")
        print(f"Batch sizes: {self.batch_sizes}")
        print(f"Diffusion times: {self.diffusion_times}")
        print(f"Sample weight scales: {self.sample_weight_scales}")
        print(f"Composite min fractions: {self.composite_min_fractions}")
        print(f"{'#'*70}\n")

        for idx, combo in enumerate(combos, start=1):
            (k_neighbors, (arch_name, latent_dims), entropy_cap, tau_init, tau_final,
             tau_decay_start, tau_decay_end, n_arch, combo_weight, batch_entropy_weight,
             kl_warmup, kl_ramp, purity_warmup, purity_ramp, cell_entropy_max,
             learning_rate, batch_size, diffusion_time, sample_weight_scale,
             composite_min_fraction, aux_lineage_weight, aux_subtype_weight) = combo

            print(f"\n[Experiment {idx}/{total}] {arch_name} | k={k_neighbors} | entropy={entropy_cap} | tau={tau_init}→{tau_final} | archetypes={n_arch} | lr={learning_rate} | batch={batch_size}")
            try:
                self.run_experiment(
                    train_data,
                    k_neighbors=k_neighbors,
                    latent_dims=latent_dims,
                    arch_name=arch_name,
                    entropy_cap=entropy_cap,
                    tau_init=tau_init,
                    tau_final=tau_final,
                    tau_decay_start=tau_decay_start,
                    tau_decay_end=tau_decay_end,
                    n_archetypes=n_arch,
                    combo_weight=combo_weight,
                    batch_entropy_weight=batch_entropy_weight,
                    kl_warmup=kl_warmup,
                    kl_ramp=kl_ramp,
                    purity_warmup=purity_warmup,
                    purity_ramp=purity_ramp,
                    cell_entropy_max=cell_entropy_max,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    diffusion_time=diffusion_time,
                    sample_weight_scale=sample_weight_scale,
                    composite_min_fraction=composite_min_fraction,
                    aux_lineage_weight=aux_lineage_weight,
                    aux_subtype_weight=aux_subtype_weight,
                    val_data=val_data,
                    n_epochs=n_epochs,
                    device=device
                )
            except Exception as exc:
                print(f"  ✗ Experiment failed: {exc}")
                continue

        self.save_summary()
        print(f"\n{'#'*70}")
        print("SWEEP COMPLETE")
        print(f"{'#'*70}")
        print(f"Results saved to: {self.base_save_dir / 'sweep_results.json'}")
        return self.results

    def _choice(self, options):
        options = list(options)
        if not options:
            raise ValueError("Empty option list encountered during random sampling.")
        idx = self._rng.integers(len(options))
        return options[idx]

    def _sample_hyperparameters(self):
        return dict(
            k_neighbors=self._choice(self.k_neighbors),
            latent_item=self._choice(list(self.latent_grid.items())),
            entropy_cap=self._choice(self.entropy_caps),
            tau_init=self._choice(self.tau_inits),
            tau_final=self._choice(self.tau_finals),
            tau_decay_start=self._choice(self.tau_decay_starts),
            tau_decay_end=self._choice(self.tau_decay_ends),
            n_archetypes=self._choice(self.archetype_counts),
            combo_weight=self._choice(self.combo_weights),
            batch_entropy_weight=self._choice(self.batch_entropy_weights),
            kl_warmup=self._choice(self.kl_warmups),
            kl_ramp=self._choice(self.kl_ramps),
            purity_warmup=self._choice(self.purity_warmups),
            purity_ramp=self._choice(self.purity_ramps),
            cell_entropy_max=self._choice(self.cell_entropy_maxes),
            learning_rate=self._choice(self.learning_rates),
            batch_size=self._choice(self.batch_sizes),
            diffusion_time=self._choice(self.diffusion_times),
            sample_weight_scale=self._choice(self.sample_weight_scales),
            composite_min_fraction=self._choice(self.composite_min_fractions),
            aux_lineage_weight=self._choice(self.aux_lineage_weights),
            aux_subtype_weight=self._choice(self.aux_subtype_weights),
        )

    def run_random_search(self, train_data, val_data=None, n_trials=20, n_epochs=50, device='cpu'):
        """
        Sample random hyperparameter configurations and run individual experiments.
        Useful when the full grid would be too large.
        """
        print(f"\n{'#'*70}")
        print(f"STARTING RANDOM SWEEP ({n_trials} trials)")
        print(f"{'#'*70}")

        for trial in range(1, n_trials + 1):
            conf = self._sample_hyperparameters()
            arch_name, latent_dims = conf['latent_item']
            print(f"\n[Random Trial {trial}/{n_trials}] arch={arch_name}, k={conf['k_neighbors']}, "
                  f"tau={conf['tau_init']}->{conf['tau_final']}, archetypes={conf['n_archetypes']}, "
                  f"combo_w={conf['combo_weight']}, lr={conf['learning_rate']}")
            try:
                self.run_experiment(
                    train_data,
                    k_neighbors=conf['k_neighbors'],
                    latent_dims=latent_dims,
                    arch_name=arch_name,
                    entropy_cap=conf['entropy_cap'],
                    tau_init=conf['tau_init'],
                    tau_final=conf['tau_final'],
                    tau_decay_start=conf['tau_decay_start'],
                    tau_decay_end=conf['tau_decay_end'],
                    n_archetypes=conf['n_archetypes'],
                    combo_weight=conf['combo_weight'],
                    batch_entropy_weight=conf['batch_entropy_weight'],
                    kl_warmup=conf['kl_warmup'],
                    kl_ramp=conf['kl_ramp'],
                    purity_warmup=conf['purity_warmup'],
                    purity_ramp=conf['purity_ramp'],
                    cell_entropy_max=conf['cell_entropy_max'],
                    learning_rate=conf['learning_rate'],
                    batch_size=conf['batch_size'],
                    diffusion_time=conf['diffusion_time'],
                    sample_weight_scale=conf['sample_weight_scale'],
                    composite_min_fraction=conf['composite_min_fraction'],
                    aux_lineage_weight=conf['aux_lineage_weight'],
                    aux_subtype_weight=conf['aux_subtype_weight'],
                    val_data=val_data,
                    n_epochs=n_epochs,
                    device=device
                )
            except Exception as exc:
                print(f"  ✗ Trial failed: {exc}")
                continue

        self.save_summary()
        print(f"\n{'#'*70}")
        print("RANDOM SWEEP COMPLETE")
        print(f"{'#'*70}")
        print(f"Results saved to: {self.base_save_dir / 'sweep_results.json'}")
        return self.results

    def save_summary(self):
        if not self.results:
            return None

        df = pd.DataFrame(self.results)
        df.to_csv(self.base_save_dir / 'sweep_results.csv', index=False)

        def _json_default(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj

        with open(self.base_save_dir / 'sweep_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=_json_default)

        print("\nSummary saved!")
        return df

    def _create_biology_based_aux_labels(self, data):
        data = np.asarray(data, dtype=np.float32)
        n = len(data)

        viability_idx = self.config.input_idx['Viability']
        cd56_idx = self.config.input_idx['CD56']
        cd45ra_idx = self.config.input_idx['CD45RA']
        cd45ro_idx = self.config.input_idx['CD45RO']
        cd197_idx = self.config.input_idx['CD197']
        cd62l_idx = self.config.input_idx['CD62L']
        cd366_idx = self.config.input_idx['CD366']
        cd152_idx = self.config.input_idx['CD152']
        cd69_idx = self.config.input_idx['CD69']
        cd25_idx = self.config.input_idx['CD25']
        cd95_idx = self.config.input_idx['CD95']

        cd16tigit_idx = self.config.input_idx['CD16+TIGIT']
        cd8cd14_idx = self.config.input_idx['CD8+CD14']
        cd4cd33_idx = self.config.input_idx['CD4+CD33']
        cd279cd24_idx = self.config.input_idx['CD279+CD24']
        cd34cd223_idx = self.config.input_idx['CD34+CD223']
        cd3cd19_idx = self.config.input_idx['CD3+CD19']

        aux = {}
        aux['viability'] = torch.FloatTensor((data[:, viability_idx] > 0.2).astype(float))

        t_cell_signal = (
            (data[:, cd3cd19_idx] > np.percentile(data[:, cd3cd19_idx], 50)) &
            ((data[:, cd8cd14_idx] > 0.3) | (data[:, cd4cd33_idx] > 0.3))
        )
        aux['lineage_is_tcell'] = torch.FloatTensor(t_cell_signal.astype(float))

        b_cell_signal = (
            (data[:, cd3cd19_idx] > 0.2) &
            (data[:, cd3cd19_idx] < np.percentile(data[:, cd3cd19_idx], 60)) &
            (data[:, cd8cd14_idx] < 0.3) &
            (data[:, cd4cd33_idx] < 0.3)
        )
        aux['lineage_is_bcell'] = torch.FloatTensor(b_cell_signal.astype(float))

        myeloid_signal = (
            (data[:, cd8cd14_idx] > np.percentile(data[:, cd8cd14_idx], 60)) |
            (data[:, cd4cd33_idx] > np.percentile(data[:, cd4cd33_idx], 60)) |
            (data[:, cd34cd223_idx] > np.percentile(data[:, cd34cd223_idx], 55))
        )
        aux['lineage_is_myeloid'] = torch.FloatTensor(myeloid_signal.astype(float))

        nk_signal = (
            (data[:, cd16tigit_idx] > np.percentile(data[:, cd16tigit_idx], 60)) &
            (data[:, cd56_idx] > np.percentile(data[:, cd56_idx], 50)) &
            (data[:, cd3cd19_idx] < 0.3)
        )
        aux['lineage_is_nk'] = torch.FloatTensor(nk_signal.astype(float))

        cd4_high = data[:, cd4cd33_idx] > np.percentile(data[:, cd4cd33_idx], 60)
        cd8_high = data[:, cd8cd14_idx] > np.percentile(data[:, cd8cd14_idx], 60)

        aux['subtype_is_cd4'] = torch.FloatTensor(cd4_high.astype(float))
        aux['subtype_is_cd8'] = torch.FloatTensor(cd8_high.astype(float))
        aux['subtype_is_double_positive'] = torch.FloatTensor((cd4_high & cd8_high).astype(float))

        aux['subtype_is_naive'] = torch.FloatTensor(
            ((data[:, cd45ra_idx] > np.percentile(data[:, cd45ra_idx], 60)) &
             (data[:, cd62l_idx] > np.percentile(data[:, cd62l_idx], 50)) &
             (data[:, cd45ro_idx] < np.percentile(data[:, cd45ro_idx], 40))).astype(float)
        )

        aux['subtype_is_central_memory'] = torch.FloatTensor(
            ((data[:, cd45ro_idx] > np.percentile(data[:, cd45ro_idx], 50)) &
             (data[:, cd62l_idx] > np.percentile(data[:, cd62l_idx], 50)) &
             (data[:, cd197_idx] > np.percentile(data[:, cd197_idx], 40))).astype(float)
        )

        aux['subtype_is_effector_memory'] = torch.FloatTensor(
            ((data[:, cd45ro_idx] > np.percentile(data[:, cd45ro_idx], 50)) &
             (data[:, cd62l_idx] < np.percentile(data[:, cd62l_idx], 40)) &
             (data[:, cd197_idx] < np.percentile(data[:, cd197_idx], 40))).astype(float)
        )

        aux['subtype_is_activated'] = torch.FloatTensor(
            ((data[:, cd69_idx] > np.percentile(data[:, cd69_idx], 65)) |
             (data[:, cd25_idx] > np.percentile(data[:, cd25_idx], 70))).astype(float)
        )

        aux['subtype_is_exhausted'] = torch.FloatTensor(
            ((data[:, cd366_idx] > np.percentile(data[:, cd366_idx], 70)) &
             (data[:, cd152_idx] > np.percentile(data[:, cd152_idx], 70))).astype(float)
        )

        lymphocyte_mask = (aux['lineage_is_tcell'] + aux['lineage_is_bcell'] + aux['lineage_is_nk']) > 0.5
        lymphocyte_mask = lymphocyte_mask.float()
        aux['lymphocyte_mask'] = lymphocyte_mask

        rare_keys = ['subtype_is_double_positive', 'subtype_is_exhausted', 'subtype_is_activated']
        rare_mask = torch.zeros(n, dtype=torch.float32)
        for key in rare_keys:
            if key in aux:
                rare_mask = torch.maximum(rare_mask, aux[key])
        aux['rare_mask'] = (rare_mask > 0.5).float()
        rare_bonus = rare_mask.clamp(0.0, 1.0)

        scale = float(getattr(self, '_current_sample_weight_scale', 1.0))
        myeloid_weight = torch.where(aux['lineage_is_myeloid'] > 0.5,
                                     torch.full((n,), 1.2, dtype=torch.float32),
                                     torch.zeros(n, dtype=torch.float32))
        base_weight_raw = torch.where(
            lymphocyte_mask > 0.5,
            torch.full((n,), 2.4, dtype=torch.float32),
            torch.full((n,), 0.3, dtype=torch.float32)
        ) + myeloid_weight
        sample_weight = (base_weight_raw + 1.2 * rare_bonus) * scale
        min_w = 0.4 * scale
        max_w = 5.0 * scale
        sample_weight = sample_weight.clamp(min_w, max_w if max_w > min_w else min_w + 1e-3)
        aux['sample_weight'] = sample_weight

        return aux


ExperimentSweep = ExperimentSweepAdvanced

print("✅ Experiment sweep framework loaded")


class FlowDataset(Dataset):
    def __init__(self, data, aux_labels, targets=None):
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        processed_aux = {}
        for key, values in aux_labels.items():
            if isinstance(values, torch.Tensor):
                tensor = values.detach().clone().float()
            else:
                tensor = torch.as_tensor(values, dtype=torch.float32)
            processed_aux[key] = tensor
        self.aux_labels = processed_aux
        self.sample_weights = self.aux_labels.get('sample_weight')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {'data': self.data[idx]}
        if self.targets is not None:
            item['recon_target'] = self.targets[idx]
        else:
            item['recon_target'] = self.data[idx]
        if self.sample_weights is not None:
            item['sample_weight'] = self.sample_weights[idx]
        for task_name, labels in self.aux_labels.items():
            if task_name == 'sample_weight':
                continue
            item[task_name] = labels[idx]
        return item
print("✅ Experiment sweep framework loaded")

# Cell 11 (Enhanced Diagnostics)

# We need this for the more complex grid
import matplotlib.gridspec as gridspec

def generate_diagnostic_plots(model, val_loader, config, device, epoch, save_dir, history, live=None):
    """Generate enhanced diagnostics with composite consistency tracking."""
    if live is False:
        return

    interactive = live is True
    mode = "display" if interactive else "save"
    print(f"  Generating diagnostic plot for epoch {epoch} ({mode})...")
    model.eval()

    try:
        batch = next(iter(val_loader))
        x = batch['data'].to(device)
    except StopIteration:
        print("  Could not get validation batch.")
        return

    with torch.no_grad():
        (mu1, logvar1, z1), (mu2, logvar2, z2), (z3, alpha, archetypes, entropy) = model.encode(x)
        recon_archetypes = model.decoder(archetypes).cpu().numpy()
        recon_batch = model.decoder(z3).detach()

        alpha_norm = alpha / (alpha.sum(dim=0, keepdim=True) + 1e-6)
        avg_z1_for_archetypes = alpha_norm.T @ z1
        avg_z2_for_archetypes = alpha_norm.T @ z2
        avg_z3_for_archetypes = alpha_norm.T @ z3

        aux_scores = {}
        via_logits = model.aux_viability(avg_z1_for_archetypes)
        aux_scores['viability'] = torch.sigmoid(via_logits).cpu().numpy().flatten()

        for name, classifier in model.aux_lineage.items():
            logits = classifier(avg_z2_for_archetypes)
            aux_scores[name] = torch.sigmoid(logits).cpu().numpy().flatten()

        for name, classifier in model.aux_subtypes.items():
            logits = classifier(avg_z3_for_archetypes)
            aux_scores[name] = torch.sigmoid(logits).cpu().numpy().flatten()

        aux_df = pd.DataFrame(aux_scores, index=[f"Arch {i}" for i in range(model.n_archetypes)])

        alpha_cpu = alpha.detach().cpu()
        x_cpu = batch['data'] if isinstance(batch['data'], torch.Tensor) else torch.tensor(batch['data'])
        x_cpu = x_cpu.cpu()
        recon_cpu = recon_batch.cpu()

        lymph_mask_tensor = batch.get('lymphocyte_mask')
        if lymph_mask_tensor is not None:
            lymph_mask = (lymph_mask_tensor > 0.5).cpu()
        else:
            lymph_components = []
            for key in config.lymphocyte_aux_keys:
                if key in batch:
                    lymph_components.append(batch[key].cpu())
            if lymph_components:
                lymph_mask = (torch.stack(lymph_components).sum(dim=0) > 0)
            else:
                lymph_mask = torch.zeros(len(x_cpu), dtype=torch.bool)
        lymph_count = int(lymph_mask.sum().item())

        rare_masks = {}
        for key in config.rare_aux_keys:
            if key in batch:
                rare_masks[key] = (batch[key] > 0.5).cpu()
        if rare_masks:
            rare_mask = torch.stack(list(rare_masks.values())).any(dim=0)
        else:
            rare_mask = torch.zeros(len(x_cpu), dtype=torch.bool)
        rare_count = int(rare_mask.sum().item())

        if lymph_count > 0:
            lymph_usage = alpha_cpu[lymph_mask].mean(dim=0).numpy()
        else:
            lymph_usage = np.zeros(model.n_archetypes)

        if rare_count > 0:
            rare_usage = alpha_cpu[rare_mask].mean(dim=0).numpy()
        else:
            rare_usage = np.zeros(model.n_archetypes)

        combo_losses = []
        combo_errors = {}
        for comp_name, marker_indices in config.composite_marker_indices.items():
            idx_a, idx_b = marker_indices
            combo_idx = config.input_idx[comp_name]
            pred_combo = (recon_cpu[:, idx_a] + recon_cpu[:, idx_b])
            true_combo = x_cpu[:, combo_idx]
            error = (pred_combo - true_combo)
            combo_losses.append((error.pow(2).mean().item()))
            combo_errors[comp_name] = error.abs().mean().item()
        combo_consistency_current = float(np.mean(combo_losses)) if combo_losses else 0.0

        epochs = np.arange(1, len(history['train_loss']) + 1)
        train_combo_hist = history.get('train_combo', [])
        val_combo_hist = history.get('val_combo', [])
        batch_entropy_hist = history.get('train_batch_entropy', [])

        current_metrics = {
            'Train KL1': history['train_kl1'][-1] if history['train_kl1'] else 0.0,
            'Train KL2': history['train_kl2'][-1] if history['train_kl2'] else 0.0,
            'Batch Combo': combo_consistency_current,
            'Val Combo': history['val_combo'][-1] if history.get('val_combo') else combo_consistency_current,
            'Val Purity': history['val_purity'][-1] if history.get('val_purity') else 0.0,
            'Val PurLoss': history['val_purity_loss'][-1] if history.get('val_purity_loss') else 0.0,
            'NK Usage': history['val_nk_usage'][-1] if history.get('val_nk_usage') else 0.0,
        }

    fig = plt.figure(figsize=(24, 30))
    gs = fig.add_gridspec(5, 3)
    fig.suptitle(f'Diagnostics - Epoch {epoch}', fontsize=20, fontweight='bold')

    ax_z1 = fig.add_subplot(gs[0, 0])
    ax_z2 = fig.add_subplot(gs[0, 1])
    ax_loss_hist = fig.add_subplot(gs[0, 2])

    z1_np = z1.cpu().numpy()
    ax_z1.scatter(z1_np[:, 0], z1_np[:, 1], s=1, alpha=0.3)
    ax_z1.set_title('z1 Latent Space (Viability)')

    z2_np = z2.cpu().numpy()
    ax_z2.scatter(z2_np[:, 0], z2_np[:, 1], s=1, alpha=0.3, c='green')
    ax_z2.set_title('z2 Latent Space (Lineage)')

    ax_loss_hist.plot(epochs, history['train_loss'], label='Train Loss', alpha=0.8)
    ax_loss_hist.plot(epochs, history['val_loss'], label='Val Loss', alpha=0.8)
    ax_loss_hist.set_title('Loss History (log scale)')
    ax_loss_hist.set_xlabel('Epoch')
    ax_loss_hist.set_ylabel('Loss')
    ax_loss_hist.set_yscale('log')
    ax_loss_hist.legend()
    ax_loss_hist.grid(True, alpha=0.3)

    ax_purity = fig.add_subplot(gs[1, 0])
    ax_usage = fig.add_subplot(gs[1, 1])
    ax_combo_hist = fig.add_subplot(gs[1, 2])

    purity = alpha.max(dim=1).values.cpu().numpy()
    ax_purity.hist(purity, bins=50, range=(0, 1))
    ax_purity.set_title('Archetype Purity (max α)')

    mean_weights = alpha.mean(dim=0).cpu().numpy()
    ax_usage.bar(range(model.n_archetypes), mean_weights, color='steelblue')
    ax_usage.set_title('Mean Archetype Usage')
    ax_usage.set_xticks(range(model.n_archetypes))

    if train_combo_hist:
        ax_combo_hist.plot(epochs, train_combo_hist, label='Train Combo Loss', color='purple')
    if val_combo_hist:
        ax_combo_hist.plot(epochs[:len(val_combo_hist)], val_combo_hist, label='Val Combo Loss', color='orange')
    if batch_entropy_hist:
        ax_combo_hist_t = ax_combo_hist.twinx()
        ax_combo_hist_t.plot(epochs, batch_entropy_hist, label='Batch Entropy', color='gray', linestyle='--', alpha=0.6)
        handles1, labels1 = ax_combo_hist.get_legend_handles_labels()
        handles2, labels2 = ax_combo_hist_t.get_legend_handles_labels()
        ax_combo_hist_t.set_ylabel('Batch Entropy')
        ax_combo_hist.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    else:
        ax_combo_hist.legend(loc='upper right')
    ax_combo_hist.set_title('Composite Consistency History')
    ax_combo_hist.set_xlabel('Epoch')
    ax_combo_hist.set_ylabel('MSE')
    ax_combo_hist.grid(True, alpha=0.3)

    ax_heatmap = fig.add_subplot(gs[2, 0:2])
    ax_metric_bar = fig.add_subplot(gs[2, 2])

    if recon_archetypes.shape[0] > 1:
        row_mean = recon_archetypes.mean(axis=1, keepdims=True)
        row_std = recon_archetypes.std(axis=1, keepdims=True)
        recon_archetypes_scaled = (recon_archetypes - row_mean) / (row_std + 1e-8)
    else:
        recon_archetypes_scaled = recon_archetypes

    sns.heatmap(
        recon_archetypes_scaled,
        ax=ax_heatmap,
        yticklabels=[f"Arch {i}" for i in range(model.n_archetypes)],
        xticklabels=config.output_markers,
        cmap='vlag',
        center=0
    )
    ax_heatmap.set_title('Decoded Archetype Expression (Row-Scaled)')
    ax_heatmap.tick_params(axis='x', rotation=90)

    metric_colors = sns.color_palette("husl", len(current_metrics))
    ax_metric_bar.bar(list(current_metrics.keys()), list(current_metrics.values()), color=metric_colors)
    ax_metric_bar.set_title('Current Epoch Metrics')
    ax_metric_bar.set_ylabel('Value')

    ax_lymph = fig.add_subplot(gs[3, 0])
    ax_rare = fig.add_subplot(gs[3, 1])
    ax_combo_detail = fig.add_subplot(gs[3, 2])

    if lymph_count > 0:
        top_idx = np.argsort(lymph_usage)[::-1][:5]
        ax_lymph.bar(range(len(top_idx)), lymph_usage[top_idx], color='teal')
        ax_lymph.set_xticks(range(len(top_idx)))
        ax_lymph.set_xticklabels([f"Arch {i}" for i in top_idx])
        ax_lymph.set_title(f'Lymphocyte Archetype Usage (n={lymph_count})')
    else:
        ax_lymph.text(0.5, 0.5, 'No lymphocyte labels in batch', ha='center', va='center')
        ax_lymph.set_title('Lymphocyte Archetype Usage')

    if rare_count > 0:
        top_idx = np.argsort(rare_usage)[::-1][:5]
        ax_rare.bar(range(len(top_idx)), rare_usage[top_idx], color='darkorange')
        ax_rare.set_xticks(range(len(top_idx)))
        ax_rare.set_xticklabels([f"Arch {i}" for i in top_idx])
        rare_summary = ', '.join([f"{k}:{int(v.sum().item())}" for k, v in rare_masks.items()])
        ax_rare.set_title(f'Rare Population Usage (n={rare_count}), {rare_summary}')
    else:
        ax_rare.text(0.5, 0.5, 'No rare-pop labels in batch', ha='center', va='center')
        ax_rare.set_title('Rare Population Usage')

    if combo_errors:
        combo_names = list(combo_errors.keys())
        combo_values = [combo_errors[name] for name in combo_names]
        ax_combo_detail.barh(combo_names, combo_values, color='mediumpurple')
        ax_combo_detail.set_xlabel('Mean |Prediction - Raw Combo|')
        ax_combo_detail.set_title('Composite Channel Residuals')
    else:
        ax_combo_detail.text(0.5, 0.5, 'No composite channels available', ha='center', va='center')
        ax_combo_detail.set_title('Composite Channel Residuals')

    ax_aux_heatmap = fig.add_subplot(gs[4, :])
    sns.heatmap(
        aux_df,
        ax=ax_aux_heatmap,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0.5,
        linewidths=0.5
    )
    ax_aux_heatmap.set_title('Archetype Auxiliary Label Scores (Probability)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save_dir is not None:
        plot_dir = Path(save_dir)
        plot_dir.mkdir(exist_ok=True, parents=True)
        plot_path = plot_dir / f'diagnostic_epoch_{epoch}.png'
        plt.savefig(plot_path, dpi=100)
    if interactive:
        display(fig)
    plt.close(fig)


class LiveMetricRegistry:
    """Collect and expose the most recent metrics for each live training run."""

    def __init__(self):
        self._latest = {}
        self._history = {}

    @staticmethod
    def _sanitize(metrics):
        clean = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.detach().cpu().item()
                else:
                    value = value.detach().cpu().tolist()
            elif isinstance(value, np.generic):
                value = float(value)
            clean[key] = value
        return clean

    def update(self, run_name, metrics):
        clean = self._sanitize(metrics)
        self._latest[run_name] = clean
        self._history.setdefault(run_name, []).append(clean)
        return clean

    def latest(self):
        return {name: metrics.copy() for name, metrics in self._latest.items()}

    def history(self, run_name):
        return list(self._history.get(run_name, []))

    def as_dataframe(self):
        if not self._latest:
            columns = ['experiment', 'epoch', 'loss', 'val_loss', 'purity', 'nk_usage', 'combo', 'tau']
            return pd.DataFrame(columns=columns)
        records = []
        for name, metrics in self._latest.items():
            record = {'experiment': name}
            record.update(metrics)
            records.append(record)
        df = pd.DataFrame.from_records(records)
        sort_cols = [col for col in ['val_loss', 'loss', 'purity'] if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        return df.reset_index(drop=True)


class LiveTrainer:
    """Wrapper around the training loop that streams metrics into a live registry."""

    def __init__(self, config, live_registry=None):
        self.config = config
        self.live_registry = live_registry or LiveMetricRegistry()
        self.last_history = None

    def fit(self, model, train_loader, val_loader, **trainer_kwargs):
        run_name = trainer_kwargs.get('run_name') or trainer_kwargs.get('experiment_name')
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trainer_kwargs['run_name'] = run_name
        trainer_kwargs.setdefault('live_registry', self.live_registry)
        model, history = train_hierarchical_vae_with_archetypes(
            model,
            train_loader,
            val_loader,
            **trainer_kwargs
        )
        self.last_history = history
        return model, history

    def latest_metrics(self):
        return self.live_registry.latest()

def plot_sweep_results(results_path='./experiments/sweep_results.csv'):
    """
    Visualize sweep outcomes. When multiple entropy caps / tau values are present,
    the plots aggregate by taking the best validation loss per (architecture, k).
    """
    df = pd.read_csv(results_path)

    if df.empty:
        print("No sweep results available.")
        return

    grouped = (
        df.groupby(['arch_name', 'k_neighbors', 'n_archetypes'], as_index=False)
        .agg({'best_val_loss': 'min', 'n_params': 'median'})
    )
    grouped['arch_label'] = grouped['arch_name'] + '_a' + grouped['n_archetypes'].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Best val loss vs k_neighbors for each architecture
    ax = axes[0, 0]
    for arch in grouped['arch_label'].unique():
        data = grouped[grouped['arch_label'] == arch]
        ax.plot(data['k_neighbors'], data['best_val_loss'], 'o-', label=arch, linewidth=2)
    ax.set_xlabel('k_neighbors (Heat Diffusion)')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Effect of Heat Diffusion k')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Best val loss vs architecture for each k_neighbors
    ax = axes[0, 1]
    unique_labels = grouped['arch_label'].unique()
    for k in grouped['k_neighbors'].unique():
        data = grouped[grouped['k_neighbors'] == k]
        y = data.set_index('arch_label').reindex(unique_labels)['best_val_loss']
        ax.plot(range(len(unique_labels)), y, 'o-', label=f'k={k}', linewidth=2)
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels, rotation=45, ha='right')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Effect of Archetype Dimensions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Heatmap: k_neighbors × architecture
    ax = axes[1, 0]
    pivot = grouped.pivot_table(index='arch_label', columns='k_neighbors', values='best_val_loss')
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis_r', ax=ax)
    ax.set_title('Best Val Loss Heatmap')
    ax.set_xlabel('k_neighbors')
    ax.set_ylabel('Architecture × n_archetypes')

    # 4. Model complexity vs performance
    ax = axes[1, 1]
    for k in grouped['k_neighbors'].unique():
        data = grouped[grouped['k_neighbors'] == k]
        ax.scatter(data['n_params'], data['best_val_loss'], label=f'k={k}', s=100, alpha=0.7)
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Model Complexity vs Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(results_path).parent / 'sweep_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\n" + "="*70)
    print("SWEEP SUMMARY")
    print("="*70)
    best_config = df.loc[df['best_val_loss'].idxmin()]
    print(f"\nBest Configuration:")
    print(f"  Experiment: {best_config['experiment_name']}")
    print(f"  k_neighbors: {best_config['k_neighbors']}")
    print(f"  Architecture: {best_config['arch_name']}")
    print(f"  Entropy cap: {best_config['entropy_cap']}")
    print(f"  Tau init: {best_config['tau_init']}")
    print(f"  Tau final: {best_config.get('tau_final')}")
    print(f"  Combo weight: {best_config.get('combo_weight')}")
    print(f"  Batch entropy weight: {best_config.get('batch_entropy_weight')}")
    print(f"  Learning rate: {best_config.get('learning_rate')}")
    print(f"  Batch size: {best_config.get('batch_size')}")
    print(f"  Best Val Loss: {best_config['best_val_loss']:.4f}")
    print(f"  Parameters: {best_config['n_params']:,}")

    print("\n" + "-"*70)
    print("Top 5 Configurations:")
    print("-"*70)
    top5 = df.nsmallest(5, 'best_val_loss')
    for i, row in enumerate(top5.itertuples(), 1):
        print(f"{i}. {row.experiment_name:20s} | Loss: {row.best_val_loss:.4f} | "
              f"k={row.k_neighbors}, arch={row.arch_name}, archetypes={row.n_archetypes}, ent={row.entropy_cap}, "
              f"tau={row.tau_init}->{row.tau_final}, lr={row.learning_rate}, batch={row.batch_size}")

def print_sweep_progress(results_path='./experiments/sweep_results.csv',
                         objective='best_val_loss',
                         purity_min=0.85,
                         nk_min=0.35,
                         combo_max=0.05,
                         top_k=5):
    """Console summary of sweep progress with constraint tracking."""
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"No results yet at {results_path}")
        return
    df = pd.read_csv(results_path)
    if df.empty:
        print("Results file is empty.")
        return

    if objective not in df.columns:
        raise ValueError(f"Objective column '{objective}' not found in results.")
    df = df.dropna(subset=[objective])
    if df.empty:
        print("No runs with objective values yet.")
        return

    constraint_checks = []
    if 'best_val_purity' in df.columns:
        constraint_checks.append(df['best_val_purity'] >= purity_min)
    if 'best_val_nk_usage' in df.columns:
        constraint_checks.append(df['best_val_nk_usage'] >= nk_min)
    if 'best_val_combo' in df.columns:
        constraint_checks.append(df['best_val_combo'] <= combo_max)
    if constraint_checks:
        meets = pd.concat(constraint_checks, axis=1).all(axis=1)
    else:
        meets = pd.Series(True, index=df.index)
    df['meets_constraints'] = meets

    n_runs = len(df)
    n_meet = int(df['meets_constraints'].sum())
    best_idx = df[objective].idxmin()
    best = df.loc[best_idx]

    print("\n" + "="*70)
    print("SWEEP PROGRESS")
    print("="*70)
    print(f"Runs completed         : {n_runs}")
    print(f"Constraint-satisfying  : {n_meet} ({n_meet / n_runs * 100:0.1f}%)")
    print(f"Best {objective:<17}: {best[objective]:.4f} ({best['experiment_name']})")
    if 'elapsed_sec' in df.columns:
        print(f"Avg run time            : {df['elapsed_sec'].mean() / 60:0.1f} min")

    print("\nTop configurations:")
    cols = [c for c in ['experiment_name', objective, 'meets_constraints', 'best_val_purity',
                        'best_val_nk_usage', 'best_val_combo', 'n_archetypes', 'k_neighbors',
                        'entropy_cap', 'tau_init', 'tau_final', 'combo_weight', 'learning_rate']
            if c in df.columns]
    display_df = df.sort_values(objective).head(top_k)[cols]
    print(display_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    print("\nParameter coverage:")
    for col in ['n_archetypes', 'k_neighbors', 'tau_init', 'tau_final', 'combo_weight', 'learning_rate']:
        if col in df.columns:
            counts = df[col].value_counts().sort_index()
            formatted = ", ".join(f"{idx}:{val}" for idx, val in counts.items())
            print(f"  {col:<18} {formatted}")
    print("="*70)

def live_dashboard(live_registry, refresh_seconds=2.0):
    """
    Display a lightweight dashboard that streams the latest metrics from the live registry.
    """
    if widgets is None:
        print("ipywidgets not installed; run `pip install ipywidgets` for the live dashboard.")
        return None

    output = widgets.Output()
    refresh_button = widgets.Button(description='Refresh', icon='refresh')
    auto_toggle = widgets.ToggleButton(description='Auto Refresh', icon='repeat', value=False)
    interval_slider = widgets.FloatSlider(
        description='Interval (s)',
        min=0.5,
        max=10.0,
        step=0.5,
        value=refresh_seconds,
        readout=True
    )
    play = widgets.Play(interval=int(refresh_seconds * 1000), value=0, min=0, max=10_000, step=1)

    def render(_=None):
        with output:
            clear_output(wait=True)
            df = live_registry.as_dataframe()
            if df.empty:
                print("No live metrics yet. Start a run to stream updates here.")
                return
            ordered_cols = [col for col in ['experiment', 'epoch', 'loss', 'val_loss', 'purity', 'nk_usage', 'combo', 'tau'] if col in df.columns]
            selected = df[ordered_cols].copy() if ordered_cols else df.copy()
            for col in selected.select_dtypes(include=[np.number]).columns:
                selected[col] = selected[col].round(4)
            sort_col = 'val_loss' if 'val_loss' in selected.columns else selected.columns[0]
            display(selected.sort_values(sort_col))

    refresh_button.on_click(render)

    def on_interval(change):
        play.interval = int(change['new'] * 1000)

    interval_slider.observe(on_interval, names='value')

    def on_toggle(change):
        if change['new']:
            play.interval = int(interval_slider.value * 1000)
            play.start()
        else:
            play.stop()

    auto_toggle.observe(on_toggle, names='value')
    play.observe(render, names='value')

    controls = widgets.HBox([refresh_button, auto_toggle, interval_slider])
    dashboard = widgets.VBox([controls, output])
    display(dashboard)
    render()
    return dashboard

print("✅ Results visualization loaded")

# ============================================================================
# LOAD YOUR FCS FILES
# ============================================================================

# Option 1: Load real FCS files
USE_REAL_DATA = True  # Set to True when you have FCS files

if USE_REAL_DATA:
    loader = FCSLoader(config, arcsinh_cofactor=5.0)
    
    # Load normal samples for training
    X_train, sample_ids = loader.load_multiple(
    file_pattern='/Users/joshuafein/Library/CloudStorage/OneDrive-WeillCornellMedicine/flow_experiments/flow_September/raw_fcs/bma_normal/*.fcs',
    max_files=4,       # Will load first 4 FCS files found
    subsample=200000    # larger subsample for purity focus
    )
    
    # Preprocess
    X_train = loader.preprocess(X_train, apply_arcsinh=True, qc_viability=False)
    n_train = 80000


else:
    # Option 2: Dummy data for testing
    print("Using dummy data for demonstration")
    np.random.seed(42)
    n_train = 50000
    n_val = 10000
    
    X_train_split = np.random.randn(n_train, config.n_input).astype(np.float32)
    X_val = np.random.randn(n_val, config.n_input).astype(np.float32)
    
    print(f"Train: {X_train_split.shape}")
    print(f"Val: {X_val.shape}")

# # Test single configuration before full sweep
# train_source = X_train if USE_REAL_DATA else X_train_split
# val_source = None if USE_REAL_DATA else X_val

live_registry = LiveMetricRegistry()
  
sweep = ExperimentSweepAdvanced(
    config,
    base_save_dir='./experiments_full',
    entropy_caps=(None, 0.7),
    tau_inits=(0.5, 0.35),
    tau_finals=(0.15, 0.1),
    tau_decay_starts=(0,),
    tau_decay_ends=(50,),
    archetype_counts=list(range(6, 25, 3)),
    combo_weights=(0.7, 0.9),
    batch_entropy_weights=(0.02, 0.05),
    kl_warmups=(6,),
    kl_ramps=(20,),
    purity_warmups=(6,),
    purity_ramps=(20,),
    cell_entropy_maxes=(0.28,),
    learning_rates=(5e-4, 8e-4),
    batch_sizes=(512,),
    diffusion_times=(0.3, 0.5),
    sample_weight_scales=(1.2, 1.4),
    composite_min_fractions=(0.02, 0.035),
    aux_lineage_weights=(0.2, 0.3),
    aux_subtype_weights=(1.2, 1.5),
    trial_epochs=None,          # let each run finish
    purity_min=0.75,
    nk_usage_min=0.3,
    live_registry=live_registry
)

xl_dims = sweep.latent_grid['xlarge']
model, history = sweep.run_experiment(
    X_train,
    k_neighbors=15,
    latent_dims=xl_dims,
    arch_name='xlarge',
    entropy_cap=0.7,
    tau_init=0.35,
    tau_final=0.1,
    tau_decay_start=0,
    tau_decay_end=80,
    n_archetypes=15,
    combo_weight=0.85,
    batch_entropy_weight=0.03,
    kl_warmup=6,
    kl_ramp=20,
    purity_warmup=6,
    purity_ramp=20,
    cell_entropy_max=0.28,
    learning_rate=8e-4,
    batch_size=512,
    diffusion_time=0.4,
    sample_weight_scale=1.3,
    composite_min_fraction=0.025,
    aux_lineage_weight=0.28,
    aux_subtype_weight=1.4,
    purity_margin=0.7,
    purity_lambda=0.85,
    stage_epoch=90,
    stage_cell_entropy_scale=2.0,
    stage_purity_lambda_scale=2.0,
    use_early_stop=False,
    val_data=None,
    n_epochs=150,
    device=device
)

results = sweep.results

if widgets is not None:
    live_dashboard(live_registry)
else:
    print("Install ipywidgets to enable the live dashboard.")

# # ============================================================================
# # LOAD MODEL - FIXED FOR OLD CHECKPOINTS
# # ============================================================================

# def load_trained_model_safe(checkpoint_path, config, device='cpu'):
#     """
#     Load model from checkpoint that may not have config saved.
    
#     Args:
#         checkpoint_path: Path to checkpoint
#         config: Your PanelConfig (pass from notebook)
#         device: Device to load to
#     """
#     print(f"Loading model from: {checkpoint_path}")
    
#     # Load checkpoint
#     checkpoint = torch.load(checkpoint_path, map_location=device)
    
#     # Check what's in the checkpoint
#     print(f"  Checkpoint keys: {checkpoint.keys()}")
    
#     # Try to get architecture from checkpoint, or infer from path
#     if 'latent_dims' in checkpoint:
#         latent_dims = checkpoint['latent_dims']
#         hidden_dims = checkpoint['hidden_dims']
#     else:
#         # Infer from experiment name
#         if 'medium' in str(checkpoint_path).lower():
#             latent_dims = [16, 12, 8]
#             hidden_dims = [128, 96, 64]
#             print("  Inferred architecture: MEDIUM")
#         elif 'large' in str(checkpoint_path).lower():
#             latent_dims = [24, 18, 12]
#             hidden_dims = [192, 144, 96]
#             print("  Inferred architecture: LARGE")
#         elif 'xlarge' in str(checkpoint_path).lower():
#             latent_dims = [32, 24, 16]
#             hidden_dims = [256, 192, 128]
#             print("  Inferred architecture: XLARGE")
#         elif 'small' in str(checkpoint_path).lower():
#             latent_dims = [8, 6, 4]
#             hidden_dims = [64, 48, 32]
#             print("  Inferred architecture: SMALL")
#         else:
#             # Default to medium
#             latent_dims = [16, 12, 8]
#             hidden_dims = [128, 96, 64]
#             print("  Using default MEDIUM architecture")
    
#     print(f"  Architecture: latent_dims={latent_dims}, hidden_dims={hidden_dims}")
    
#     # Reconstruct model
#     model = HierarchicalVAE(
#         config=config,  # Use the config from your notebook
#         latent_dims=latent_dims,
#         hidden_dims=hidden_dims
#     )
    
#     # Load weights
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
#     model.eval()
    
#     # Print info
#     if 'epoch' in checkpoint:
#         print(f"  Trained for: {checkpoint['epoch']} epochs")
#     if 'best_val_loss' in checkpoint:
#         print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    
#     print("✅ Model loaded successfully!")
    
#     return model, checkpoint


# # ============================================================================
# # USAGE: Load with your existing config
# # ============================================================================

# # Use the config you already have in your notebook
# model, checkpoint = load_trained_model_safe(
#     'experiments_test/k15_arch_medium/best_model.pt',
#     config=config,  # This is your PanelConfig from earlier in the notebook
#     device=device
# )

# print(f"\nModel ready!")
# print(f"Latent dims: z1={model.latent_dims[0]}, z2={model.latent_dims[1]}, z3={model.latent_dims[2]}")

# # ===========================================================================
# # SAFE HIERARCHICAL VAE VALIDATION - CORRECTED
# # ===========================================================================

# def safe_validate_vae_with_archetypes(model, data, aux_labels, config, n_cells=5000, device='cpu'):
#     """
#     Safe validation for the ArchetypalLayer model.
#     Uses PCA and plots archetype usage.
#     """
#     from sklearn.decomposition import PCA
#     import matplotlib.pyplot as plt
#     import seaborn as sns
    
#     model.eval()
#     model = model.to(device)
    
#     # Subsample
#     print(f"📊 Subsampling {n_cells} cells...")
#     if len(data) > n_cells:
#         idx = np.random.choice(len(data), n_cells, replace=False)
#         data_subset = data[idx]
#         aux_subset = {k: v[idx].numpy() for k, v in aux_labels.items()}
#     else:
#         data_subset = data
#         aux_subset = {k: v.numpy() for k, v in aux_labels.items()}

#     # Get latent representations in batches
#     print("🔍 Extracting latent representations...")
#     all_z1, all_z2, all_z3 = [], [], []
#     all_alpha = []
    
#     batch_size = 1000
#     with torch.no_grad():
#         for i in range(0, len(data_subset), batch_size):
#             batch_data = data_subset[i:i+batch_size]
#             x = torch.FloatTensor(batch_data).to(device)
            
#             (mu1, logvar1, z1), (mu2, logvar2, z2), (z3, alpha, archetypes, entropy) = model.encode(x)
            
#             all_z1.append(z1.cpu().numpy())
#             all_z2.append(z2.cpu().numpy())
#             all_z3.append(z3.cpu().numpy())
#             all_alpha.append(alpha.cpu().numpy())
    
#     z1 = np.vstack(all_z1)
#     z2 = np.vstack(all_z2)
#     z3 = np.vstack(all_z3)
#     alpha_matrix = np.vstack(all_alpha)
    
#     print(f"✅ Extracted: z1={z1.shape}, z2={z2.shape}, z3={z3.shape}, alpha={alpha_matrix.shape}")
    
#     # Use PCA for visualization
#     print("📉 Computing PCA projections...")
#     pca_z1 = PCA(n_components=2, random_state=42).fit_transform(z1)
#     pca_z2 = PCA(n_components=2, random_state=42).fit_transform(z2)
#     pca_z3 = PCA(n_components=2, random_state=42).fit_transform(z3)
#     print("✅ PCA complete")
    
#     # --- Define cell populations for coloring ---
#     lineage_colors = np.zeros(len(data_subset))
#     lineage_colors[aux_subset['lineage_is_tcell'] > 0.5] = 1
#     lineage_colors[aux_subset['lineage_is_bcell'] > 0.5] = 2
#     lineage_colors[aux_subset['lineage_is_myeloid'] > 0.5] = 3
#     lineage_colors[aux_subset['lineage_is_nk'] > 0.5] = 4
#     lineage_names = ['Unknown', 'T-cells', 'B-cells', 'Myeloid', 'NK']
#     lineage_cmap = ['lightgray', 'red', 'blue', 'green', 'purple']
    
#     # --- Archetype coloring ---
#     dominant_archetype = alpha_matrix.argmax(axis=1)
#     purity = alpha_matrix.max(axis=1)
#     n_archetypes = alpha_matrix.shape[1]
    
#     # --- Create Figure ---
#     print("🎨 Creating plots...")
#     fig = plt.figure(figsize=(20, 12))
    
#     # Plot z1 (Viability) by Lineage
#     ax = plt.subplot(2, 3, 1)
#     for i, (name, color) in enumerate(zip(lineage_names, lineage_cmap)):
#         mask = lineage_colors == i
#         if mask.sum() > 0:
#             ax.scatter(pca_z1[mask, 0], pca_z1[mask, 1], c=color, s=1, alpha=0.5, label=f'{name} ({mask.sum()})')
#     ax.set_title('z1 (Viability Space) by Lineage', fontsize=12, fontweight='bold')
#     ax.legend(markerscale=3, fontsize=8)
    
#     # Plot z2 (Lineage Space) by Lineage
#     ax = plt.subplot(2, 3, 2)
#     for i, (name, color) in enumerate(zip(lineage_names, lineage_cmap)):
#         mask = lineage_colors == i
#         if mask.sum() > 0:
#             ax.scatter(pca_z2[mask, 0], pca_z2[mask, 1], c=color, s=1, alpha=0.5, label=f'{name} ({mask.sum()})')
#     ax.set_title('z2 (Lineage Space) by Lineage', fontsize=12, fontweight='bold')
#     ax.legend(markerscale=3, fontsize=8)

#     # Plot z3 (Archetype Space) by Dominant Archetype
#     ax = plt.subplot(2, 3, 3)
    
#     # --- FIX 1: Get the color list from Seaborn ---
#     color_list = sns.color_palette('husl', n_archetypes)
    
#     for i in range(n_archetypes):
#         mask = dominant_archetype == i
#         if mask.sum() > 0:
#             # --- FIX 2: Index the color list ---
#             ax.scatter(pca_z3[mask, 0], pca_z3[mask, 1], color=color_list[i], s=1, alpha=0.5, label=f'Archetype {i} ({mask.sum()})')
    
#     ax.set_title('z3 (Archetype Space) by Dominant Archetype', fontsize=12, fontweight='bold')
#     ax.legend(markerscale=3, fontsize=8, ncol=2)

#     # Plot z2 (Lineage Space) by Purity
#     ax = plt.subplot(2, 3, 4)
#     sc = ax.scatter(pca_z2[:, 0], pca_z2[:, 1], c=purity, cmap='viridis', s=1, alpha=0.5, vmin=0, vmax=1)
#     plt.colorbar(sc, ax=ax, label='Archetype Purity')
#     ax.set_title('z2 (Lineage Space) by Archetype Purity', fontsize=12, fontweight='bold')

#     # Plot z3 (Archetype Space) by Purity
#     ax = plt.subplot(2, 3, 5)
#     sc = ax.scatter(pca_z3[:, 0], pca_z3[:, 1], c=purity, cmap='viridis', s=1, alpha=0.5, vmin=0, vmax=1)
#     plt.colorbar(sc, ax=ax, label='Archetype Purity')
#     ax.set_title('z3 (Archetype Space) by Archetype Purity', fontsize=12, fontweight='bold')
    
#     # Plot Archetype Usage
#     ax = plt.subplot(2, 3, 6)
#     usage = np.bincount(dominant_archetype, minlength=n_archetypes)
    
#     # --- FIX 3: Use the color list here as well ---
#     ax.bar(range(n_archetypes), usage, color=[color_list[i] for i in range(n_archetypes)])
    
#     ax.set_title('Dominant Archetype Counts', fontsize=12, fontweight='bold')
#     ax.set_xlabel('Archetype Index')
#     ax.set_ylabel('Number of Cells')
#     ax.set_xticks(range(n_archetypes))

#     plt.tight_layout()
#     plt.savefig('vae_validation_archetypes.png', dpi=100, bbox_inches='tight')
#     print("\n✅ Plot saved: vae_validation_archetypes.png")
#     plt.show()

#     return fig

# # --- 3. Re-create Auxiliary Labels ---
# # Aux labels are generated from the data (using the function in Cell 18/24).
# print("Generating auxiliary labels...")
# # We need the ExperimentSweepAdvanced class just to get the label creation method
# label_generator = ExperimentSweepAdvanced(config) 
# aux_labels = label_generator._create_biology_based_aux_labels(X_train)
# print(f"Generated {len(aux_labels)} auxiliary label sets.")


# # --- 4. Load the Trained Model ---
# # This uses your load_trained_model_safe function from Cell 22.
# print("Loading trained model...")
# experiment_path = 'experiments_full/k30_arch_xlarge/best_model.pt' # Path to your best model

# safe_validate_vae_with_archetypes(model = model, data = X_train, aux_labels = aux_labels, config=config)

# # Run complete sweep (this will take hours!)
# RUN_FULL_SWEEP = True  # Set to True to run all 24 experiments

# if RUN_FULL_SWEEP:
#     sweep = ExperimentSweepAdvanced(config, base_save_dir='./experiments_full')
    
#     results = sweep.run_full_sweep(
#         X_train,
#         n_epochs=50,
#         device=device
#     )
    
#     # Visualize results
#     plot_sweep_results('./experiments_full/sweep_results.csv')
# else:
#     print("Set RUN_FULL_SWEEP=True to run all 24 experiments")
#     print("\nExperiment sweep configuration:")
#     print(f"  k_neighbors: {sweep.k_neighbors_sweep}")
#     print(f"  k_archetypes: {list(sweep.k_archetypes_sweep.keys())}")
#     print(f"  Total: {len(sweep.k_neighbors_sweep) * len(sweep.k_archetypes_sweep)} experiments")
