
"""
Synthetic Refinery Unit Generator + PIML/PINN demo (anonymized, simulated data only)

What you get (single-run):
1) Synthetic unit topology from P&ID-like connectivity (equipment, piping edges)
2) TSV exports:
   - unit_topology_anonymized.tsv
   - equipment_anonymized.tsv
   - piping_attributes_anonymized.tsv
   - scc_alternatives_100.tsv
3) Interactive 3D HTML:
   - synthetic_refinery_unit_3d.html
   Includes: tanks, vessels, columns, exchangers, pumps, racks, colored piping,
   hover tooltips, and "crack overlays" on selected high-risk lines.

4) PIML/PINN model demo for cracking corrosion (screening-level):
   - Predicts SCC susceptibility (0-1)
   - Predicts crack depth growth a(t) using physics-informed constraints:
       da/dN = C * (ΔK)^m (Paris-type surrogate)
     with ML residual for unmodeled effects.
   - Saves trained weights to: pinn_checkpoint.pt

Important:
- This is NOT a certified engineering calculator.
- It is a GitHub-shareable demo intended to illustrate how PIML/PINN can be
  structured for integrity analytics workflows aligned with common API/ASME
  concepts (materials, design/operating conditions, damage mechanisms, RBI).
"""

from __future__ import annotations

import math
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Optional ML (PINN/PIML) dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    optim = None

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------
# Output paths
# ----------------------------
OUT_DIR = Path(__file__).resolve().parent
OUT_HTML = OUT_DIR / "synthetic_refinery_unit_3d.html"
OUT_TOPO = OUT_DIR / "unit_topology_anonymized.tsv"
OUT_EQUIP = OUT_DIR / "equipment_anonymized.tsv"
OUT_PIPES = OUT_DIR / "piping_attributes_anonymized.tsv"
OUT_SCC = OUT_DIR / "scc_alternatives_100.tsv"
OUT_PINN = OUT_DIR / "pinn_checkpoint.pt"

# ----------------------------
# Synthetic standards metadata (labels only)
# ----------------------------
# Keep these as references in metadata for GitHub/readme usage.
STANDARDS = {
    "RBI": ["API 580", "API 581"],
    "Piping inspection / integrity": ["API 570", "API 571"],
    "Fitness-for-service": ["API 579-1/ASME FFS-1"],
    "Pressure design": ["ASME B31.3", "ASME Section VIII (conceptual)"],
    "SCC guidance": ["API 571 (SCC mechanisms, screening concepts)"],
}

# ----------------------------
# Domain vocab and defaults (simulated)
# ----------------------------
EQUIP_TYPES = [
    "TANK", "COLUMN", "VESSEL", "DRUM", "SEPARATOR",
    "PUMP", "EXCHANGER", "COOLER", "HEATER", "COMPRESSOR",
]

PIPING_CLASSES = ["CLS-A", "CLS-B", "CLS-C", "CLS-D", "CLS-E"]
WELD_TYPES = ["BW", "SW", "FW"]
INSULATION = ["None", "Mineral Wool", "Calcium Silicate"]
COATING_EXT = ["None", "Epoxy", "PU"]
COATING_INT = ["None", "PTFE Lining"]
MATERIAL_SPEC = ["A106", "A312", "A333", "A358"]
MATERIAL_GRADE = ["Gr.B", "TP316/316L", "TP304/304L", "Gr.6", "Gr.3"]

# Damage mechanisms and simplified grouping for RBI style features
DM_INTERNAL = [
    "Cl-SCC", "Caustic SCC", "Amine SCC", "Sulfide Stress Cracking",
    "HIC/SOHIC", "MIC", "CO2 Corrosion", "H2S Corrosion", "Erosion-Corrosion",
]
DM_EXTERNAL = ["Atmospheric Corrosion", "CUI", "External SCC", "Soil Corrosion"]

SERVICES = [
    ("Wet Sour Water", 0.22),
    ("Hydrocarbon + Water", 0.25),
    ("Lean Amine", 0.08),
    ("Rich Amine", 0.08),
    ("Condensate", 0.12),
    ("Steam Condensate", 0.10),
    ("Utility Water", 0.10),
    ("Fuel Gas", 0.05),
]

# ----------------------------
# Utility helpers
# ----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def pick_weighted(items: List[Tuple[str, float]]) -> str:
    r = random.random()
    cum = 0.0
    for name, w in items:
        cum += w
        if r <= cum:
            return name
    return items[-1][0]

def hash_color(key: str) -> str:
    """Deterministic vibrant-ish color from a string key."""
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # boost saturation: mix with 128
    r = int((r + 128) / 2)
    g = int((g + 128) / 2)
    b = int((b + 128) / 2)
    return f"rgb({r},{g},{b})"

def mm_per_year_from_mpy(mpy: float) -> float:
    # 1 mil = 0.001 inch; 1 inch = 25.4 mm
    return float(mpy) * 0.001 * 25.4

# ----------------------------
# Equipment and layout
# ----------------------------
@dataclass
class Equip:
    tag: str
    etype: str
    name: str
    x: float
    y: float
    z: float
    height: float
    radius: float

def _new_tag(prefix: str, i: int) -> str:
    return f"{prefix}-{i:04d}"

def place_equipment(n: int = 55) -> pd.DataFrame:
    rows: List[Dict] = []
    for i in range(1, n + 1):
        etype = random.choices(
            EQUIP_TYPES,
            weights=[10, 9, 9, 6, 6, 14, 8, 5, 4, 4],
            k=1
        )[0]
        tag = _new_tag("EQ", i)
        name = f"{etype} {i}"
        x = random.uniform(0, 340)
        y = random.uniform(0, 260)
        z = 0.0

        # simple geometry
        if etype in {"TANK", "COLUMN", "VESSEL", "DRUM", "SEPARATOR"}:
            height = random.uniform(10, 35) if etype != "TANK" else random.uniform(12, 22)
            radius = random.uniform(3.5, 8.0) if etype != "COLUMN" else random.uniform(2.5, 5.0)
        elif etype in {"EXCHANGER", "COOLER", "HEATER"}:
            height = random.uniform(3, 6)
            radius = random.uniform(1.5, 3.0)
        else:  # PUMP, COMPRESSOR
            height = random.uniform(2.0, 4.0)
            radius = random.uniform(1.2, 2.2)

        rows.append(
            dict(tag=tag, etype=etype, name=name, x=x, y=y, z=z, height=height, radius=radius)
        )
    return pd.DataFrame(rows)

def build_pipe_rack(n_racks: int = 4) -> pd.DataFrame:
    racks = []
    for i in range(n_racks):
        x0 = random.uniform(30, 80) + i * random.uniform(60, 90)
        y0 = random.uniform(40, 70)
        length = random.uniform(220, 320)
        width = random.uniform(18, 30)
        racks.append(
            dict(
                rack=f"RACK-{i+1}",
                x0=x0,
                x1=x0 + length,
                y0=y0,
                y1=y0 + width,
                z_base=8.0,
            )
        )
    return pd.DataFrame(racks)

def nearest_rack_point(racks: pd.DataFrame, p: Tuple[float, float]) -> Tuple[float, float, str]:
    x, y = p
    best = None
    best_d = 1e18
    best_r = None
    for _, r in racks.iterrows():
        cx = clamp(x, float(r["x0"]), float(r["x1"]))
        cy = clamp(y, float(r["y0"]), float(r["y1"]))
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_d:
            best_d = d
            best = (cx, cy)
            best_r = str(r["rack"])
    return float(best[0]), float(best[1]), best_r

def orthogonal_route(
    racks: pd.DataFrame,
    p_src: Tuple[float, float, float],
    p_dst: Tuple[float, float, float],
    z_rack: float,
) -> Tuple[List[Tuple[float, float, float]], str]:
    xs, ys, zs = p_src
    xd, yd, zd = p_dst
    rsx, rsy, rack = nearest_rack_point(racks, (xs, ys))
    rdx, rdy, _ = nearest_rack_point(racks, (xd, yd))
    poly = [
        (xs, ys, zs),
        (xs, ys, z_rack),
        (rsx, rsy, z_rack),
        (rdx, rdy, z_rack),
        (xd, yd, z_rack),
        (xd, yd, zd),
    ]
    return poly, rack

# ----------------------------
# Piping attributes and scenarios
# ----------------------------
def make_pipe_attrs(service: str) -> Dict:
    # design/operating: screening-level ranges
    op_kpag = random.uniform(50, 2500)
    dp_kpag = op_kpag + random.uniform(200, 1200)
    ot_c = random.uniform(20, 180)
    dt_c = ot_c + random.uniform(5, 80)

    material_spec = random.choice(MATERIAL_SPEC)
    material_grade = random.choice(MATERIAL_GRADE)
    piping_class = random.choice(PIPING_CLASSES)

    dn_in = random.choice([1, 2, 3, 4, 6, 8, 10, 12, 14, 16])
    velocity = random.uniform(0.1, 3.5)
    water_cut = random.uniform(0.0, 0.5)

    # chemistry
    chloride = int(max(0, np.random.normal(35, 60)))
    ph = float(clamp(np.random.normal(6.2, 1.0), 3.0, 9.5))
    h2s_ppm = int(random.choice([0, 50, 200, 500, 1000, 5000]))

    insulation_thk = int(random.choice([0, 25, 50, 75]))
    ins_mat = random.choice(INSULATION) if insulation_thk > 0 else "None"
    ext_coat = random.choice(COATING_EXT)
    int_coat = random.choice(COATING_INT)
    pwht = random.choice(["Y", "N"])
    weld = random.choice(WELD_TYPES)

    # "damage mechanism as per corrosion study" label
    dm = random.choices(
        DM_INTERNAL + DM_EXTERNAL,
        weights=[6] * len(DM_INTERNAL) + [4] * len(DM_EXTERNAL),
        k=1
    )[0]

    # corrosion rates
    base_cr_mpy = abs(np.random.normal(1.2, 0.8))
    # service modifiers
    if "Sour" in service or "Amine" in service:
        base_cr_mpy *= 1.3
    if service == "Fuel Gas":
        base_cr_mpy *= 0.25
    if service == "Utility Water":
        base_cr_mpy *= 0.8

    stcr = base_cr_mpy * random.uniform(0.7, 1.3)
    ltcr = base_cr_mpy * random.uniform(0.5, 1.1)

    # SCC screening score proxy (0-1), used for crack placement and ML targets
    scc_like = 0.0
    if dm in {"Cl-SCC", "External SCC", "Caustic SCC", "Amine SCC", "Sulfide Stress Cracking"}:
        scc_like = (
            0.35 * clamp(chloride / 500.0, 0, 1) +
            0.35 * clamp((ot_c - 40) / 90.0, 0, 1) +
            0.20 * clamp((7.0 - ph) / 3.0, 0, 1) +
            0.10 * random.random()
        )
        scc_like = clamp(scc_like, 0, 1)

    return {
        "Piping Class": piping_class,
        "Material Spec (Main Lines)": material_spec,
        "Material Grade (Main Lines)": material_grade,
        "Design Pressure (KPag)": round(dp_kpag, 1),
        "Design Temperature (deg. C)": round(dt_c, 1),
        "Operating Pressure (KPag)": round(op_kpag, 1),
        "Operating Temperature (deg. C)": round(ot_c, 1),
        "Nominal Size (in)": dn_in,
        "Velocity (m/s)": round(velocity, 3),
        "Water cut (0-1)": round(water_cut, 3),
        "Chloride (mg/L)": chloride,
        "pH": round(ph, 2),
        "H2S (ppmv)": h2s_ppm,
        "Insulation Type/Thick (mm)": insulation_thk,
        "Insulation Material": ins_mat,
        "Internal Coating/Cladding": int_coat,
        "External Coating/Cladding": ext_coat,
        "PWHT": pwht,
        "Weld Type": weld,
        "Damage Mechanism as per Corrosion Study": dm,
        "Max STCR (MPY)": round(stcr, 3),
        "Max LTCR (MPY)": round(ltcr, 3),
        "Expected CR  Based on CCM (mpy)": round(base_cr_mpy, 3),
        "SCC Susceptibility (0-1)": round(float(scc_like), 3),
        "Service": service,
        "Standards (labels)": ", ".join(STANDARDS["RBI"] + STANDARDS["Piping inspection / integrity"]),
    }

def simulate_scc_alternatives(n: int = 100) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        service = pick_weighted(SERVICES)
        attrs = make_pipe_attrs(service)
        # force some SCC cases to exist for training
        if i <= int(0.35 * n):
            attrs["Damage Mechanism as per Corrosion Study"] = "Cl-SCC"
            # push chloride and temp higher
            attrs["Chloride (mg/L)"] = int(clamp(attrs["Chloride (mg/L)"] + random.randint(150, 450), 0, 900))
            attrs["Operating Temperature (deg. C)"] = float(clamp(attrs["Operating Temperature (deg. C)"] + random.uniform(20, 70), 10, 220))
            attrs["pH"] = float(clamp(attrs["pH"] - random.uniform(0.3, 1.5), 3.0, 9.5))
            # recompute SCC proxy
            chlor = float(attrs["Chloride (mg/L)"])
            ot = float(attrs["Operating Temperature (deg. C)"])
            ph = float(attrs["pH"])
            scc = 0.35 * clamp(chlor / 500.0, 0, 1) + 0.35 * clamp((ot - 40) / 90.0, 0, 1) + 0.2 * clamp((7.0 - ph) / 3.0, 0, 1) + 0.1 * random.random()
            attrs["SCC Susceptibility (0-1)"] = round(clamp(scc, 0, 1), 3)

        attrs["Scenario"] = f"SCC_ALT_{i:03d}"
        attrs["Piping or Vessel"] = "Piping"
        attrs["Notes"] = "Synthetic scenario for demo and model testing."
        rows.append(attrs)
    return pd.DataFrame(rows)

# ----------------------------
# 3D geometry builders
# ----------------------------
def _basis_from_axis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    if abs(axis[2]) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, up)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(axis, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v

def add_cylinder_mesh(
    fig: go.Figure,
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    radius: float,
    color: str,
    opacity: float,
    hover: str | None = None,
):
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-6:
        return
    axis = v / (L + 1e-12)
    u, w = _basis_from_axis(axis)

    ntheta = 18
    angles = np.linspace(0, 2 * math.pi, ntheta, endpoint=False)
    ring0 = [p0 + radius * (math.cos(a) * u + math.sin(a) * w) for a in angles]
    ring1 = [p1 + radius * (math.cos(a) * u + math.sin(a) * w) for a in angles]

    xs, ys, zs = [], [], []
    for pt in ring0 + ring1:
        xs.append(float(pt[0])); ys.append(float(pt[1])); zs.append(float(pt[2]))

    ii, jj, kk = [], [], []
    for t in range(ntheta):
        a0 = t
        a1 = (t + 1) % ntheta
        b0 = t + ntheta
        b1 = ((t + 1) % ntheta) + ntheta
        ii += [a0, a1]
        jj += [b0, b1]
        kk += [a1, b0]

    fig.add_trace(
        go.Mesh3d(
            x=xs, y=ys, z=zs,
            i=ii, j=jj, k=kk,
            color=color,
            opacity=opacity,
            hovertext=hover,
            hoverinfo="text" if hover else "skip",
            showlegend=False,
        )
    )

def add_vertical_vessel(fig: go.Figure, x: float, y: float, z: float, height: float, radius: float, color: str, hover: str):
    add_cylinder_mesh(fig, (x, y, z), (x, y, z + height), radius=radius, color=color, opacity=0.55, hover=hover)

def add_horizontal_exchanger(fig: go.Figure, x: float, y: float, z: float, length: float, radius: float, color: str, hover: str):
    add_cylinder_mesh(fig, (x - length / 2, y, z + 1.0), (x + length / 2, y, z + 1.0), radius=radius, color=color, opacity=0.55, hover=hover)

def add_pump_block(fig: go.Figure, x: float, y: float, z: float, size: float, color: str, hover: str):
    # simple box via mesh (8 vertices)
    s = size
    verts = np.array([
        [x - s, y - s, z],
        [x + s, y - s, z],
        [x + s, y + s, z],
        [x - s, y + s, z],
        [x - s, y - s, z + 2*s],
        [x + s, y - s, z + 2*s],
        [x + s, y + s, z + 2*s],
        [x - s, y + s, z + 2*s],
    ], float)
    faces = [
        (0, 1, 2), (0, 2, 3),  # bottom
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]
    i, j, k = zip(*faces)
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=list(i), j=list(j), k=list(k),
            color=color, opacity=0.65,
            hovertext=hover, hoverinfo="text",
            showlegend=False,
        )
    )

def crack_curve(poly: List[Tuple[float, float, float]], crack_len: float = 14.0, npts: int = 60) -> List[Tuple[float, float, float]]:
    if len(poly) < 3:
        return []
    seg = random.randint(0, len(poly) - 2)
    p0 = np.array(poly[seg], float)
    p1 = np.array(poly[seg + 1], float)
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-6:
        return []
    t0 = random.uniform(0.2, 0.8)
    start = p0 + t0 * v
    d = v / (L + 1e-12)
    u, _ = _basis_from_axis(d)
    s = np.linspace(0, 1, npts)
    amp = 0.9
    pts = []
    for si in s:
        pt = start + (si * crack_len) * d + amp * math.sin(6 * math.pi * si) * u
        pts.append((float(pt[0]), float(pt[1]), float(pt[2])))
    return pts

# ----------------------------
# PIML / PINN model demo
# ----------------------------
def _require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch not installed. Install it to run the PINN demo, or run with --no-ml."
        )

class FeatureScaler:
    def __init__(self):
        self.mu = None
        self.sig = None

    def fit(self, X: np.ndarray):
        self.mu = X.mean(axis=0)
        self.sig = X.std(axis=0) + 1e-12

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sig

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

def build_ml_dataset(scc_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Minimal features that map to typical corrosion study inputs and IOWs
    feat_cols = [
        "Operating Temperature (deg. C)",
        "Operating Pressure (KPag)",
        "Nominal Size (in)",
        "Velocity (m/s)",
        "Water cut (0-1)",
        "Chloride (mg/L)",
        "pH",
        "H2S (ppmv)",
        "Expected CR  Based on CCM (mpy)",
    ]
    X = scc_df[feat_cols].astype(float).values
    y = scc_df["SCC Susceptibility (0-1)"].astype(float).values.reshape(-1, 1)
    return X, y, feat_cols

class PIML_PINN(nn.Module):
    """
    Two-head network:
      - head_susc: SCC susceptibility (0-1)
      - head_phys: outputs (logC, m, resid_scale) for Paris-type surrogate
    Physics:
      a_{t+1} = a_t + da/dN * dN,  da/dN = exp(logC) * (ΔK)^m + ML_residual
    """
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head_susc = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.head_phys = nn.Linear(hidden, 3)

    def forward(self, x):
        h = self.backbone(x)
        susc = self.head_susc(h)
        phys = self.head_phys(h)
        logC = phys[:, 0:1]
        m = 1.0 + torch.nn.functional.softplus(phys[:, 1:2])  # m >= 1
        resid_scale = torch.nn.functional.softplus(phys[:, 2:3])  # >= 0
        return susc, logC, m, resid_scale

def deltaK_surrogate(op_kpag_t: torch.Tensor, dn_in_t: torch.Tensor) -> torch.Tensor:
    """
    Screening surrogate for stress intensity factor range:
      ΔK ~ sigma * sqrt(pi*a) is the true structure, but we do not know sigma/a in detail.
    Here we approximate sigma scaling with pressure and size.
    This is for ML demo only.
    """
    p_mpa = op_kpag_t / 1000.0  # crude
    d_m = (dn_in_t * 0.0254)  # inch to meter
    sigma = 12.0 * p_mpa * (d_m / 0.2).clamp(0.4, 2.5)  # arbitrary scaling
    # return sigma-like term (a will be embedded in the ODE rollout)
    return sigma.clamp(1e-3, 1e6)

def simulate_crack_growth_torch(
    model: PIML_PINN,
    x: torch.Tensor,
    a0_mm: float = 0.5,
    years: float = 5.0,
    cycles_per_year: float = 2.0e5,
    n_steps: int = 60,
) -> torch.Tensor:
    """
    Returns crack depth trajectory a(t) in mm (batch, n_steps).
    Enforces monotonicity by clamping da >= 0.
    """
    susc, logC, m, resid_scale = model(x)
    # pick key columns for ΔK surrogate by index
    # columns: OT, OP, DN, V, WC, Cl, pH, H2S, CR
    op = x[:, 1:2]
    dn = x[:, 2:3]
    DK = deltaK_surrogate(op, dn)

    dt_year = years / (n_steps - 1)
    dN = cycles_per_year * dt_year
    a = torch.full((x.shape[0], 1), float(a0_mm), device=x.device)
    traj = [a]

    # ML residual is a small correction scaled by susceptibility
    for _ in range(n_steps - 1):
        # embed a in DK via sqrt(a) style scaling
        DK_eff = DK * torch.sqrt(torch.clamp(a / 10.0, 1e-6, 1e3))  # a in mm scaled to ~cm
        dadN = torch.exp(logC) * torch.pow(torch.clamp(DK_eff, 1e-6, 1e6), m)
        # residual term: limited magnitude
        resid = resid_scale * (susc - 0.5)
        dadN = torch.clamp(dadN + 0.05 * resid, 0.0, 1e6)
        a = a + dadN * dN
        a = torch.clamp(a, 0.0, 50.0)
        traj.append(a)

    return torch.cat(traj, dim=1)

def train_pinn_demo(
    scc_df: pd.DataFrame,
    epochs: int = 450,
    lr: float = 2e-3,
    device: str = "cpu",
) -> Dict:
    _require_torch()

    X, y, feat_cols = build_ml_dataset(scc_df)

    scaler = FeatureScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)
    y = y.astype(np.float32)

    xt = torch.tensor(Xs, device=device)
    yt = torch.tensor(y, device=device)

    model = PIML_PINN(in_dim=xt.shape[1], hidden=72).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    # Create a synthetic "target" crack depth after 5y based on susceptibility, temp, chloride
    # This is still simulated to enable training without real inspection crack sizing data.
    ot = X[:, 0]
    cl = X[:, 5]
    ph = X[:, 6]
    # crack depth mm after 5y (synthetic)
    a5 = 0.6 + 8.5 * y[:, 0] + 0.006 * cl + 0.02 * np.maximum(0, ot - 60) + 0.4 * np.maximum(0, 6.5 - ph)
    a5 = np.clip(a5, 0.6, 25.0).astype(np.float32)
    a5t = torch.tensor(a5.reshape(-1, 1), device=device)

    # Physics regularization targets (range constraints)
    # Encourage m in [2,6] and C in a plausible log range for the surrogate.
    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        susc, logC, m, resid = model(xt)

        # data loss: susceptibility proxy
        loss_s = torch.mean((susc - yt) ** 2)

        # rollout crack growth and match terminal depth target
        traj = simulate_crack_growth_torch(model, xt, years=5.0, n_steps=50)
        aT = traj[:, -1:].detach() * 0.0 + traj[:, -1:]  # keep graph
        loss_a = torch.mean((aT - a5t) ** 2)

        # physics-informed penalties
        loss_m = torch.mean(torch.relu(2.0 - m) ** 2 + torch.relu(m - 6.0) ** 2)
        loss_c = torch.mean(torch.relu(-10.0 - logC) ** 2 + torch.relu(logC - 2.0) ** 2)
        loss_res = torch.mean(resid ** 2)

        loss = loss_s + 0.35 * loss_a + 0.05 * loss_m + 0.02 * loss_c + 0.01 * loss_res
        loss.backward()
        opt.step()

    # save
    torch.save(
        {"state_dict": model.state_dict(), "mu": scaler.mu, "sig": scaler.sig, "feat_cols": feat_cols},
        str(OUT_PINN),
    )
    return {"checkpoint": str(OUT_PINN), "feat_cols": feat_cols}

# ----------------------------
# Main synthetic unit builder
# ----------------------------
def build_unit() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
    equip_df = place_equipment(n=60)
    racks_df = build_pipe_rack(n_racks=4)
    equip_pos = equip_df.set_index("tag")[["x", "y", "z"]].to_dict("index")

    # Build connectivity roughly like a unit:
    tanks = equip_df[equip_df["etype"] == "TANK"]["tag"].tolist()
    columns = equip_df[equip_df["etype"] == "COLUMN"]["tag"].tolist()
    pumps = equip_df[equip_df["etype"] == "PUMP"]["tag"].tolist()
    exch = equip_df[equip_df["etype"].isin(["EXCHANGER", "COOLER"])]["tag"].tolist()
    vess = equip_df[equip_df["etype"].isin(["VESSEL", "DRUM", "SEPARATOR"])]["tag"].tolist()
    heaters = equip_df[equip_df["etype"] == "HEATER"]["tag"].tolist()
    comps = equip_df[equip_df["etype"] == "COMPRESSOR"]["tag"].tolist()

    def sample(lst: List[str], k: int) -> List[str]:
        if len(lst) <= k:
            return lst[:]
        return random.sample(lst, k)

    edges: List[Tuple[str, str, str]] = []

    for t in tanks:
        for p in sample(pumps, k=2):
            edges.append((t, p, "TRANSFER"))
    for p in pumps:
        if random.random() < 0.65 and exch:
            edges.append((p, random.choice(exch), "PUMP_OUT"))
        else:
            edges.append((p, random.choice(heaters) if heaters else random.choice(vess), "PUMP_OUT"))
    for e in exch + heaters:
        if columns and random.random() < 0.55:
            edges.append((e, random.choice(columns), "HEAT_XFER"))
        else:
            edges.append((e, random.choice(vess), "HEAT_XFER"))
    for c in columns:
        for v in sample(vess, k=min(3, len(vess))):
            edges.append((c, v, "COLUMN_OUT"))
    for v in sample(vess, k=min(14, len(vess))):
        edges.append((v, random.choice(pumps), "RECYCLE"))
    for ktag in comps:
        edges.append((ktag, random.choice(vess), "GAS_OUT"))
        edges.append((random.choice(vess), ktag, "GAS_IN"))

    edges = [e for e in edges if e[0] != e[1]]
    # preserve order while removing duplicates
    edges = list(dict.fromkeys(edges))

    topo_rows = []
    pipe_rows = []
    pipes: List[Dict] = []
    line_counter = 1

    for src, dst, role in edges:
        ps = (float(equip_pos[src]["x"]), float(equip_pos[src]["y"]), float(equip_pos[src]["z"]))
        pdst = (float(equip_pos[dst]["x"]), float(equip_pos[dst]["y"]), float(equip_pos[dst]["z"]))
        z_rack = random.choice([10.0, 18.0, 26.0])

        poly, rack = orthogonal_route(racks_df, ps, pdst, z_rack=z_rack)
        service = pick_weighted(SERVICES)
        attrs = make_pipe_attrs(service)

        line_tag = f"L-{line_counter:05d}"
        line_counter += 1

        topo_rows.append(
            {
                "Circuit /Equipment Tag": line_tag,
                "Circuit From": src,
                "Circuit to": dst,
                "Equipment Name": "",
                "Piping or Vessel": "Piping",
            }
        )
        pipe_row = {"line_tag": line_tag, "src": src, "dst": dst, "role": role, "rack": rack, **attrs}
        pipe_rows.append(pipe_row)

        pipes.append({"line_tag": line_tag, "poly": poly, **pipe_row})

    topo_df = pd.DataFrame(topo_rows)
    pipes_df = pd.DataFrame(pipe_rows)
    scc_df = simulate_scc_alternatives(100)

    return topo_df, equip_df, pipes_df, scc_df, pipes

# ----------------------------
# 3D renderer
# ----------------------------
def render_3d(equip_df: pd.DataFrame, pipes_df: pd.DataFrame, pipes: List[Dict]) -> None:
    fig = go.Figure()

    # Render equipment as shapes by type
    for _, row in equip_df.iterrows():
        et = str(row["etype"])
        tag = str(row["tag"])
        name = str(row["name"])
        x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        h = float(row["height"])
        r = float(row["radius"])

        hover = f"{tag}<br>{name}<br>Type: {et}"
        if et in {"TANK", "COLUMN", "VESSEL", "DRUM", "SEPARATOR"}:
            col = "rgb(120,140,160)" if et != "COLUMN" else "rgb(100,120,200)"
            add_vertical_vessel(fig, x, y, z, height=h, radius=r, color=col, hover=hover)
        elif et in {"EXCHANGER", "COOLER", "HEATER"}:
            col = "rgb(170,120,90)" if et != "COOLER" else "rgb(120,170,150)"
            add_horizontal_exchanger(fig, x, y, z, length=12 + 1.2 * r, radius=0.85 * r, color=col, hover=hover)
        elif et == "PUMP":
            add_pump_block(fig, x, y, z, size=1.8, color="rgb(80,150,80)", hover=hover)
        else:  # COMPRESSOR
            add_pump_block(fig, x, y, z, size=2.2, color="rgb(150,80,150)", hover=hover)

    # Decide crack lines using SCC proxy and chloride
    crack = pipes_df.copy()
    crack["crack_score"] = crack["SCC Susceptibility (0-1)"].astype(float)
    crack["crack_score"] += (crack["Chloride (mg/L)"].astype(float).clip(0, 700) / 700.0) * 0.25
    crack = crack.sort_values("crack_score", ascending=False).head(35)
    crack_set = set(crack["line_tag"].astype(str).tolist())

    # Pipes: colored by circuit tag, hover contains all key attributes
    for p in pipes:
        poly = p["poly"]
        line_tag = str(p["line_tag"])
        color = hash_color(line_tag)
        dn = float(p["Nominal Size (in)"])
        radius = 0.28 + 0.03 * dn
        hover = (
            f"{line_tag}<br>From: {p['src']}<br>To: {p['dst']}<br>"
            f"Role: {p['role']}<br>Service: {p['Service']}<br>"
            f"DM: {p['Damage Mechanism as per Corrosion Study']}<br>"
            f"OP (kPag): {p['Operating Pressure (KPag)']}<br>"
            f"OT (C): {p['Operating Temperature (deg. C)']}<br>"
            f"Cl (mg/L): {p['Chloride (mg/L)']}<br>"
            f"pH: {p['pH']}<br>"
            f"SCC proxy: {p['SCC Susceptibility (0-1)']}<br>"
            f"Mat: {p['Material Spec (Main Lines)']} {p['Material Grade (Main Lines)']}<br>"
            f"Class: {p['Piping Class']}<br>"
            f"Standards: {p['Standards (labels)']}"
        )
        for a, b in zip(poly[:-1], poly[1:]):
            add_cylinder_mesh(fig, a, b, radius=radius, color=color, opacity=0.35, hover=hover)

        if line_tag in crack_set:
            cpts = crack_curve(poly)
            if cpts:
                fig.add_trace(
                    go.Scatter3d(
                        x=[q[0] for q in cpts],
                        y=[q[1] for q in cpts],
                        z=[q[2] for q in cpts],
                        mode="lines",
                        line=dict(width=7),
                        hovertext=f"{line_tag}<br>Crack overlay (synthetic)",
                        hoverinfo="text",
                        showlegend=False,
                    )
                )

    fig.update_layout(
        title="Synthetic Refinery Unit 3D (Anonymized, Simulated)",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn")

# ----------------------------
# CLI-like entry
# ----------------------------
def run(no_ml: bool = False) -> None:
    topo_df, equip_df, pipes_df, scc_df, pipes = build_unit()

    topo_df.to_csv(OUT_TOPO, sep="\t", index=False)
    equip_df.to_csv(OUT_EQUIP, sep="\t", index=False)
    pipes_df.to_csv(OUT_PIPES, sep="\t", index=False)
    scc_df.to_csv(OUT_SCC, sep="\t", index=False)

    render_3d(equip_df, pipes_df, pipes)

    if not no_ml:
        train_pinn_demo(scc_df, device="cpu")

if __name__ == "__main__":
    # If you do not want ML, set no_ml=True
    run(no_ml=False)
