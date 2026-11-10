"""
helpers.py — the behind-the-scenes code for Fossil_Explorer.ipynb.

Students never need to open this file.

Teachers: everything you may want to tune is in the SETTINGS block below.
Heavy steps (fossil latent optimization) are done ahead of time on a GPU with
teacher/precompute.py; this file only does fast lookups, plotting, and the
small "break a bone and fix it" shape completion, all on CPU.
"""
from __future__ import annotations

import csv
import inspect
import os
import random
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# =============================================================== SETTINGS ===
REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"
MODEL_DIR = DATA_DIR / "model"            # model_params_config.json, model/, latent_codes/, encoder/
FOSSIL_DIR = DATA_DIR / "fossils"         # fossil meshes (.vtk)
MODERN_DIR = DATA_DIR / "modern"          # modern vertebra meshes students can view / break
PRECOMPUTED_DIR = DATA_DIR / "precomputed"  # <fossil>_latent.pt made by teacher/precompute.py
LABELS_CSV = DATA_DIR / "labels.csv"      # friendly names for every training mesh
TMP_DIR = DATA_DIR / "tmp"                # broken / fixed bones students make

CKPT = "3000"
TOP_K = 5

# Mission 5 (fix a bone). Lower numbers = faster but rougher. Run
# `python -c "import helpers; helpers.benchmark()"` in a codespace to time it.
REFINE_ITERS = 200          # Colab demo used 200 on GPU; try 50–100 if too slow on CPU
RECON_RESOLUTION = 128      # Colab demo used 256; 96–128 is a reasonable CPU range
MAX_DISPLAY_FACES = 20_000  # meshes are simplified to this many triangles for display only

# Friendly names students type  ->  file name (no extension) in data/fossils/
FOSSILS = {
    "mystery_fossil": "UF271967",
    "broken_fossil": "uf546657",
    "broken_fossil_ai_fixed": "uf546657_shape_completion",
    "broken_fossil_ai_fixed_cleaned": "uf546657_fillholes_smooth_shape_completion",
}
FOSSIL_INFO = {
    "mystery_fossil": "A complete fossil vertebra. Nobody knows for sure what animal it came from!",
    "broken_fossil": "A broken fossil vertebra. Scientists identified it as Wautaugetegu, "
                     "an extinct relative of tegu lizards (Bourque & Stanley 2025).",
    "broken_fossil_ai_fixed": "The broken fossil after the AI rebuilt the missing pieces.",
    "broken_fossil_ai_fixed_cleaned": "The AI-rebuilt fossil after cleaning it up "
                                      "(filling small holes and smoothing).",
}
KNOWN_ANSWER = {"broken_fossil": "Wautaugetegu (an extinct tegu relative)"}

# Directions for break_bone(). These follow the MorphoWeave alignment axes;
# rename them if "top" doesn't look like the top of the bone on screen.
SIDES = {
    "top": (0, 0, 1), "bottom": (0, 0, -1),
    "left": (-1, 0, 0), "right": (1, 0, 0),
    "front": (0, 1, 0), "back": (0, -1, 0),
}

# Colors
FOSSIL_COLOR = "goldenrod"
MODERN_COLOR = "deepskyblue"
BROKEN_COLOR = "lightcoral"
FIXED_COLOR = "mediumseagreen"
# ============================================================================

_STATE: dict = {}


# --------------------------------------------------------------- utilities --
def _say(msg: str) -> None:
    print(msg, flush=True)


@contextmanager
def _cd(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class _Timer:
    def __init__(self):
        self.t0 = time.time()

    def lap(self) -> str:
        t, self.t0 = time.time() - self.t0, time.time()
        return f"{t:.0f} s" if t < 90 else f"{t / 60:.1f} min"


def _quiet_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="(?s).*algorithm.*extract_surface.*")


_quiet_warnings()  # keep student output clean from the first import


def _cpu_safety_net():
    """Make CUDA-saved checkpoints and stray .cuda() calls work on a CPU-only machine."""
    import torch

    if getattr(torch.load, "_fossil_patched", False):
        return
    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return _orig_load(*args, **kwargs)

    _load._fossil_patched = True
    torch.load = _load
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *a, **k: self
        torch.nn.Module.cuda = lambda self, *a, **k: self


def _patch_pymskt():
    """Same data-type monkey patches as the Colab demos."""
    try:
        import pymskt.mesh.meshes as meshes
        import pymskt.mesh.meshTools as meshTools
        from NSM.helper_funcs import fixed_point_coords, safe_load_mesh_scalars
    except Exception:
        return
    meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
    meshes.Mesh.point_coords = property(fixed_point_coords)
    if getattr(meshTools.pcu.signed_distance_to_mesh, "_fossil_patched", False):
        return
    _orig = meshTools.pcu.signed_distance_to_mesh

    def _sdf(pts, points, faces):
        return _orig(np.asarray(pts, dtype=np.float64),
                     np.asarray(points, dtype=np.float64),
                     np.asarray(faces, dtype=np.int32))

    _sdf._fossil_patched = True
    meshTools.pcu.signed_distance_to_mesh = _sdf


def _missing_data() -> list[str]:
    needed = [
        MODEL_DIR / "model_params_config.json",
        MODEL_DIR / "model" / f"{CKPT}.pth",
        MODEL_DIR / "latent_codes" / f"{CKPT}.pth",
        MODEL_DIR / "encoder" / "checkpoints" / "encoder.pt",
    ]
    needed += [FOSSIL_DIR / f"{stem}.vtk" for stem in FOSSILS.values()]
    needed += [PRECOMPUTED_DIR / f"{stem}_latent.pt" for stem in FOSSILS.values()]
    return [str(p.relative_to(REPO_DIR)) for p in needed if not p.exists()]


# ----------------------------------------------------------------- labels ---
def _guess_species(fname: str) -> str:
    stem = Path(fname).stem
    try:
        from NSM.helper_funcs import extract_species_prefix
        s = extract_species_prefix(stem)
        if isinstance(s, str) and s:
            return s.replace("_", " ")
    except Exception:
        pass
    return stem.replace("_", " ")


def _load_labels(names: list[str]) -> dict:
    labels = {n: {"species": _guess_species(n), "position": "", "common_name": "", "group": ""}
              for n in names}
    if LABELS_CSV.exists():
        with open(LABELS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get("file") or "").strip()
                if name in labels:
                    for key in ("species", "position", "common_name", "group"):
                        val = (row.get(key) or "").strip()
                        if val:
                            labels[name][key] = val
    return labels


def _describe(fname: str, html: bool = False) -> str:
    lab = _STATE.get("labels", {}).get(fname)
    if lab is None:
        return Path(fname).stem
    name = lab["common_name"] or lab["species"]
    if lab["common_name"] and lab["species"]:
        name += f" ({lab['species']})"
    if lab["position"]:
        name += (", " if not html else "<br>") + f"{lab['position']} vertebra"
    return name


# ------------------------------------------------------------------ setup ---
def setup(verbose: bool = True):
    """Load the AI model. Run once at the start of the notebook."""
    if _STATE.get("ready"):
        if verbose:
            _say("✅ The AI is already awake. You're good to go!")
        return
    t = _Timer()
    _quiet_warnings()
    if verbose:
        _say("🦎 Waking up the AI... (this takes about a minute)")

    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    _cpu_safety_net()

    try:
        from NSM.helper_funcs import load_config, load_model_and_latents
        from NSM.optimization import get_top_k_pcs
    except ImportError as e:
        raise RuntimeError(
            "😕 The AI toolkit (NSM) isn't installed in this codespace. "
            "Please ask your teacher for help.") from e
    _patch_pymskt()

    missing = _missing_data()
    if missing:
        raise RuntimeError(
            "😕 Some fossil data files are missing. Please ask your teacher for help.\n"
            "   Missing: " + ", ".join(missing))

    with _cd(MODEL_DIR):
        config = load_config(config_path="model_params_config.json")
        config["device"] = "cpu"
        model, _, latent_codes = load_model_and_latents(
            f"model/{CKPT}.pth", f"latent_codes/{CKPT}.pth", config, "cpu")
    model.eval()
    latent_codes = latent_codes.detach().cpu()

    _, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.99)
    names = [Path(p).name for p in config["list_mesh_paths"]]

    _STATE.update(
        ready=True,
        config=config,
        model=model,
        latent_codes=latent_codes,
        mean_latent=latent_codes.mean(dim=0, keepdim=True),
        latent_std=latent_codes.std().mean(),
        top_k_reg=top_k_reg,
        names=names,
        labels=_load_labels(names),
    )
    if verbose:
        _say(f"✅ The AI is awake! It has studied {len(names)} modern snake and lizard "
             f"vertebrae. (took {t.lap()})")


def _ensure_setup():
    if not _STATE.get("ready"):
        setup()


# ---------------------------------------------------------------- display ---
def _read(path):
    import pyvista as pv
    return pv.read(str(path))


def _to_trace(mesh, color: str, name: str):
    import plotly.graph_objects as go

    surf = mesh.extract_surface().triangulate()
    if surf.n_cells > MAX_DISPLAY_FACES:
        surf = surf.decimate(1 - MAX_DISPLAY_FACES / surf.n_cells)
    faces = surf.faces.reshape(-1, 4)
    pts = surf.points
    return go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        i=faces[:, 1], j=faces[:, 2], k=faces[:, 3],
        color=color, name=name, flatshading=False, hoverinfo="skip",
        lighting=dict(ambient=0.15, diffuse=0.85, specular=0.05, roughness=0.9, fresnel=0.0),
        lightposition=dict(x=0, y=0, z=2),
    )


def _show_side_by_side(items, title: str | None = None):
    """items: list of (mesh, label, color)."""
    from plotly.subplots import make_subplots

    n = len(items)
    fig = make_subplots(rows=1, cols=n, specs=[[{"type": "scene"}] * n],
                        subplot_titles=[label for _, label, _ in items],
                        horizontal_spacing=0.02)
    for col, (mesh, label, color) in enumerate(items, start=1):
        fig.add_trace(_to_trace(mesh, color, label), row=1, col=col)
    fig.update_scenes(aspectmode="data",
                      xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(height=430 if n > 1 else 520, showlegend=False,
                      margin=dict(l=0, r=0, t=90 if title else 60, b=0),
                      title=dict(text=title, x=0.5) if title else None)
    fig.show()
    if n == 1:
        _say("🖱️  Drag to spin it. Scroll to zoom.")
    else:
        _say("🖱️  Drag each bone to spin it. Scroll to zoom.")


# ---------------------------------------------------------------- fossils ---
def _fossil_key(name: str) -> str:
    if name in FOSSILS:
        return name
    for key, stem in FOSSILS.items():
        if name.lower() == stem.lower():
            return key
    raise ValueError(f"😕 I don't know a fossil called '{name}'. "
                     f"Try one of these: {', '.join(FOSSILS)}")


def list_fossils():
    """Print the fossils you can explore."""
    _say("🦴 Fossils you can explore:\n")
    for key in FOSSILS:
        _say(f"  • \"{key}\" — {FOSSIL_INFO.get(key, '')}")


def show_fossil(name: str = "mystery_fossil"):
    """Show a fossil in 3D."""
    key = _fossil_key(name)
    mesh = _read(FOSSIL_DIR / f"{FOSSILS[key]}.vtk")
    _show_side_by_side([(mesh, f"{key}<br><sub>specimen {FOSSILS[key]}</sub>", FOSSIL_COLOR)])
    _say(f"ℹ️  {FOSSIL_INFO.get(key, '')}")


# ---------------------------------------------------------------- matching --
class MatchList(list):
    """The AI's top matches. Behaves like a list of dicts."""

    def __init__(self, rows, query_label, latent, query_mesh=None, true_file=None):
        super().__init__(rows)
        self.query_label = query_label
        self.latent = latent
        self.query_mesh = query_mesh
        self.true_file = true_file

    def __repr__(self):
        return f"<{len(self)} matches for {self.query_label}>"


def _top_k(latent, k: int):
    import torch
    import torch.nn.functional as F

    codes = _STATE["latent_codes"]
    ids = None
    try:
        from NSM.optimization import find_similar_cos
        ids, _ = find_similar_cos(latent, codes, top_k=k, n_std=2, device="cpu")
        ids = [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]
    except Exception:
        ids = None
    sims_all = F.cosine_similarity(latent.reshape(1, -1).float(), codes.float(), dim=1)
    if not ids:  # fallback: plain cosine similarity
        ids = torch.topk(sims_all, k).indices.tolist()
    return ids, [float(sims_all[i]) for i in ids]


def find_matches(what="mystery_fossil", show: bool = True) -> MatchList:
    """Ask the AI which modern bones look most like a fossil (or a bone you fixed)."""
    import torch
    _ensure_setup()

    true_file = None
    query_mesh = None
    if isinstance(what, dict) and "latent" in what:          # a bone from fix_bone()
        latent = what["latent"]
        label = what.get("label", "your fixed bone")
        query_mesh = what.get("fixed_mesh")
        true_file = what.get("file")
    else:
        key = _fossil_key(what)
        stem = FOSSILS[key]
        latent = torch.load(PRECOMPUTED_DIR / f"{stem}_latent.pt")
        label = key
        query_mesh = FOSSIL_DIR / f"{stem}.vtk"
    latent = latent.detach().cpu().reshape(1, -1).float()

    ids, sims = _top_k(latent, TOP_K)
    rows = []
    for rank, (i, s) in enumerate(zip(ids, sims), start=1):
        fname = _STATE["names"][i]
        lab = _STATE["labels"][fname]
        rows.append(dict(rank=rank, file=fname, index=i, similarity=round(s, 3),
                         species=lab["species"], position=lab["position"] or "?",
                         common_name=lab["common_name"], group=lab["group"]))
    matches = MatchList(rows, label, latent, query_mesh, true_file)

    if show:
        _say(f"🤖 The AI's top {TOP_K} matches for {label}:\n")
        _say(f"  {'#':<3}{'Animal':<44}{'Spine position':<18}Similarity")
        _say("  " + "-" * 74)
        for r in rows:
            animal = r["common_name"] or r["species"]
            if r["common_name"]:
                animal += f" ({r['species']})"
            star = "  ⭐" if true_file and r["file"] == true_file else ""
            _say(f"  {r['rank']:<3}{animal[:43]:<44}{r['position'][:17]:<18}{r['similarity']:.2f}{star}")
        _say("\n  (Similarity: 1.00 means the shapes are identical.)")
        if true_file:
            if any(r["file"] == true_file for r in rows):
                rank = next(r["rank"] for r in rows if r["file"] == true_file)
                _say(f"\n🎉 The AI recognized it! The real bone is match #{rank} (⭐).")
            else:
                _say("\n🤔 The real bone isn't in the top 5. The break may have hidden "
                     "the parts the AI needed to recognize it.")
    return matches


def show_match(matches: MatchList, rank: int = 1):
    """Show the fossil next to one of the AI's matches (rank 1–5)."""
    if not 1 <= rank <= len(matches):
        raise ValueError(f"😕 Pick a rank from 1 to {len(matches)}.")
    r = matches[rank - 1]
    path = MODERN_DIR / r["file"]
    if not path.exists():
        _say(f"😕 The 3D file for match #{rank} isn't in our collection. Try another rank!")
        return
    q = matches.query_mesh
    q_mesh = _read(q) if isinstance(q, (str, Path)) else q
    items = []
    if q_mesh is not None:
        items.append((q_mesh, matches.query_label, FOSSIL_COLOR))
    items.append((_read(path), f"Match #{rank}: {_describe(r['file'], html=True)}"
                               f"<br><sub>similarity {r['similarity']:.2f}</sub>", MODERN_COLOR))
    _show_side_by_side(items)


def show_map(what="mystery_fossil"):
    """Show where a fossil lands on a 2-D map of every bone the AI knows."""
    import plotly.graph_objects as go
    from plotly.colors import qualitative
    from sklearn.decomposition import PCA

    _ensure_setup()
    matches = find_matches(what, show=False)
    codes = _STATE["latent_codes"].numpy()
    pca = PCA(n_components=2).fit(codes)
    xy = pca.transform(codes)
    q = pca.transform(matches.latent.numpy())[0]
    names = _STATE["names"]
    labels = _STATE["labels"]
    hover = "%{text}<extra></extra>"

    fig = go.Figure()
    groups = sorted({labels[n]["group"] for n in names if labels[n]["group"]})
    if groups:
        palette = qualitative.Safe
        for gi, g in enumerate(groups):
            idx = [i for i, n in enumerate(names) if labels[n]["group"] == g]
            fig.add_scatter(x=xy[idx, 0], y=xy[idx, 1], mode="markers", name=g,
                            marker=dict(color=palette[gi % len(palette)], size=7, opacity=0.45),
                            text=[_describe(names[i]) for i in idx], hovertemplate=hover)
        rest = [i for i, n in enumerate(names) if not labels[n]["group"]]
    else:
        rest = list(range(len(names)))
    if rest:
        fig.add_scatter(x=xy[rest, 0], y=xy[rest, 1], mode="markers", name="Modern bones",
                        marker=dict(color="gray", size=7, opacity=0.35),
                        text=[_describe(names[i]) for i in rest], hovertemplate=hover)

    top = [m["index"] for m in matches]
    fig.add_scatter(x=xy[top, 0], y=xy[top, 1], mode="markers", name="AI's top 5",
                    marker=dict(color="royalblue", size=13, line=dict(color="white", width=1.5)),
                    text=[f"#{m['rank']}: {_describe(m['file'])}" for m in matches],
                    hovertemplate=hover)
    fig.add_scatter(x=[q[0]], y=[q[1]], mode="markers", name=matches.query_label,
                    marker=dict(color="red", size=20, symbol="star",
                                line=dict(color="black", width=1)),
                    text=[matches.query_label], hovertemplate=hover)
    fig.update_layout(height=600, plot_bgcolor="white",
                      title=dict(text="🗺️ The shape map: close together = similar shape", x=0.5),
                      xaxis=dict(title="shape direction 1", showticklabels=False, zeroline=False,
                                 gridcolor="#eee"),
                      yaxis=dict(title="shape direction 2", showticklabels=False, zeroline=False,
                                 gridcolor="#eee"),
                      legend=dict(bgcolor="rgba(255,255,255,0.8)"))
    fig.show()
    _say("🖱️  Hover over any dot to see what animal it is. Drag a box to zoom in; "
         "double-click to zoom back out.")


def compare_versions():
    """Compare the AI's answer for the broken fossil before and after fixing it."""
    _ensure_setup()
    keys = ["broken_fossil", "broken_fossil_ai_fixed", "broken_fossil_ai_fixed_cleaned"]
    nice = {"broken_fossil": "1. Broken (as found)",
            "broken_fossil_ai_fixed": "2. AI rebuilt",
            "broken_fossil_ai_fixed_cleaned": "3. AI rebuilt + cleaned"}
    _show_side_by_side([(_read(FOSSIL_DIR / f"{FOSSILS[k]}.vtk"), nice[k],
                         FOSSIL_COLOR if k == "broken_fossil" else FIXED_COLOR) for k in keys],
                       title="The same fossil, three ways")
    _say("\n🤖 The AI's #1 match for each version:\n")
    _say(f"  {'Version':<28}{'#1 match':<46}Similarity")
    _say("  " + "-" * 84)
    for k in keys:
        m = find_matches(k, show=False)[0]
        animal = m["common_name"] or m["species"]
        if m["position"] != "?":
            animal += f", {m['position']}"
        _say(f"  {nice[k]:<28}{animal[:45]:<46}{m['similarity']:.2f}")
    _say(f"\n🔬 What scientists concluded: {KNOWN_ANSWER['broken_fossil']}.")
    _say("   (Tip: use find_matches(\"broken_fossil_ai_fixed_cleaned\") to see all 5 matches.)")


# ------------------------------------------------------- break & fix a bone --
def list_modern_bones():
    """Print the modern bones you can break."""
    _ensure_setup()
    files = _modern_files()
    _say(f"🦴 {len(files)} modern bones you can use:\n")
    for f in files:
        _say(f"  • \"{f.name}\" — {_describe(f.name)}")


def _modern_files():
    return sorted(list(MODERN_DIR.glob("*.vtk")) + list(MODERN_DIR.glob("*.ply")))


def pick_modern_bone(name: str | None = None, show: bool = True) -> dict:
    """Pick a modern vertebra (random if no name is given)."""
    _ensure_setup()
    files = _modern_files()
    if not files:
        raise RuntimeError("😕 No modern bones found. Please ask your teacher for help.")
    if name is None:
        path = random.choice(files)
    else:
        path = MODERN_DIR / name
        if not path.exists():
            raise ValueError(f"😕 No bone called '{name}'. Run list_modern_bones() to see options.")
    mesh = _read(path)
    bone = {"file": path.name, "mesh": mesh, "label": _describe(path.name)}
    if show:
        _show_side_by_side([(mesh, _describe(path.name, html=True), MODERN_COLOR)])
        _say(f"🦴 You picked: {bone['label']}. Run this cell again for a different bone.")
    return bone


def break_bone(bone: dict, how_much: float = 0.3, side: str = "top", show: bool = True) -> dict:
    """Chop part of a bone off. how_much: 0.1 (small chip) to 0.6 (more than half)."""
    if side not in SIDES:
        raise ValueError(f"😕 side must be one of: {', '.join(SIDES)}")
    how_much = float(min(max(how_much, 0.05), 0.6))
    mesh = bone["mesh"].extract_surface().triangulate()
    n = np.asarray(SIDES[side], dtype=float)
    proj = mesh.points @ n
    cut = proj.max() - how_much * (proj.max() - proj.min())
    broken = mesh.clip(normal=n, origin=n * cut, invert=True).extract_surface()
    if broken.n_points < 50:
        raise ValueError("😕 That broke off almost everything! Try a smaller how_much.")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / f"broken_{Path(bone['file']).stem}_{side}_{int(how_much * 100)}.vtk"
    broken.save(str(path))
    out = {**bone, "broken_mesh": broken, "broken_path": path, "how_much": how_much, "side": side}
    if show:
        _show_side_by_side([(bone["mesh"], "Before", MODERN_COLOR),
                            (broken, f"After: {int(how_much * 100)}% chopped off the {side}",
                             BROKEN_COLOR)])
    return out


def _recon_kwargs_and_config(fn):
    """Lower the reconstruction resolution so mesh-building is fast on CPU."""
    cfg = dict(_STATE["config"])
    keys = ("n_pts_per_axis", "gridN", "grid_n", "resolution", "mesh_resolution")
    changed = [k for k in keys if k in cfg]
    for k in changed:
        cfg[k] = RECON_RESOLUTION
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs = {k: RECON_RESOLUTION for k in keys if k in params}
    if not changed and not kwargs and not _STATE.get("warned_res"):
        _STATE["warned_res"] = True
        _say("   (teacher note: couldn't find a resolution setting to lower — "
             "using the model's default, which may be slow)")
    return cfg, kwargs


def fix_bone(broken: dict, show: bool = True) -> dict:
    """Ask the AI to rebuild the missing part of a broken bone. Takes a few minutes."""
    _ensure_setup()
    from NSM.optimization import (build_sdf_dataset, encode_latent_pointnet,
                                  get_norm_params, normalize_mesh,
                                  optimize_latent_partial, reconstruct_mesh_from_latent)
    s = _STATE
    path = str(broken["broken_path"])
    encoder_ckpt = str(MODEL_DIR / "encoder" / "checkpoints" / "encoder.pt")
    t = _Timer()

    _say("🔍 Step 1 of 3: The AI is measuring the broken bone...")
    points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(path, s["config"], n_samples=None)
    _say(f"   done ({t.lap()})")

    _say("🧠 Step 2 of 3: The AI is figuring out what the whole bone probably looked like...")
    latent = encode_latent_pointnet(encoder_ckpt, points, sdf_vals, "cpu")
    if REFINE_ITERS > 0:
        latent, _ = optimize_latent_partial(
            s["model"], points.squeeze(), sdf_vals, s["config"]["latent_size"],
            latent_init=latent, top_k=s["top_k_reg"], iters=REFINE_ITERS, lr=1e-3,
            lambda_reg=1e-6, clamp_val=None, latent_std=s["latent_std"], scheduler_step=800,
            scheduler_gamma=0.7, batch_inference_size=32768, multi_stage=True, device="cpu")
    _say(f"   done ({t.lap()})")

    _say("🏗️  Step 3 of 3: The AI is building the 3D shape...")
    cfg, kwargs = _recon_kwargs_and_config(reconstruct_mesh_from_latent)
    mesh_out = reconstruct_mesh_from_latent(path, s["model"], latent, cfg, **kwargs)
    center, max_radius = get_norm_params(sdf_dataset, sample_dict, path)
    fixed = normalize_mesh(mesh_out, path, cfg, center, max_radius)
    fixed.save(str(Path(path).with_name(Path(path).stem.replace("broken_", "fixed_") + ".vtk")))
    _say(f"   done ({t.lap()})\n")

    out = {**broken, "fixed_mesh": fixed,
           "latent": latent.detach().cpu().reshape(1, -1),
           "label": f"your fixed bone"}
    if show:
        _show_side_by_side([(broken["mesh"], "1. Original", MODERN_COLOR),
                            (broken["broken_mesh"], "2. Broken", BROKEN_COLOR),
                            (fixed, "3. Fixed by the AI", FIXED_COLOR)])
    return out


# --------------------------------------------------------- teacher tools ----
def self_check():
    """Quick check that packages, data and model load. Run by setup.sh."""
    missing = _missing_data()
    if missing:
        print("❌ Missing data files:\n   " + "\n   ".join(missing))
        return False
    setup(verbose=True)
    n_modern = len(_modern_files())
    print(f"✅ Self-check passed: model loaded, {len(FOSSILS)} fossils, "
          f"{n_modern} modern bones available for Mission 5.")
    return True


def benchmark(bone_name: str | None = None):
    """Time every step students run. Use this to tune REFINE_ITERS / RECON_RESOLUTION."""
    t = _Timer()
    setup(verbose=False)
    print(f"setup():            {t.lap()}")
    find_matches("mystery_fossil", show=False)
    print(f"find_matches():     {t.lap()}")
    files = _modern_files()
    bone = pick_modern_bone(bone_name or files[0].name, show=False)
    broken = break_bone(bone, 0.3, "top", show=False)
    print(f"break_bone():       {t.lap()}")
    print(f"fix_bone() with REFINE_ITERS={REFINE_ITERS}, RECON_RESOLUTION={RECON_RESOLUTION}:")
    fixed = fix_bone(broken, show=False)
    print(f"fix_bone() total:   {t.lap()}")
    find_matches(fixed, show=True)


__all__ = ["setup", "list_fossils", "show_fossil", "find_matches", "show_match", "show_map",
           "compare_versions", "list_modern_bones", "pick_modern_bone", "break_bone", "fix_bone"]
