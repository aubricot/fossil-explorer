"""
precompute.py — TEACHER ONLY. Run once on Colab with a GPU.

Builds the data bundle the student codespace needs, so students never have to
run the slow latent optimization on CPU:

  student_data/
    model/                      config, model + latent checkpoints, PointNet encoder
    fossils/<stem>.vtk          the 4 fossil variants (PLY converted to VTK)
    precomputed/<stem>_latent.pt   optimized latent for each fossil (the slow part)
    modern/<file>.vtk           modern meshes: every fossil's top-5 + N extras for Mission 5
    labels.csv                  friendly names for every training mesh (fill in by hand!)
    matches_report.txt          top-5 per fossil, to compare with your Colab demo results
  student_data.zip

HOW TO RUN (in Colab, GPU runtime):
  1. Open your shape_completion_demo.ipynb and run sections 1 and 2
     (installs NSM + downloads run_v44 WITH the encoder, fossils, vertebrae_meshes).
  2. Also download the classification demo's fossils zip, which has the completed
     variants (uf546657_shape_completion.vtk, ..._fillholes_smooth_...):
        !gdown 1UgKYDj4d5d0D4M8MfHFAh-IkW-dkmujf
        !unzip -o fossils.zip -d fossils_cls
  3. Upload this file and run:
        !python precompute.py --model-dir run_v44 --fossil-dir fossils fossils_cls \\
                              --modern-dir vertebrae_meshes --out student_data
  4. Download student_data.zip.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

FOSSIL_FILES = [
    "UF271967.vtk",
    "uf546657.ply",
    "uf546657_shape_completion.vtk",
    "uf546657_fillholes_smooth_shape_completion.vtk",
]


def find_config_dir(root: Path) -> Path:
    hits = sorted(root.rglob("model_params_config.json"))
    if not hits:
        raise SystemExit(f"No model_params_config.json under {root}")
    return hits[0].parent


def find_file(name: str, dirs: list[Path]) -> Path:
    for d in dirs:
        for p in d.rglob("*"):
            if p.name.lower() == name.lower():
                return p
    raise SystemExit(f"Could not find {name} in {', '.join(map(str, dirs))}")


def patch_pymskt():
    import pymskt.mesh.meshes as meshes
    import pymskt.mesh.meshTools as meshTools
    from NSM.helper_funcs import fixed_point_coords, safe_load_mesh_scalars
    meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
    meshes.Mesh.point_coords = property(fixed_point_coords)
    orig = meshTools.pcu.signed_distance_to_mesh
    meshTools.pcu.signed_distance_to_mesh = lambda pts, points, faces: orig(
        np.asarray(pts, dtype=np.float64), np.asarray(points, dtype=np.float64),
        np.asarray(faces, dtype=np.int32))


def guess_species(stem: str) -> str:
    try:
        from NSM.helper_funcs import extract_species_prefix
        s = extract_species_prefix(stem)
        if isinstance(s, str) and s:
            return s.replace("_", " ")
    except Exception:
        pass
    return stem.replace("_", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--ckpt", default="3000")
    ap.add_argument("--fossil-dir", required=True, type=Path, nargs="+")
    ap.add_argument("--modern-dir", required=True, type=Path)
    ap.add_argument("--encoder", type=Path, default=None, help="path to encoder.pt if not inside --model-dir")
    ap.add_argument("--out", default=Path("student_data"), type=Path)
    ap.add_argument("--n-extra-modern", type=int, default=12,
                    help="extra random modern bones for Mission 5 (break & fix)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from NSM.datasets import SDFSamples
    from NSM.helper_funcs import convert_ply_to_vtk, load_config, load_model_and_latents
    from NSM.optimization import find_similar_cos, get_top_k_pcs, optimize_latent
    patch_pymskt()

    out = args.out.resolve()
    for sub in ["model/model", "model/latent_codes", "model/encoder/checkpoints",
                "fossils", "precomputed", "modern"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---- model -------------------------------------------------------------
    model_root = find_config_dir(args.model_dir.resolve())
    cwd = os.getcwd()
    os.chdir(model_root)
    config = load_config(config_path="model_params_config.json")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found — this will be very slow.")
    config["device"] = device
    model, _, latent_codes = load_model_and_latents(
        f"model/{args.ckpt}.pth", f"latent_codes/{args.ckpt}.pth", config, device)
    os.chdir(cwd)
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    _, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.99)
    names = [os.path.basename(p) for p in config["list_mesh_paths"]]
    print(f"Loaded model: {len(names)} training meshes, latent size {config['latent_size']}")

    shutil.copy(model_root / "model_params_config.json", out / "model")
    shutil.copy(model_root / "model" / f"{args.ckpt}.pth", out / "model" / "model")
    shutil.copy(model_root / "latent_codes" / f"{args.ckpt}.pth", out / "model" / "latent_codes")
    enc = args.encoder or next(iter(sorted(args.model_dir.resolve().rglob("encoder.pt"))), None)
    if enc is None:
        print("WARNING: encoder.pt not found — Mission 5 won't work. Pass --encoder.")
    else:
        shutil.copy(enc, out / "model" / "encoder" / "checkpoints" / "encoder.pt")

    # ---- fossils: the slow part ---------------------------------------------
    report = []
    needed_modern = set()
    for fname in FOSSIL_FILES:
        src = find_file(fname, [d.resolve() for d in args.fossil_dir])
        stem = Path(fname).stem
        if src.suffix.lower() == ".ply":
            _, vtk_path = convert_ply_to_vtk(str(src), save=True)
            src = Path(vtk_path)
        shutil.copy(src, out / "fossils" / f"{stem}.vtk")
        print(f"\n=== {stem}: optimizing latent ===")
        ds = SDFSamples(
            list_mesh_paths=[str(src)], multiprocessing=False,
            subsample=config["samples_per_object_per_batch"], print_filename=True,
            n_pts=config["n_pts_per_object"], p_near_surface=config["percent_near_surface"],
            p_further_from_surface=config["percent_further_from_surface"],
            sigma_near=config["sigma_near"], sigma_far=config["sigma_far"],
            rand_function=config["random_function"], center_pts=config["center_pts"],
            norm_pts=config["normalize_pts"], scale_method=config["scale_method"],
            reference_mesh=None, verbose=config["verbose"], save_cache=config["cache"],
            equal_pos_neg=config["equal_pos_neg"], fix_mesh=config["fix_mesh"])
        sample_dict, _ = ds[0]
        points = sample_dict["xyz"].to(device)
        sdf_vals = sample_dict["gt_sdf"]
        latent = optimize_latent(model, points, sdf_vals, config["latent_size"],
                                 top_k_reg, mean_latent, latent_codes)
        torch.save(latent.detach().cpu(), out / "precomputed" / f"{stem}_latent.pt")

        ids, dists = find_similar_cos(latent, latent_codes, top_k=5, n_std=2, device=device)
        report.append(f"{stem}")
        for rank, (i, d) in enumerate(zip(ids, dists), 1):
            i = int(i)
            needed_modern.add(names[i])
            report.append(f"  {rank}. {names[i]}  (index {i}, distance {float(d):.4f})")

    # ---- modern meshes ------------------------------------------------------
    available = {p.name: p for p in args.modern_dir.resolve().rglob("*") if p.suffix.lower() in (".vtk", ".ply")}
    rng = random.Random(args.seed)
    pool = sorted(n for n in names if n in available and n not in needed_modern)
    extras = rng.sample(pool, min(args.n_extra_modern, len(pool)))
    copied, missing = 0, []
    for n in sorted(needed_modern) + extras:
        if n in available:
            shutil.copy(available[n], out / "modern" / n)
            copied += 1
        else:
            missing.append(n)
    print(f"\nCopied {copied} modern meshes ({len(extras)} extras for Mission 5).")
    if missing:
        print("WARNING: not found in --modern-dir:", ", ".join(missing))

    # ---- labels template ----------------------------------------------------
    labels_path = out / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "common_name", "species", "position", "group"])
        for n in names:
            w.writerow([n, "", guess_species(Path(n).stem), "", ""])
    print(f"Wrote {labels_path}  <-- fill in common_name / position / group (e.g. snake, lizard)")

    (out / "matches_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))

    zip_path = shutil.make_archive(str(out), "zip", root_dir=out)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"\nDone. Bundle: {zip_path}  ({size:.0f} MB unzipped)")
    if size > 90:
        print("NOTE: bigger than GitHub's ~100 MB/file comfort zone — host the zip as a "
              "Release asset and set DATA_URL instead of committing data/.")


if __name__ == "__main__":
    main()
