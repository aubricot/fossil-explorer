# 🦴 Fossil Explorer

Use AI to identify fossil snake and lizard vertebrae, and to rebuild broken ones. This repo is built for middle school classrooms and runs in **GitHub Codespaces** on a normal 2-core machine, with no GPU and nothing to install.

## For students

1. Click the green **Code** button → **Codespaces** → **Create codespace on main**.
2. Wait for it to open. `Fossil_Explorer.ipynb` opens automatically.
3. If it asks you to **Select Kernel**, choose **Fossil Explorer**.
4. Follow the missions in the notebook, running each box with **Shift + Enter**.

When you're done, close the browser tab. The codespace shuts itself off.

---

## For teachers

### How it works
The Colab research demos optimize each fossil's latent code on a GPU, which would take far too long on a 2-core codespace. So the slow work is done **once, ahead of time**, by `teacher/precompute.py`. Students then run only the fast parts:

| Notebook mission | What runs live | Typical speed on 2 cores |
|---|---|---|
| 1. Meet the fossil | Load and render a mesh | seconds |
| 2. Ask the AI | Cosine similarity on a precomputed latent | instant |
| 3. Shape map | PCA of the latent codes | seconds |
| 4. Broken fossil | Same lookup for the 3 precomputed variants | seconds |
| 5. Break & fix a bone | PointNet encoder + short refinement + low-res mesh | a few minutes (**benchmark it**) |

Cut from the research demos: 2-phase optimization, t-SNE, the uncertainty tools, GIF export, Google Drive, and Colab forms.

### One-time setup

**1. Build the data bundle (Colab, GPU).** Follow the instructions at the top of `teacher/precompute.py`. It produces `student_data.zip` containing the model, the fossils, the precomputed latents, a subset of modern meshes, and `labels.csv`.

**2. Fill in `labels.csv`.** This is what makes the results readable for kids. For every training mesh, add:
- `common_name`, e.g. *Green iguana*
- `position`, e.g. *front*, *middle*, *tail*
- `group`, e.g. *snake* or *lizard*. This colors the shape map.

Blank fields fall back to a name guessed from the file name.

**3. Get the data into the codespace.** Pick one:
- **Commit it:** unzip into `data/`, so you have `data/model/...`, `data/fossils/...` and so on. This is fine if the total is well under 100 MB per file.
- **Release asset (recommended if large):** attach `student_data.zip` to a GitHub Release, then put its download URL in `.devcontainer/devcontainer.json` → `containerEnv.DATA_URL`.

Avoid Google Drive links. Drive rate-limits a classroom's worth of simultaneous downloads.

**4. Turn on prebuilds.** Go to Repo **Settings → Codespaces → Set up prebuild**. Pick the `main` branch and your region. With prebuilds on, students open a ready codespace in under a minute instead of waiting through a roughly 10-minute install.

**5. Test and tune in a codespace.** Open a terminal and run:
```bash
python -c "import helpers; helpers.benchmark()"
```
If `fix_bone()` is too slow for your class period, lower `REFINE_ITERS` and/or `RECON_RESOLUTION` at the top of `helpers.py`.

### Things to verify on your first run
These depend on NSM internals that weren't visible from the demo notebooks:
- **Mesh resolution setting.** `helpers.py` tries to lower the reconstruction grid by overriding config keys and function arguments named `n_pts_per_axis`, `gridN` and similar. If `fix_bone()` prints a *"teacher note: couldn't find a resolution setting"* line, find the real parameter in `NSM.optimization.reconstruct_mesh_from_latent` and add its name to the `keys` tuple in `_recon_kwargs_and_config`.
- **Hard-coded CUDA.** `helpers.py` makes `torch.load` default to CPU and turns `.cuda()` into a no-op. Any `torch.device("cuda")` inside NSM would still fail and needs a small fix in NSM.
- **Dependencies.** NSM is installed with `--no-deps` so it can't pull in the CUDA build of torch. If an import fails, add the missing package to `requirements.txt`. If NSM imports fine without `open3d`, remove it, because it adds about 400 MB.
- **`break_bone` directions.** The `"top"`, `"left"` and other names assume the MorphoWeave axes. Rename the entries in `SIDES` if they don't match what students see.
- **Pin NSM** to a commit in `.devcontainer/setup.sh` so an upstream change can't break class day.

### Accounts and quotas
- GitHub requires users to be **13 or older**. Check how your school provides accounts before assigning this.
- Through **GitHub Classroom**, verified teachers get a free monthly Codespaces allowance, and codespaces are restricted to 2-core machines. That's plenty for this activity.

### Repo layout
```
Fossil_Explorer.ipynb     student notebook (5 missions)
helpers.py                all the code students call; SETTINGS block at the top
data/                     model + fossil data (from precompute.py)
.devcontainer/            Codespaces config + setup script (runs during prebuild)
scripts/get_data.sh       downloads the data bundle if data/ isn't committed
teacher/precompute.py     run once on Colab GPU to build the data bundle
```

### Credits
Modern vertebrae: oVert Initiative (Blackburn et al. 2024). Fossils: UF271967, UF546657 via MorphoSource. Identification of UF546657: Bourque & Stanley 2025. Model: Neural Shape Model (Gatti et al. 2025; Park et al. 2019); Porto et al. 2026.
