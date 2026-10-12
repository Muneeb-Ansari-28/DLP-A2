# Deep Learning for Perception – Assignment 2, Section B
**23F-0608 – Muneeb Akbar**

Robust CIFAR-10 classification from only 500 training images (400 train + 100 validation), using an Inception-style CNN trained from scratch and evaluated on a shifted test set (rotation, brightness/contrast, occlusion and a hidden corruption).

## Repository contents
| File | Description |
|---|---|
| `23f-0608-dlp-a2.ipynb` | Fully executed notebook (Section B) with all outputs and plots |
| `README.md` | This file |
| `Summary_23F-0608_DLP_A2.docx` | One-page summary (final score, configuration, most useful change) |

Running the notebook also creates `results/` (CSV tables, `summary.json`) and `figures/` (PNG plots).

## Final results (mean ± std over seeds 0, 1, 2)
| Metric | Value |
|---|---|
| Shifted-test accuracy (final) | **24.93 ± 0.91 %** |
| Clean-test accuracy | 37.99 ± 0.58 % |
| Validation accuracy | 43.00 ± 5.57 % |
| Baseline shifted-test (no augmentation) | 17.11 ± 0.48 % |
| Trainable parameters | 161,330 (limit 1,000,000) |

**Final configuration:** geometric + photometric augmentation + Cutout + Mixup/CutMix (`both`, prob 0.5, α = 1.0), label smoothing 0.1, lr 0.1, weight decay 5e-4, dropout 0.5, SGD (Nesterov) with One-Cycle schedule, batch size 32, max 60 epochs.

**Change that helped most:** Cutout + Mixup/CutMix + label smoothing on top of geometric + photometric augmentation (+3.96 pp on shifted-test).

## Method summary
- **Model:** conv stem (2 × Conv-BN-ReLU) → 3 `InceptionModule`s (1×1, 1×1→3×3, 1×1→5×5, pool→1×1) → Global Average Pooling → Dropout → Linear.
- **Data:** 500 images drawn from the CIFAR-10 training set (50 per class), stratified into 400 train / 100 validation. `clean_test` is the 10,000 CIFAR-10 test images; `shifted_test` is built from it with a fixed-seed corruption pipeline.
- **Rules followed:** no pretrained weights; ≤ 1M parameters; augmentation applied on the fly to training data only; hyper-parameters, checkpoints and early stopping chosen on the validation split only (test sets are locked during the search); ≤ 60 epochs; 3 fixed seeds.
- **Experiments:** baseline, ablation (geometric / photometric / full / tuned), single-corruption analysis, error analysis (per-class accuracy, confusion matrix, misclassified images), hidden-corruption test, and test-time augmentation (bonus).

## How to reproduce
1. **Requirements:** Python 3.10+ (original run: Python 3.13, torch 2.11.0, torchvision 0.26.0, CUDA GPU). A GPU is recommended; the full run takes roughly 10–15 minutes on GPU.
2. **Install dependencies:**
   ```bash
   pip install torch torchvision numpy pandas matplotlib jupyter
   ```
3. **Get the notebook:**
   ```bash
   git clone https://github.com/Muneeb-Ansari-28/DLP-A2.git
   cd DLP-A2
   ```
4. **Run:** open `23f-0608-dlp-a2.ipynb` in Jupyter / Colab / VS Code and choose **Kernel → Restart & Run All**.
   - CIFAR-10 is downloaded automatically to `./data` on first run (about 170 MB).
   - Set `QUICK_TEST = True` in the first code cell for a fast smoke test (3 epochs, 1 seed); keep it `False` to reproduce the reported numbers.
5. **Outputs:** tables are saved to `results/` and plots to `figures/`.

## Reproducibility notes
- Seeds: `GLOBAL_SEED = 42` (fixes the 500-image subset, the 400/100 split and the shifted-test pipeline); training seeds `[0, 1, 2]`. Python, NumPy, PyTorch, CUDA, DataLoader shuffling and augmentation randomness are all seeded.
- The notebook prints MD5 hashes of the train/val index lists (`b3d064a7c5a0d465a3eeb9326017bf28` / `e2b94dc8bcd4bef5dbad5f1081f81e20`) so the split can be verified.
- Small numerical differences can occur across GPUs / library versions.
- **Hidden corruption:** the notebook uses a stand-in (random grayscale / RGB channel swap) in `hidden_corruption(...)`. When the official corruption is released, replace the body of that function (or set `PROVIDED_SHIFTED`) and re-run all cells; the shifted-test numbers above will then change.
