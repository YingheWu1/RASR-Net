# RASR-Net

PyTorch implementation and data for inverting surface-wave apparent-velocity curves into velocity structures.

## Contents

- `Train.py` — model definition, training, validation, test evaluation, and inference helper.
- `Test.py` — minimal example entry point for `quick_predict` (replace its placeholder paths before use).
- `Va_r.mat` — apparent-velocity curves (`Va_r`).
- `km_vs_dz010.mat` — velocity structures (`km_vs`).

## Environment

Python 3.9+ is recommended. Install the dependencies with:

```bash
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) if GPU acceleration is required.

## Training

Place the two provided MATLAB data files in the repository root, then run:

```bash
python Train.py
```

The script uses an 80/10/10 train/validation/test split and writes training artifacts including `best_velocity_model.pth`, `loss_data.mat`, and `training_history.png` to the working directory. These generated outputs are intentionally excluded from version control.

## Inference

`quick_predict` in `Train.py` accepts a MATLAB file containing `Va_r` and produces a MATLAB file containing `velocity_structures`:

```python
from Train import quick_predict

predictions = quick_predict(
    input_file="path/to/input.mat",
    output_file="predictions.mat",
    checkpoint_path="best_velocity_model.pth",
)
```

## Citation

If you use this code or dataset, please cite the accompanying paper. Citation details will be added after publication.

## License

No license is currently specified. Please contact the authors before redistribution or reuse beyond what is permitted by applicable law.
