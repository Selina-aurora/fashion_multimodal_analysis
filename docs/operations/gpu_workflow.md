# GPU Workflow

## Principle

Use the local environment for code development, syntax checks and small pilots. Use the GPU server only for model-heavy batch inference or training.

```text
Local development
        ↓
Small smoke test
        ↓
Package required scripts / benchmark manifest
        ↓
GPU batch run
        ↓
Download structured reports
        ↓
Local analysis and documentation
```

## Current evaluated models

- Mask2Former baseline: `facebook/mask2former-swin-tiny-coco-instance`
- Grounding DINO baseline: `IDEA-Research/grounding-dino-tiny`

## 88-case GPU evaluation

The final 88-case FULL/ROI/LOCAL batch evaluation was completed in a CUDA environment. The repository keeps the reproducible script and structured CSV reports, while large image outputs are ignored by Git.

A minimal GPU bundle can be produced with:

```bash
python scripts/build_gpu_bundle_v3.py
```

The generated bundle is intended for transfer only and should not be committed to the repository.

## Dependency note

Do not overwrite a working GPU server's CUDA-compatible PyTorch installation unnecessarily. Install the remaining Python dependencies from `requirements.txt` after confirming the server's `torch.cuda.is_available()` result.
