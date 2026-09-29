GPU EVALUATION BUNDLE V3
========================

Directory structure:

gpu_eval_v3/
├── fashion_multimodal_analysis/
│   ├── scripts/
│   │   └── evaluate_verified_positive_benchmark_v3.py
│   ├── reports/
│   │   └── verified_positive_benchmark/
│   │       └── verified_positive_benchmark.csv
│   └── requirements_gpu.txt
│
└── fashion_data/
    └── raw/
        └── train/
            └── train/
                ├── image/
                └── annos/


After uploading and extracting on the GPU server:

cd gpu_eval_v3/fashion_multimodal_analysis

Check CUDA:

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

Install missing packages if needed:

pip install -r requirements_gpu.txt

Smoke test:

python scripts/evaluate_verified_positive_benchmark_v3.py --per-target-limit 1

Full 88-case evaluation:

python scripts/evaluate_verified_positive_benchmark_v3.py
