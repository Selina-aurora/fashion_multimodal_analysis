"""一键安装候选、核对输入并运行固定 8 图；需要在 GPU 项目根目录执行。"""
from pathlib import Path
import os
import subprocess
import sys


def main():
    root = Path.cwd()
    package = Path(__file__).resolve().parent
    if not (root / 'src/fashion_multimodal_analysis').is_dir():
        raise SystemExit('请先 cd /workspace/fashion_multimodal_analysis')
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root/'src'), HF_HOME='/workspace/hf_cache',
               HF_HUB_CACHE='/workspace/hf_cache/hub', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1')
    if not Path(env['HF_HUB_CACHE']).is_dir():
        raise SystemExit('未找到昨日模型缓存 /workspace/hf_cache/hub，未开始运行。')
    subprocess.run([sys.executable, str(package/'install_patch.py'),
                    '--project-root', str(root)], cwd=root, env=env, check=True)
    script = 'scripts/run_prd31_neckline_context_20261009.py'
    subprocess.run([sys.executable, script, '--dry-run'], cwd=root, env=env, check=True)
    subprocess.run([sys.executable, script], cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
