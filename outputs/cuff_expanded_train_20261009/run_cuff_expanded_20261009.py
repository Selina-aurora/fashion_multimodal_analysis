"""Prepare the frozen expanded dataset, then run cuff pilot training. No original V5 changes."""
from pathlib import Path
import subprocess
import sys
from prepare_cuff_expanded_data import main
if __name__=='__main__':
    dataset=main()
    subprocess.run([sys.executable,str(Path(__file__).with_name('train_cuff_expanded_v3.py')),'--dataset-dir',str(dataset)],check=True)
