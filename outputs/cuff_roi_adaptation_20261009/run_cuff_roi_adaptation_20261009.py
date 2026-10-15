"""Generate train predicted views, train independently, and evaluate predicted ROI."""
import sys,subprocess
from pathlib import Path
import prepare_roi_adaptation
import train_cuff_roi_adapt
if __name__=='__main__':
    dataset=prepare_roi_adaptation.main()
    sys.argv=[sys.argv[0],'--dataset-dir',str(dataset)]
    run=train_cuff_roi_adapt.main()
    subprocess.run([sys.executable,str(Path(__file__).with_name('eval_predicted_roi.py')),'--run-dir',str(run)],check=True)
