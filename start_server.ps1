git pull
cd realtime-backend
pip install -r requirements.txt
# Decision expert = mc_v3 (multi-corpus V3 LFCC-LCNN, the only LFCC checkpoint with
# MEASURED unseen-generator generalization) paired with its matched calibrator.
# All experts' raw per-window scores are still reported; the risk band uses mc_v3.
$env:EXPERTS="wavlm,lfcc,hindi,mc_v3"
$env:SINGLE_EXPERT="mc_v3"
$env:CALIBRATOR_PATH="artifacts/calibrator_mc_v3_combined.json"
$env:DEVICE="cuda"
# Previous config (wavlm as the default decision expert, ASVspoof-only calibrator):
#   $env:EXPERTS="wavlm,lfcc,hindi"; Remove-Item Env:SINGLE_EXPERT; Remove-Item Env:CALIBRATOR_PATH
uvicorn server:app --host 0.0.0.0 --port 8000
