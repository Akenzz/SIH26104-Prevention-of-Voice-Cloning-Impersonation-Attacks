git pull
cd realtime-backend
pip install -r requirements.txt
# Decision expert = hybrid (LFCC-LCNN trained on the 6-language hybrid clean-model
# mix, 130 spoof generators, bonafide<->spoof paired within each language so
# corpus/channel can't be a label shortcut). Best measured generalization of any
# checkpoint here: dev EER 2.42%, unseen-generator ood_en_mlaad 4.58%,
# real-world ood_itw 9.73%, pooled eval_ood 5.91%. Its matched calibrator is far
# better behaved than mc_v3's (held-out ECE 0.017 vs 0.066, EER 0.010 vs 0.266).
# All experts' raw per-window scores are still reported; the risk band uses hybrid.
$env:EXPERTS="wavlm,lfcc,hindi,mc_v3,hybrid"
$env:SINGLE_EXPERT="hybrid"
$env:CALIBRATOR_PATH="artifacts/calibrator_hybrid_clean.json"
$env:DEVICE="cuda"
# Previous config (mc_v3 as the decision expert):
#   $env:EXPERTS="wavlm,lfcc,hindi,mc_v3"; $env:SINGLE_EXPERT="mc_v3"
#   $env:CALIBRATOR_PATH="artifacts/calibrator_mc_v3_combined.json"
uvicorn server:app --host 0.0.0.0 --port 8000
