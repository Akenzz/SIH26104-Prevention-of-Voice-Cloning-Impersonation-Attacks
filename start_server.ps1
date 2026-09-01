# Start the SIH26104 realtime backend (Windows / PowerShell).
#
# No environment variables are required: config.py already defaults to the two
# shipped experts (wavlm + hybrid) with `hybrid` as the decision expert and its
# matched calibrator. On the first run the checkpoints are downloaded from
# Hugging Face into realtime-backend/model_cache/ and reused from disk after
# that. Set $env:DEVICE="cuda" only if you have a working CUDA PyTorch install;
# the default "cpu" works everywhere.
#
#   Expert-1  wavlm   Akenzz/Expert-1     WavLM Base+          (Person A)
#   Expert-2  hybrid  sarosh22/Expert2    LFCC-LCNN, 6 langs   (decision expert)
#
# Override anything only if you are experimenting, e.g.:
#   $env:SINGLE_EXPERT="wavlm"; $env:CALIBRATOR_PATH="artifacts/platt_v2_combined_dataset.json"

git pull
cd realtime-backend
pip install -r requirements.txt
$env:DEVICE = if ($env:DEVICE) { $env:DEVICE } else { "cpu" }
uvicorn server:app --host 0.0.0.0 --port 8000
