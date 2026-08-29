git pull
cd realtime-backend
pip install -r requirements.txt
$env:EXPERTS="wavlm,lfcc"
$env:DEVICE="cuda"
uvicorn server:app --host 0.0.0.0 --port 8000
