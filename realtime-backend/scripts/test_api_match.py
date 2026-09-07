import httpx
import asyncio
import json
import os
import glob
from pathlib import Path

async def test_file(client, filepath):
    url = "http://127.0.0.1:8000/predict-file"
    filename = Path(filepath).name
    try:
        with open(filepath, "rb") as f:
            files = {"file": (filename, f, "audio/wav")}
            async with client.stream("POST", url, files=files) as response:
                summary = None
                async for chunk in response.aiter_lines():
                    if chunk.startswith("data: "):
                        data_str = chunk[6:].strip()
                        if not data_str:
                            continue
                        try:
                            data = json.loads(data_str)
                            if data.get("event") == "summary":
                                summary = data
                        except json.JSONDecodeError:
                            pass
                
                if summary:
                    prob = summary.get("weighted_spoof_probability", 0.0)
                    prob_pct = prob * 100
                    state = summary.get("overall_risk_state")
                    print(f"{filename:30s} | API Prob: {prob_pct:5.1f}% | State: {state}")
                else:
                    print(f"{filename:30s} | Failed to get summary from API")
    except Exception as e:
        print(f"{filename:30s} | Error: {e}")

async def main():
    testdata_dir = "../testdata"
    files = sorted(glob.glob(os.path.join(testdata_dir, "*.*")))
    audio_files = [f for f in files if f.endswith('.wav') or f.endswith('.mp3')]
    
    print("Testing API against testdata...")
    print("-" * 60)
    
    # Wait for server to be ready
    async with httpx.AsyncClient(timeout=3.0) as client:
        for _ in range(30):
            try:
                await client.get("http://127.0.0.1:8000/health")
                break
            except Exception:
                await asyncio.sleep(2)
        else:
            print("Server did not start in time.")
            return

    async with httpx.AsyncClient(timeout=300.0) as client:
        for f in audio_files:
            await test_file(client, f)

if __name__ == "__main__":
    asyncio.run(main())
