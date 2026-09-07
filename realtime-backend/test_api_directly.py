import httpx
import asyncio
import json

async def test():
    async with httpx.AsyncClient() as client:
        with open("../testdata/bonafied2.wav", "rb") as f:
            files = {"file": ("bonafied2.wav", f, "audio/wav")}
            async with client.stream("POST", "http://127.0.0.1:8000/predict-file", files=files) as response:
                async for chunk in response.aiter_lines():
                    if chunk.startswith("data: "):
                        try:
                            data = json.loads(chunk[6:].strip())
                            if data.get("event") == "window_scored":
                                print(f"Window {data.get('window_index')}: SSL logit = {data.get('raw_per_expert_scores', {}).get('ssl')}")
                            elif data.get("event") == "summary":
                                print(f"Summary: probability = {data.get('weighted_spoof_probability')}")
                        except Exception as e:
                            print(f"Error parsing JSON: {e}")

asyncio.run(test())
