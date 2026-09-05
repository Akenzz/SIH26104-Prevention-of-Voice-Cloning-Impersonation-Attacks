import sys
import httpx
import json
import time

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_predict_file.py <audio_file>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    url = "http://127.0.0.1:8000/predict-file"
    
    start_time = time.time()
    first_result_time = None
    
    with open(filepath, "rb") as f:
        files = {"file": f}
        
        with httpx.Client(timeout=300.0) as client:
            with client.stream("POST", url, files=files) as response:
                print(f"Status: {response.status_code}")
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        if first_result_time is None:
                            first_result_time = time.time()
                            print(f"\n[TIME-TO-FIRST-RESULT] {first_result_time - start_time:.2f}s\n")
                            
                        data = json.loads(line[6:])
                        event = data.get("event")
                        
                        if event == "window_scored":
                            idx = data["window_index"]
                            sec = data["start_time_sec"]
                            prob = data["calibrated_probability"]
                            state = data["risk_state"]
                            print(f"[Window {idx:03d} | {sec:05.2f}s] Prob: {prob*100:6.2f}% | State: {state}")
                        elif event == "skipped":
                            idx = data["window_index"]
                            print(f"[Window {idx:03d}] Skipped (No Speech)")
                        elif event == "summary":
                            print("\n--- FINAL SUMMARY ---")
                            print(json.dumps(data, indent=2))
                            
    end_time = time.time()
    print(f"\n[TOTAL TIME] {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()
