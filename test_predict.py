import requests
import sys
import json

def test_predict_endpoint(file_path):
    url = "http://127.0.0.1:8000/predict-file"
    print(f"Sending POST request to {url} with file {file_path}...")
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2))
        except ValueError:
            print("Response is not JSON:")
            print(response.text)
            
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_predict.py <path_to_audio_file>")
        sys.exit(1)
        
    test_predict_endpoint(sys.argv[1])
