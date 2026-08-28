import sys
from pathlib import Path

log_path = Path(r"d:\SIH\SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks\data_pipeline\reports\v2_training_log.txt")

try:
    with open(log_path, 'r', encoding='utf-16le') as f:
        lines = f.readlines()
        
    print("Last 20 lines of log:")
    print("----------------------")
    for line in lines[-20:]:
        print(line.rstrip('\n'))
except Exception as e:
    print(f"Error reading log: {e}")
