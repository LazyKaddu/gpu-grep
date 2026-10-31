import random
import os

# Generates a ~10GB file
TARGET_SIZE_BYTES = 10 * 1024 * 1024 * 1024 
FILE_DIR = "benchmark" # Update to "benchmarks" if that is your folder name
FILE_NAME = f"{FILE_DIR}/massive_test.log"

messages = [
    "INFO System booted normally",
    "DEBUG Connection established from 192.168.1.5",
    "WARN Memory usage exceeding 80%",
    "FATAL_CRASH_ERR_509 Segmentation fault in module x",
    "INFO User authentication successful"
]

# 1. Force the directory to exist
os.makedirs(FILE_DIR, exist_ok=True)

print(f"Generating 10GB log file at {FILE_NAME}...")
print("This may take 3-10 minutes depending on your SSD speed.\n")

with open(FILE_NAME, 'w') as f:
    bytes_written = 0
    last_printed_gb = 0
    
    while bytes_written < TARGET_SIZE_BYTES:
        # Write 1000 lines at a time to reduce overhead
        batch = "\n".join(random.choices(messages, weights=[40, 30, 20, 1, 9], k=1000)) + "\n"
        f.write(batch)
        bytes_written += len(batch.encode('utf-8')) # Accurate byte counting
        
        # 2. Print progress every 1 GB
        current_gb = bytes_written // (1024 * 1024 * 1024)
        if current_gb > last_printed_gb:
            print(f"Progress: {current_gb}GB / 10GB written...")
            last_printed_gb = current_gb

print(f"\n File generation complete: {FILE_NAME}")