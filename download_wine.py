import urllib.request
import os
import sys

print(f"CWD: {os.getcwd()}")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dest = "data/winequality-red.csv"

if not os.path.exists("data"):
    print("Creating data directory...")
    os.makedirs("data")

req = urllib.request.Request(
    url, 
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
)

print(f"Downloading {url} to {dest}...")
try:
    with urllib.request.urlopen(req) as response:
        data = response.read()
        with open(dest, "wb") as f:
            f.write(data)
    print(f"Download complete. Size: {os.path.getsize(dest)} bytes")
except Exception as e:
    print(f"Error: {e}")
