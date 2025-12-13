import os
import requests
import zipfile
from tqdm import tqdm

DATASET_URL = "https://public.roboflow.com/ds/RFGqaneDDL?key=cLVMvfHCep"
DATA_DIR = "data"

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def setup_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    zip_path = os.path.join(DATA_DIR, "dataset.zip")
    
    print(f"Downloading dataset from {DATASET_URL}...")
    download_file(DATASET_URL, zip_path)
    
    print("Unzipping dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    # Clean up
    os.remove(zip_path)
    print("Dataset setup complete.")

if __name__ == "__main__":
    setup_data()
