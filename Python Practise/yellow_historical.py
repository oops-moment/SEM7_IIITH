import requests
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_file(url, local_path):
    """
    Download a single file from URL with progress bar
    """
    try:
        # Make a streaming request
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Get file size from headers
        file_size = int(response.headers.get('content-length', 0))
        
        # Create progress bar
        progress = tqdm(total=file_size, unit='B', unit_scale=True,
                        desc=f'Downloading {os.path.basename(local_path)}')
        
        # Download and write the file
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
        
        progress.close()
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        if os.path.exists(local_path):
            os.remove(local_path)  # Remove partial download
        return False

def main():
    # Base URL
    base_url = "https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com/NEX-GDDP-CMIP6"
    
    # Models, scenarios, and variables to iterate over
    models = [
        "ACCESS-CM2", "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5", 
        "CMCC-CM2-SR5", "CMCC-ESM2", "EC-Earth3-Veg-LR", "EC-Earth3",
        "GFDL-ESM4", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G",
        "MIROC6", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MRI-ESM2-0", 
        "NorESM2-LM", "NorESM2-MM", "TaiESM1"
    ]

    # Scenarios and variables
    # scenarios = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
    variables = ["pr", "tas", "tasmax", "tasmin"]
    
    # Ensemble member
    ensemble_member = "r1i1p1f1"
    
    # Years for which the data is available
    years = range(1950, 1952)  # Adjust the range as needed
    
    # Create download directory if it doesn't exist
    download_dir = "yellow_historical"
    os.makedirs(download_dir, exist_ok=True)
    
    # Prepare download tasks
    download_tasks = []
    for model in models:
        # for scenario in scenarios:
            for variable in variables:
                for year in years:
                    file_name = f"{variable}_day_{model}_historical_{ensemble_member}_gn_{year}_v1.1.nc"
                    url = f"{base_url}/{model}/historical/{ensemble_member}/{variable}/{file_name}"
                    local_path = os.path.join(download_dir, file_name)
                    download_tasks.append((url, local_path))
    
    # Download files using thread pool
    print("Starting downloads...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda x: download_file(*x), download_tasks))
    
    # Print summary
    successful = sum(results)
    print(f"\nDownload complete: {successful}/{len(download_tasks)} files downloaded successfully")

if __name__ == "__main__":
    main()
