import requests
import os

def download_scimago(year=2024, output_path="scimago_latest.csv"):
    """
    Attempt to download the SCImago Journal Rank CSV.
    Note: SCImago uses Cloudflare, so this may fail in some environments.
    """
    url = "https://www.scimagojr.com/journalrank.php?out=xls"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/csv,application/vnd.ms-excel,*/*",
        "Referer": "https://www.scimagojr.com/journalrank.php"
    }

    print(f"Attempting to download latest SCImago data from: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        # Check if we hit a Cloudflare 'Just a moment' page
        if "Just a moment..." in response.text or "Cloudflare" in response.text:
            print("Error: Blocked by Cloudflare bot protection. Please download the file manually at:")
            print("https://www.scimagojr.com/journalrank.php (Click the 'Download data' button)")
            return False

        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
            
        print(f"Successfully downloaded to: {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

if __name__ == "__main__":
    download_scimago()
