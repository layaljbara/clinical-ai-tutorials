import urllib.request
import os
import sys

def download_pmc_xml(pmcid, output_dir="."):
    """
    Download the full-text XML for a given PMC ID using NCBI E-utilities.
    
    Args:
        pmcid (str or int): The PMC ID (e.g., 'PMC8713430' or 8713430).
        output_dir (str): Directory to save the file.
        
    Returns:
        str: Path to the downloaded file.
    """
    # Clean PMCID string
    pmcid_str = str(pmcid).strip()
    
    # Strip common extensions if the user accidentally included them
    for ext in ['.xml', '.nxml', '.XML', '.NXML']:
        if pmcid_str.endswith(ext):
            pmcid_str = pmcid_str[:-len(ext)]
            
    pmcid_str = pmcid_str.upper()
    
    if not pmcid_str.startswith("PMC"):
        pmcid_numeric = pmcid_str
        pmcid_full = f"PMC{pmcid_str}"
    else:
        pmcid_numeric = pmcid_str[3:]
        pmcid_full = pmcid_str

    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid_numeric}"
    output_path = os.path.join(output_dir, f"{pmcid_full}.xml")

    print(f"Downloading {pmcid_full} from {url}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                content = response.read()
                # Check if we got an error message instead of XML
                if b"Error" in content[:100] and len(content) < 500:
                    print(f"Error: Received an error response from NCBI for {pmcid_full}")
                    return None
                
                with open(output_path, "wb") as f:
                    f.write(content)
                print(f"Successfully saved to: {output_path}")
                return output_path
            else:
                print(f"Failed to download. Status code: {response.status}")
                return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_pmc.py <PMCID1> <PMCID2> ...")
        print("Example: python download_pmc.py 8713430 7067710")
    else:
        for arg in sys.argv[1:]:
            download_pmc_xml(arg)
