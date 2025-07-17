import requests
from bs4 import BeautifulSoup
import os



def webscrape(url):
    
    text = ""
    
    headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36"}
    

        
    response = requests.get(url, headers=headers)
   
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
            
        # Extract 
        main = soup.find('main')
        text += main.text if main is not None else " "
    
    else:
        print(f"Failed to retrieve the page, status code: {response.status_code}")
    
    return text


def main():
    webpages = {
        "Visa/Consular Services": "https://www.ph.emb-japan.go.jp/itpr_en/00_000035.html",
        "Important Note on the Document Submission": "https://www.ph.emb-japan.go.jp/itprtop_en/11_000001_00896.html",
        "INQUIRIES CONCERNING VISA": "https://www.ph.emb-japan.go.jp/itpr_en/11_000001_00254.html",
        "General Information": "https://www.ph.emb-japan.go.jp/itpr_en/00_000261.html",
        "List of accredited agencies": "https://www.ph.emb-japan.go.jp/itpr_ja/00_000253.html",
        "Frequently Asked Questions": "https://www.ph.emb-japan.go.jp/itpr_ja/00_001000.html",
        "For Visitors to MIYAGI, FUKUSHIMA and IWATE Prefectures": "https://www.ph.emb-japan.go.jp/itpr_en/00_000260.html",
        "List of Registered Travel Agencies for Package Tour": "https://www.ph.emb-japan.go.jp/itpr_ja/00_000921.html",
    }
    
    text = ""
    print("Loading....")
    for title, url in webpages.items():
        
        text += webscrape(url)
        print(f"{title}")
    print("done")
    
    FILE_PATH = os.path.join(os.getcwd(), "dataset", "webscrape", "raw.txt")
    with open(FILE_PATH, "w") as file:
        file.write(text)
    
    print(f"{FILE_PATH} is generated")


if __name__ == "__main__":
    main()