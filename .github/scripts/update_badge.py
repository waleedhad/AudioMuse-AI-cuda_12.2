import requests
from bs4 import BeautifulSoup
import re

URL = "https://github.com/NeptuneHub/AudioMuse-AI/pkgs/container/audiomuse-ai"
README_PATH = "README.md"

def get_download_count():
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html.parser")
    h3 = soup.find("h3", {"title": re.compile(r"\d+")})
    return h3.get("title") if h3 else "0"

def update_readme(downloads):
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    new_badge = f"![Downloads](https://img.shields.io/badge/downloads-{downloads}-blue)"
    content = re.sub(r"!\[Downloads\]\([^)]+\)", new_badge, content)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    count = get_download_count()
    update_readme(count)
