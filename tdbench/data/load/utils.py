from pathlib import Path
import requests
from rich.progress import Progress

DOWNLOAD_BLOCK = 8192


def download(
    download_url: str,
    download_dir: Path,
    progress: Progress = None,
) -> None:
    """Downloads a file specified by the url into the specified directory.

    Args:
        download_url (str): URL to download the file from.
        download_dir (Path): Directory to save the file under.
    """
    with requests.get(download_url, stream=True) as r:
        total_length = r.headers.get("content-length")
        with open(download_dir, "wb") as f:
            if total_length == None:
                f.write(r.content)
            else:
                total_length = int(total_length)
                for d in r.iter_content(DOWNLOAD_BLOCK):
                    f.write(d)
