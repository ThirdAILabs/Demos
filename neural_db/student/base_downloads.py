import json
from pathlib import Path
from urllib.parse import urljoin

import requests

base_url = "https://model-zoo.azurewebsites.net/"


# Gets the list of pre-trained models matching with this name.
def list_pretrained_models(filter_by_name: str):
    url = urljoin(base_url, "list")
    response = requests.get(url, params={"name": filter_by_name})
    content = json.loads(response.content)
    return content["data"]


# Gets the signed url link for the given model.
def get_pretrained_model_download_link(display_name: str):
    url = urljoin(base_url, "download")
    response = requests.get(url, params={"display_name": display_name})
    content = json.loads(response.content)
    return content["url"]


def download_pretrained_model(display_name: str, destination: Path):
    url = get_pretrained_model_download_link(display_name)
    r = requests.get(url, allow_redirects=True)
    open(destination, "wb").write(r.content)
