import json
import os
import shutil
from pathlib import Path
from urllib.parse import urljoin
import requests
from thirdai import neural_db

from pydantic import BaseModel
from tqdm import tqdm

from model_bazaar.utils import get_directory_size, hash_path, http_get_with_error


class BazaarEntry(BaseModel):
    display_name: str
    trained_on: str
    num_params: str
    size: int
    hash: str

    @staticmethod
    def from_dict(entry):
        return BazaarEntry(
            display_name=entry["display_name"],
            trained_on=entry["trained_on"],
            num_params=entry["num_params"],
            size=entry["size"],
            hash=entry["hash"],
        )


class Bazaar:
    def __init__(
        self, cache_dir: Path, base_url: str = "https://model-zoo.azurewebsites.net/"
    ):
        self._base_url = base_url
        self._cache_dir = cache_dir
        self._registry = {}

    def fetch(self, filter: str = ""):
        url = urljoin(self._base_url, "list")
        response = http_get_with_error(url, params={"name": filter})
        json_entries = json.loads(response.content)["data"]
        entries = [BazaarEntry.from_dict(entry) for entry in json_entries]
        self._registry = {entry.display_name: entry for entry in entries}

    def list_model_names(self):
        return list(self._registry.keys())

    # TODO: On_progress
    def get_model_dir(self, identifier: str):
        if identifier not in self._registry:
            raise ValueError(
                f"Cannot find '{identifier}' in registry. Try fetching first."
            )
        cached_model_dir = self._model_dir_in_cache(identifier)
        if cached_model_dir:
            return cached_model_dir

        if self._cached_model_zip_path(identifier).is_file():
            self._unpack_and_remove_zip(identifier)
            cached_model_dir = self._model_dir_in_cache(identifier)
            if cached_model_dir:
                return cached_model_dir

        self._download(identifier)
        return self._unpack_and_remove_zip(identifier)

    def get_model(self, identifier: str):
        model_dir = self.get_model_dir(identifier=identifier)
        return neural_db.NeuralDB.from_checkpoint(checkpoint_path=model_dir, user_id="")

    def _cached_model_dir_path(self, identifier: str):
        return self._cache_dir / identifier

    def _cached_model_zip_path(self, identifier: str):
        return self._cache_dir / f"{identifier}.zip"

    def _model_dir_in_cache(self, identifier: str):
        cached_model_dir = self._cached_model_dir_path(identifier)
        if cached_model_dir.is_dir():
            bazaar_entry = self._registry[identifier]
            hash_match = hash_path(cached_model_dir) == bazaar_entry.hash
            size_match = get_directory_size(cached_model_dir) == bazaar_entry.size
            if hash_match and size_match:
                return cached_model_dir
        return None

    def _unpack_and_remove_zip(self, identifier: str):
        zip_path = self._cached_model_zip_path(identifier)
        extract_dir = self._cached_model_dir_path(identifier)
        shutil.unpack_archive(filename=zip_path, extract_dir=extract_dir)
        os.remove(zip_path)
        return extract_dir

    def _download(self, identifier: str):
        signing_url = urljoin(self._base_url, "download")
        signing_response = http_get_with_error(
            signing_url, params={"display_name": identifier}
        )
        download_url = json.loads(signing_response.content)["url"]
        destination = self._cached_model_zip_path(identifier)
        response = requests.get(download_url, allow_redirects=True, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024 * 4  # 4MiB
        size_so_far = 0
        with open(destination, "wb") as file, tqdm(
            total=total_size_in_bytes, unit="B", unit_scale=True, desc="Downloading"
        ) as progress_bar:
            for data in response.iter_content(block_size):
                size_so_far += len(data)
                file.write(data)
                progress_bar.update(len(data))

        if size_so_far != total_size_in_bytes:
            raise ValueError("Failed to download.")