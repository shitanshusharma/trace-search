"""Semgrep --test fixtures for the custom trace-search rules.

The gating scan excludes this file. semgrep --test loads it on purpose.
"""

# ruleid: trace-search-no-face-recognition
import face_recognition

# ruleid: trace-search-no-face-recognition
from face_recognition import load_image_file

# ruleid: trace-search-no-face-recognition
import insightface

# ok: trace-search-no-face-recognition
import os


# ruleid: trace-search-no-watchlist
watchlist = []

# ruleid: trace-search-no-watchlist
label = "watchlist"

# ok: trace-search-no-watchlist
camera_name = "cam-01"


# ruleid: trace-search-no-face-embedding
face_embedding = object()

# ruleid: trace-search-no-face-embedding
face_crop = object()

# ok: trace-search-no-face-embedding
file_name = "clip.mp4"


# ruleid: trace-search-no-cloud-apis
import boto3

# ruleid: trace-search-no-cloud-apis
from boto3 import client

# ruleid: trace-search-no-cloud-apis
from google.cloud import storage

# ok: trace-search-no-cloud-apis
from pathlib import Path


# ruleid: trace-search-no-name-field
name: str = "Ada"


# ruleid: trace-search-no-name-field
def _identity(name: str) -> str:
    return name


# ruleid: trace-search-no-name-field
class _Person:
    name: str


def _ingest(self) -> None:
    self.name = "ingest"
    camera_name = "cam-01"
    file_name = "clip.mp4"


# ok: trace-search-no-name-field
def _filename(file_name: str) -> str:
    return file_name


# ruleid: trace-search-no-research-datasets
DATASET = "market1501"

# ruleid: trace-search-no-research-datasets
OTHER = "Market-1501"

# ruleid: trace-search-no-research-datasets
WEIGHTS = "MSMT17"

# ruleid: trace-search-no-research-datasets
DUKE = "DukeMTMC"

# ok: trace-search-no-research-datasets
LOCAL = "office-cam-01"
