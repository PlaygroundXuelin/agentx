"""Filesystem-backed datasets store."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import pydantic.dataclasses as pydantic_dataclasses


@pydantic_dataclasses.dataclass(frozen=True)
class FsSettings:
    """
    file system store settings
    """

    # root of datasets files in the file system
    root: str = "/tmp/_datasets"


class FsStore:
    """Persist dataset payloads and metadata on the local filesystem."""

    _DATA_FILENAME = "data.bin"
    _METADATA_FILENAME = "metadata.json"
    _DATA_VERSION_PREFIX = "data"
    _METADATA_VERSION_PREFIX = "metadata"

    def __init__(self, settings: FsSettings) -> None:
        self._root = Path(settings.root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._locks_lock = threading.Lock()
        self._dataset_locks: dict[UUID, threading.Lock] = {}

    def store_metadata(self, dataset_id: UUID, metadata: Mapping[str, Any] | Any) -> None:
        """Serialize dataset metadata to JSON."""

        payload = self._normalize_metadata(metadata)
        with self._locked(dataset_id):
            dataset_dir = self._dataset_dir(dataset_id)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            version_path = self._write_metadata_version(dataset_id, payload)
            self._update_metadata_symlink(dataset_id, version_path)

    def fetch_metadata(self, dataset_id: UUID) -> dict[str, Any]:
        """Load dataset metadata from the filesystem."""

        metadata_path = self._metadata_symlink_path(dataset_id)
        if not metadata_path.is_file():
            msg = f"metadata for dataset {dataset_id} does not exist"
            raise FileNotFoundError(msg)
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def update_metadata(
        self,
        dataset_id: UUID,
        metadata: Mapping[str, Any] | Any,
        *,
        overlay: bool = False,
    ) -> dict[str, Any]:
        """Update dataset metadata optionally overlaying existing values."""

        with self._locked(dataset_id):
            if not self.dataset_dir_exists(dataset_id):
                msg = f"dataset {dataset_id} does not exist"
                raise FileNotFoundError(msg)

            payload = self._normalize_metadata(metadata)
            if overlay:
                try:
                    current = self._read_metadata(dataset_id)
                except FileNotFoundError:
                    current = {}
                merged = {**current, **payload}
            else:
                merged = payload

            version_path = self._write_metadata_version(dataset_id, merged)
            self._update_metadata_symlink(dataset_id, version_path)
            return merged

    def store_data(
        self,
        dataset_id: UUID,
        data: bytes | str,
        *,
        encoding: str = "utf-8",
    ) -> Path:
        """Write dataset content to the filesystem and return the versioned path."""

        with self._locked(dataset_id):
            if not self.dataset_dir_exists(dataset_id):
                msg = f"dataset {dataset_id} does not exist"
                raise FileNotFoundError(msg)

            payload: bytes
            if isinstance(data, bytes):
                payload = data
            else:
                payload = data.encode(encoding)

            version_path = self._write_data_version(dataset_id, payload)
            self._update_data_symlink(dataset_id, version_path)
            return version_path

    def fetch_data(
        self, dataset_id: UUID, *, as_text: bool = False, encoding: str = "utf-8"
    ) -> bytes | str:
        """Read dataset content from the filesystem."""

        data_path = self._data_symlink_path(dataset_id)
        if not data_path.is_file():
            msg = f"data for dataset {dataset_id} does not exist"
            raise FileNotFoundError(msg)
        if as_text:
            return data_path.read_text(encoding=encoding)
        return data_path.read_bytes()

    def get_data_path(self, dataset_id: UUID) -> Path:
        """Return the resolved path to the latest dataset file."""

        data_path = self._data_symlink_path(dataset_id)
        if not data_path.is_symlink() and not data_path.exists():
            msg = f"data for dataset {dataset_id} does not exist"
            raise FileNotFoundError(msg)
        if data_path.is_symlink():
            target = data_path.resolve()
        else:
            target = data_path
        if not target.exists():
            msg = f"data file '{target}' for dataset {dataset_id} is missing"
            raise FileNotFoundError(msg)
        return target

    def dataset_exists(self, dataset_id: UUID) -> bool:
        """Return True when both data and metadata files exist for the dataset."""

        metadata_path = self._metadata_symlink_path(dataset_id)
        data_path = self._data_symlink_path(dataset_id)
        return metadata_path.is_file() and data_path.is_file()

    def dataset_dir_exists(self, dataset_id: UUID) -> bool:
        """Return True when the dataset directory exists regardless of contents."""

        return self._dataset_dir(dataset_id).exists()

    def delete_dataset(self, dataset_id: UUID) -> None:
        """Remove dataset files from the filesystem if they exist."""

        dataset_dir = self._dataset_dir(dataset_id)
        if dataset_dir.is_dir():
            for path in dataset_dir.iterdir():
                path.unlink(missing_ok=True)
            dataset_dir.rmdir()
        with self._locks_lock:
            self._dataset_locks.pop(dataset_id, None)

    def _dataset_dir(self, dataset_id: UUID) -> Path:
        return self._root / str(dataset_id)

    def _data_symlink_path(self, dataset_id: UUID) -> Path:
        return self._dataset_dir(dataset_id) / self._DATA_FILENAME

    def _metadata_symlink_path(self, dataset_id: UUID) -> Path:
        return self._dataset_dir(dataset_id) / self._METADATA_FILENAME

    def _metadata_version_path(self, dataset_id: UUID, timestamp_ms: int) -> Path:
        filename = f"{self._METADATA_VERSION_PREFIX}-{timestamp_ms}.json"
        return self._dataset_dir(dataset_id) / filename

    def _data_version_path(self, dataset_id: UUID, timestamp_ms: int) -> Path:
        filename = f"{self._DATA_VERSION_PREFIX}-{timestamp_ms}.bin"
        return self._dataset_dir(dataset_id) / filename

    def _write_metadata_version(self, dataset_id: UUID, payload: Mapping[str, Any]) -> Path:
        dataset_dir = self._dataset_dir(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        timestamp_ms = self._timestamp_ms()
        version_path = self._metadata_version_path(dataset_id, timestamp_ms)
        while version_path.exists():
            timestamp_ms += 1
            version_path = self._metadata_version_path(dataset_id, timestamp_ms)
        version_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return version_path

    def _update_metadata_symlink(self, dataset_id: UUID, version_path: Path) -> None:
        symlink_path = self._metadata_symlink_path(dataset_id)
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        # use relative path to make the symlink portable within the dataset directory
        symlink_path.symlink_to(version_path.name)

    def _write_data_version(self, dataset_id: UUID, payload: bytes) -> Path:
        dataset_dir = self._dataset_dir(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        timestamp_ms = self._timestamp_ms()
        version_path = self._data_version_path(dataset_id, timestamp_ms)
        while version_path.exists():
            timestamp_ms += 1
            version_path = self._data_version_path(dataset_id, timestamp_ms)
        version_path.write_bytes(payload)
        return version_path

    def _update_data_symlink(self, dataset_id: UUID, version_path: Path) -> None:
        symlink_path = self._data_symlink_path(dataset_id)
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(version_path.name)

    def _normalize_metadata(self, metadata: Mapping[str, Any] | Any) -> dict[str, Any]:
        if isinstance(metadata, Mapping):
            return dict(metadata)
        if is_dataclass(metadata):
            return asdict(metadata)
        msg = f"metadata must be a mapping or dataclass instance; got {type(metadata).__name__}"
        raise TypeError(msg)

    def _read_metadata(self, dataset_id: UUID) -> dict[str, Any]:
        metadata_path = self._metadata_symlink_path(dataset_id)
        if not metadata_path.is_file():
            msg = f"metadata for dataset {dataset_id} does not exist"
            raise FileNotFoundError(msg)
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    @contextmanager
    def _locked(self, dataset_id: UUID):
        lock = self._get_lock(dataset_id)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def _get_lock(self, dataset_id: UUID) -> threading.Lock:
        with self._locks_lock:
            lock = self._dataset_locks.get(dataset_id)
            if lock is None:
                lock = threading.Lock()
                self._dataset_locks[dataset_id] = lock
        return lock

    @staticmethod
    def _timestamp_ms() -> int:
        return int(time.time_ns() // 1_000_000)
