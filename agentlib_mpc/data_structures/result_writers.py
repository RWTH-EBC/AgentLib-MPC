# agentlib_mpc/data_structures/result_writers.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Type
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class ResultWriter(ABC):
    """Abstract base class for result writers."""

    def __init__(self, file_path: Path, options: dict = None):
        self.file_path = Path(file_path)
        self.options = options or {}

    @abstractmethod
    def write_results(self, df: pd.DataFrame, time: float, iteration: int = 0):
        """Write results for a single time step/iteration."""
        pass

    @abstractmethod
    def write_stats(self, stats_line: str):
        """Write optimization statistics."""
        pass

    @abstractmethod
    def initialize(self, df: pd.DataFrame):
        """Initialize the file with column headers/structure."""
        pass

    @abstractmethod
    def finalize(self):
        """Cleanup and finalize the file."""
        pass


class ResultWriterFactory:
    """Factory to create appropriate writer based on file extension."""

    writers: Dict[str, Type[ResultWriter]] = {}

    @classmethod
    def register(cls, extension: str, writer_class: Type[ResultWriter]):
        """Register a writer for a file extension."""
        cls.writers[extension.lower()] = writer_class

    @classmethod
    def create(cls, file_path: Path, options: dict = None) -> ResultWriter:
        """Create appropriate writer based on file extension."""
        extension = file_path.suffix.lower()

        if extension not in cls.writers:
            # Default to CSV for unknown extensions
            logger.warning(f"Unknown extension {extension}, defaulting to CSV")
            extension = ".csv"

        writer_class = cls.writers[extension]
        return writer_class(file_path, options)


# agentlib_mpc/data_structures/result_writers.py (continued)


class HDF5ResultWriter(ResultWriter):
    """HDF5 writer with robustness features."""

    def __init__(self, file_path: Path, options: dict = None):
        if not HAS_H5PY:
            from agentlib.core.errors import OptionalDependencyError

            raise OptionalDependencyError(
                used_object="HDF5 result writer",
                dependency_install="agentlib[hdf5]",
                dependency_name="hdf5",
            )

        super().__init__(file_path, options)
        self.compression = options.get("compression", "gzip")
        self.compression_level = options.get("compression_level", 6)
        self.chunk_cache_size = options.get("chunk_cache_size", 10 * 1024**2)  # 10MB
        self.swmr = options.get("swmr", True)  # Single Writer Multiple Readers
        self._file_handle: Optional[h5py.File] = None
        self._buffer_size = options.get("buffer_size", 1)  # Buffer iterations
        self._buffer: list = []
        self._stats_buffer: list = []
        self._initialized = False

    def _ensure_file_open(self):
        """Ensure file handle is open, opening if necessary."""
        if self._file_handle is None:
            if self.file_path.exists():
                # Reopen existing file
                self._file_handle = h5py.File(
                    self.file_path,
                    "r+",  # read/write mode for existing file
                    libver="latest" if self.swmr else "earliest",
                )
                if self.swmr and not self._file_handle.swmr_mode:
                    self._file_handle.swmr_mode = True
                self._initialized = True
            else:
                # File doesn't exist, need to call initialize()
                raise RuntimeError(
                    f"HDF5 file {self.file_path} doesn't exist. "
                    "Call initialize() first."
                )

    def initialize(self, df: pd.DataFrame):
        """Initialize HDF5 file structure."""
        # Create file with SWMR mode if requested
        self._file_handle = h5py.File(
            self.file_path, "w", libver="latest" if self.swmr else "earliest"
        )

        # Store metadata
        self._file_handle.attrs["created"] = pd.Timestamp.now().isoformat()
        self._file_handle.attrs["format_version"] = "1.0"

        # Create groups
        self._file_handle.create_group("results")
        self._file_handle.create_group("stats")

        # Store column information
        self._store_column_info(df)

        if self.swmr:
            self._file_handle.swmr_mode = True

        self._initialized = True

    def _store_column_info(self, df: pd.DataFrame):
        """Store column names and structure for later reconstruction."""
        meta_group = self._file_handle.create_group("metadata")

        # For MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            for i, level in enumerate(df.columns.names):
                meta_group.attrs[f"column_name_{i}"] = level or f"level_{i}"

            # Store the full column tuples - encode as bytes
            col_data = np.array(
                [str(col) for col in df.columns], dtype="S"
            )  # 'S' for byte string
            meta_group.create_dataset("columns", data=col_data)
            meta_group.attrs["n_column_levels"] = df.columns.nlevels
        else:
            # Encode column names as bytes
            col_data = np.array(df.columns.values, dtype="S")
            meta_group.create_dataset("columns", data=col_data)
            meta_group.attrs["n_column_levels"] = 1

    def write_results(self, df: pd.DataFrame, time: float, iteration: int = 0):
        """Write results with buffering for efficiency."""
        # Ensure file is open
        self._ensure_file_open()

        # Add to buffer
        self._buffer.append({"time": time, "iteration": iteration, "data": df})

        # Flush buffer if it's full
        if len(self._buffer) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Flush buffer to disk."""
        if not self._buffer:
            return

        # Ensure file is open
        self._ensure_file_open()

        for item in self._buffer:
            time = item["time"]
            iteration = item["iteration"]
            df = item["data"]

            # Create time group if it doesn't exist
            time_group = self._file_handle["results"].require_group(f"time_{time}")

            # Create iteration dataset
            iter_name = f"iter_{iteration}"

            # Convert DataFrame to numpy array
            data = df.values

            # Store with compression
            time_group.create_dataset(
                iter_name,
                data=data,
                compression=self.compression,
                compression_opts=self.compression_level,
                chunks=True,  # Enable chunking for better performance
            )

            # Store index information
            if iter_name + "_index" not in time_group:
                time_group.create_dataset(iter_name + "_index", data=df.index.values)

        # Clear buffer
        self._buffer.clear()

        # Flush to disk
        self._file_handle.flush()

    def write_stats(self, stats_line: str):
        """Buffer and write statistics."""
        # Ensure file is open
        self._ensure_file_open()

        self._stats_buffer.append(stats_line)

        if len(self._stats_buffer) >= self._buffer_size:
            self._flush_stats_buffer()

    def _flush_stats_buffer(self):
        """Flush statistics buffer."""
        if not self._stats_buffer:
            return

        # Ensure file is open
        self._ensure_file_open()

        stats_group = self._file_handle["stats"]

        # Get current number of stats entries
        n_existing = len(stats_group.keys())

        for i, line in enumerate(self._stats_buffer):
            stats_group.create_dataset(
                f"entry_{n_existing + i}", data=line.encode("utf-8")
            )

        self._stats_buffer.clear()
        self._file_handle.flush()

    def finalize(self):
        """Finalize file and cleanup."""
        if self._file_handle:
            # Flush any remaining data
            self._flush_buffer()
            self._flush_stats_buffer()

            # Store final metadata
            self._file_handle.attrs["finalized"] = pd.Timestamp.now().isoformat()

            # Close file
            self._file_handle.close()
            self._file_handle = None
            self._initialized = False


# agentlib_mpc/data_structures/result_writers.py (continued)


class CSVResultWriter(ResultWriter):
    """CSV writer maintaining backward compatibility."""

    def __init__(self, file_path: Path, options: dict = None):
        super().__init__(file_path, options)
        self.stats_path = Path(str(file_path).replace(".csv", "_stats.csv"))
        self._initialized = False

    def initialize(self, df: pd.DataFrame):
        """Write CSV headers."""
        # Write the header row
        df.to_csv(self.file_path, mode="w", header=True, index=False)
        self._initialized = True

        # Initialize stats file with header if needed
        # This depends on your stats format

    def write_results(self, df: pd.DataFrame, time: float, iteration: int = 0):
        """Append results to CSV."""
        # Don't modify the index if it's already been formatted (ADMM case)
        if not df.index[0].startswith("("):
            # Format index to match existing behavior
            if iteration == 0:  # MPC case
                df.index = [f"({time}, {idx})" for idx in df.index]
            else:  # ADMM case - this shouldn't happen as ADMM pre-formats
                df.index = [f"({time}, {iteration}, {idx})" for idx in df.index]

        # Append to file
        df.to_csv(self.file_path, mode="a", header=False)

    def write_stats(self, stats_line: str):
        """Write statistics line."""
        with open(self.stats_path, "a") as f:
            f.write(stats_line)

    def finalize(self):
        """No special finalization needed for CSV."""
        pass

        # Append to file
        mode = "a" if self._initialized else "w"
        header = not self._initialized
        df.to_csv(self.file_path, mode=mode, header=header)

        if not self._initialized:
            self._initialized = True

    def write_stats(self, stats_line: str):
        """Write statistics line."""
        with open(self.stats_path, "a") as f:
            f.write(stats_line)

    def finalize(self):
        """No special finalization needed for CSV."""
        pass


# Register writers
ResultWriterFactory.register(".csv", CSVResultWriter)
if HAS_H5PY:
    ResultWriterFactory.register(".h5", HDF5ResultWriter)
    ResultWriterFactory.register(".hdf5", HDF5ResultWriter)
