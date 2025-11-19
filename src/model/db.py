import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from typing import List, Union, Literal
from dataclasses import dataclass

@dataclass
class SyntheticBaseballData: 
    image_path: str
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    bbox_area: float
    date: str

def create_parquet_schema() -> pa.Schema: 
    return pa.schema([
        pa.field("image_path", pa.string()),
        pa.field("bbox_x", pa.float64()),
        pa.field("bbox_y", pa.float64()),
        pa.field("bbox_width", pa.float64()),
        pa.field("bbox_height", pa.float64()),
        pa.field("bbox_area", pa.float64()),
        pa.field("date", pa.string()),
    ])

def _data_to_dict(data: SyntheticBaseballData) -> dict:
    """Convert SyntheticBaseballData instance to dictionary."""
    return {
        "image_path": data.image_path,
        "bbox_x": data.bbox_x,
        "bbox_y": data.bbox_y,
        "bbox_width": data.bbox_width,
        "bbox_height": data.bbox_height,
        "bbox_area": data.bbox_area,
        "date": data.date,
    }


def write_to_parquet(data: Union[SyntheticBaseballData, List[SyntheticBaseballData]], path: str, append: bool = False):
    """
    Write SyntheticBaseballData to a Parquet file on disk.
    
    Args:
        data: Single SyntheticBaseballData instance or list of instances
        path: Path to output Parquet file (e.g., "data/training.parquet")
        append: If True, append to existing file. If False, overwrite.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    schema = create_parquet_schema()
    
    if isinstance(data, SyntheticBaseballData):
        records = [_data_to_dict(data)]
    else:
        records = [_data_to_dict(d) for d in data]
    
    table = pa.Table.from_pylist(records, schema=schema)
    if append and file_path.exists():
        existing_table = pq.read_table(file_path)
        combined_table = pa.concat_tables([existing_table, table])
        pq.write_table(combined_table, file_path)
    else:
        pq.write_table(table, file_path)


def read_from_parquet(path: str, format: Literal["dataframe", "table", "objects"] = "dataframe"):
    """
    Read data from a Parquet file in various formats.
    
    Args:
        path: Path to Parquet file
        format: Output format:
            - "dataframe": pandas DataFrame (fastest, most efficient for most operations)
            - "table": PyArrow Table (most memory efficient, columnar operations)
            - "objects": List of SyntheticBaseballData (slowest, but type-safe)
        
    Returns:
        Depending on format:
        - "dataframe": pd.DataFrame
        - "table": pa.Table
        - "objects": List[SyntheticBaseballData]
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    
    if format == "dataframe":
        # Fastest: Direct pandas read (uses PyArrow under the hood)
        return pd.read_parquet(path, engine="pyarrow")
    
    elif format == "table":
        # Most memory efficient: Keep as PyArrow Table
        return pq.read_table(path)
    
    elif format == "objects":
        # Slowest: Convert to Python objects (kept for backward compatibility)
        table = pq.read_table(path)
        records = table.to_pylist()
        return [SyntheticBaseballData(**record) for record in records]
    
    else:
        raise ValueError(f"Unknown format: {format}. Must be 'dataframe', 'table', or 'objects'")


def read_from_parquet_iter(path: str, batch_size: int = 1000):
    """
    Read Parquet file in batches (memory-efficient for large files).
    
    Args:
        path: Path to Parquet file
        batch_size: Number of rows per batch
        
    Yields:
        pd.DataFrame batches
    """
    parquet_file = pq.ParquetFile(path)
    
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()