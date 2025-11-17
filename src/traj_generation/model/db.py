import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Union
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
        pq.write_table(combined_table, file_path, schema=schema)
    else:
        pq.write_table(table, file_path, schema=schema)


def read_from_parquet(path: str) -> List[SyntheticBaseballData]:
    """
    Read SyntheticBaseballData from a Parquet file.
    
    Args:
        path: Path to Parquet file
        
    Returns:
        List of SyntheticBaseballData instances
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    
    # Read table from Parquet
    table = pq.read_table(path)
    
    # Convert to list of dictionaries
    records = table.to_pylist()
    
    # Convert to SyntheticBaseballData instances
    return [SyntheticBaseballData(**record) for record in records]