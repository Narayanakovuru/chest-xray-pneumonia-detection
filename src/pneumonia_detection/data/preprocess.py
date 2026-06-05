import pandas as pd
from pathlib import Path
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def load_metadata(csv_path: Path) -> pd.DataFrame:
    """Load the metadata CSV and isolate essential features (patientId, Target).

    Args:
        csv_path (Path): Path to the stage2 training metadata CSV file.

    Returns:
        pd.DataFrame: Cleaned pandas DataFrame.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV file not found at: {path}")
        
    logger.info(f"Loading metadata from {path}")
    df = pd.read_csv(path)
    
    # Assert essential columns exist
    required_cols = {"patientId", "Target"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Metadata CSV is missing one or more required columns: {required_cols}")
        
    df = df[["patientId", "Target"]]
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    return df

def get_class_distribution(df: pd.DataFrame) -> Tuple[int, int, float]:
    """Calculate the target class distribution details (negative and positive counts).

    Args:
        df (pd.DataFrame): Dataframe containing 'Target' column.

    Returns:
        Tuple[int, int, float]:
            neg_count, pos_count, and pos_weight ratio.
    """
    neg_count = int((df["Target"] == 0).sum())
    pos_count = int((df["Target"] == 1).sum())
    
    if pos_count == 0:
        raise ValueError("Positive class count is zero. Cannot compute class weights.")
        
    pos_weight = float(neg_count / pos_count)
    logger.info(f"Class distribution: Negative={neg_count} | Positive={pos_count} | pos_weight ratio={pos_weight:.4f}")
    return neg_count, pos_count, pos_weight
