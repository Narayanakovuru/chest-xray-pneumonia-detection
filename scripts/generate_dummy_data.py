import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

def generate_dummy_dataset(target_dir: str = "data/raw", num_samples: int = 50):
    """Generate a mock dataset containing synthetic PNG files and a metadata CSV.

    Args:
        target_dir (str): Base folder to write the mock data.
        num_samples (int): Number of mock samples to generate.
    """
    base_path = Path(target_dir)
    
    # 1. Create target directories matching configs/data.yaml
    img_dir = base_path / "Training" / "Images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Generate random patient IDs and targets (labels)
    patient_ids = [f"dummy_patient_{i:04d}" for i in range(num_samples)]
    # Create class imbalance (35 normal, 15 pneumonia)
    targets = [0 if i % 3 != 0 else 1 for i in range(num_samples)]
    
    # 3. Create and save metadata CSV file
    df = pd.DataFrame({
        "patientId": patient_ids,
        "Target": targets
    })
    csv_path = base_path / "stage2_train_metadata.csv"
    df.to_csv(csv_path, index=False)
    
    # 4. Generate and save synthetic grayscale images (mode "L")
    print(f"Generating {num_samples} synthetic images in {img_dir}...")
    for pid in patient_ids:
        # Create a 224x224 grayscale mock X-ray image (random noise)
        img_data = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
        img = Image.fromarray(img_data, mode="L")
        img.save(img_dir / f"{pid}.png")
        
    print(f"Metadata CSV successfully saved to: {csv_path.resolve()}")
    print("Dummy dataset generation complete. You can now run the training pipeline!")

if __name__ == "__main__":
    generate_dummy_dataset()
