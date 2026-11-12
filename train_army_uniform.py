from ultralytics import YOLO
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "army_uniform_dataset"

RUNS_DIR = BASE_DIR / "runs"

RUN_NAME = "army_uniform_classifier"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("       ARMY UNIFORM CLASSIFIER TRAINING")
    print("=" * 60)

    print()
    print("Project folder:")
    print(BASE_DIR)

    print()
    print("Dataset folder:")
    print(DATASET_DIR)

    print()
    print("Checking dataset...")

    train_army = DATASET_DIR / "train" / "army"
    train_non_army = DATASET_DIR / "train" / "non_army"
    val_army = DATASET_DIR / "val" / "army"
    val_non_army = DATASET_DIR / "val" / "non_army"

    required_folders = [
        train_army,
        train_non_army,
        val_army,
        val_non_army
    ]

    for folder in required_folders:
        if not folder.exists():
            print()
            print("ERROR: Missing folder:")
            print(folder)
            return

    print("Dataset folders found successfully.")

    print()
    print("Loading classification model...")

    # YOLO classification model
    model = YOLO("yolo11n-cls.pt")

    print()
    print("Starting training...")

    model.train(
        data=str(DATASET_DIR),
        epochs=50,
        imgsz=224,
        batch=4,
        patience=10,
        device="cpu",
        workers=0,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=False,
        verbose=True
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print()
    print("Model output:")
    print(RUNS_DIR / RUN_NAME)


if __name__ == "__main__":
    main()