from pathlib import Path

LABEL_DIR = Path(r"C:\border_project\dataset\labels\all")

print("=" * 50)
print("YOLO LABEL VALIDATION")
print("=" * 50)

label_files = list(LABEL_DIR.glob("*.txt"))

print(f"Checking {len(label_files)} label files...")

invalid = 0

for file in sorted(label_files):
    lines = file.read_text().strip().splitlines()

    for line_no, line in enumerate(lines, start=1):
        parts = line.split()

        # YOLO format:
        # class_id x_center y_center width height
        if len(parts) != 5:
            print(f"INVALID: {file.name} line {line_no}")
            print(f"  Expected 5 values, got {len(parts)}")
            invalid += 1
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            print(f"INVALID: {file.name} line {line_no}")
            print("  Contains non-numeric values")
            invalid += 1
            continue

        # Valid class IDs: 0-5
        if class_id < 0 or class_id > 5:
            print(f"INVALID: {file.name} line {line_no}")
            print(f"  Invalid class ID: {class_id}")
            invalid += 1
            continue

        # YOLO coordinates must be between 0 and 1
        values = [x_center, y_center, width, height]

        if any(value < 0 or value > 1 for value in values):
            print(f"INVALID: {file.name} line {line_no}")
            print(f"  Values outside 0-1 range: {values}")
            invalid += 1
            continue

        # Width and height must be greater than zero
        if width <= 0 or height <= 0:
            print(f"INVALID: {file.name} line {line_no}")
            print("  Width or height is zero/negative")
            invalid += 1
            continue

print()
print("=" * 50)
print("VALIDATION RESULT")
print("=" * 50)

if invalid == 0:
    print("INVALID: 0")
    print("All YOLO labels are valid.")
else:
    print(f"INVALID: {invalid}")
    print("Please fix the invalid labels before training.")

print("=" * 50)