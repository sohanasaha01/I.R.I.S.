from pathlib import Path
from ultralytics import YOLO


class ArmyUniformClassifier:

    def __init__(self, model_path=None, confidence_threshold=0.60):

        # Project folder
        base_dir = Path(__file__).resolve().parent

        # If no model path is provided, use our trained model
        if model_path is None:
            model_path = (
                base_dir
                / "runs"
                / "army_uniform_classifier"
                / "weights"
                / "best.pt"
            )

        self.model_path = str(model_path)

        # Minimum confidence required
        self.confidence_threshold = confidence_threshold

        print("Loading Army Uniform Classifier...")
        print("Model:", self.model_path)

        self.model = YOLO(self.model_path)

        print("Army Uniform Classifier loaded successfully.")


    def classify(self, person_crop):

        """
        Classifies a cropped person image as:

        army
        non_army

        Returns:
            label, confidence
        """

        if person_crop is None:
            return "unknown", 0.0

        # Run classification
        results = self.model.predict(
            source=person_crop,
            verbose=False
        )

        if not results:
            return "unknown", 0.0

        result = results[0]

        # Classification probabilities
        if result.probs is None:
            return "unknown", 0.0

        top_class = result.probs.top1
        confidence = float(result.probs.top1conf)

        label = result.names[top_class]

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            return "unknown", confidence

        return label, confidence