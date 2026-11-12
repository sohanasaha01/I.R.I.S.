from army_uniform_classifier import ArmyUniformClassifier


classifier = ArmyUniformClassifier()


image_path = r"C:\border_project\army_uniform_dataset\val\army\army10.png"


from cv2 import imread

image = imread(image_path)


label, confidence = classifier.classify(image)


print()
print("====================================")
print("ARMY UNIFORM CLASSIFIER TEST")
print("====================================")
print("Image:", image_path)
print("Prediction:", label)
print("Confidence:", confidence)
print("====================================")