from transformers import pipeline
classifier = pipeline("image-classification")
result = classifier("C:\Image processing\OIP.jpg")
print(result)