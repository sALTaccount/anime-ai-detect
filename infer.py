from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import requests
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--url')
parser.add_argument('--image')
args = parser.parse_args()

if args.url:
    image = Image.open(requests.get(args.url, stream=True).raw).convert('RGB')
elif args.image:
    image = Image.open(args.image).convert('RGB')
else:
    print('Please chose a url or image!')
    parser.print_usage()
    exit()
feature_extractor = BeitFeatureExtractor.from_pretrained('saltacc/anime-ai-detect')
model = BeitForImageClassification.from_pretrained('saltacc/anime-ai-detect')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
probs = logits.softmax(-1)
predicted_class_idx = probs.argmax().item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
print("Probability:", probs[0][predicted_class_idx].item())
