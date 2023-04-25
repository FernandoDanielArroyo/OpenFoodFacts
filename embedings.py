from PIL import Image
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model_checkpoints')
args = parser.parse_args()

model_ckpt = args.model_checkpoints
# model_ckpt = 'microsoft/swin-tiny-patch4-window7-224'
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to('cuda:0')
data = []
index = []
path = Path('data/img')

total = len(list(path.rglob('*.jpg')))
for file in tqdm(path.rglob('*.jpg'), total=total):
    # print(file)
    image = Image.open(file)
    try:
        inputs = extractor(image, return_tensors="pt").to('cuda:0')
        # print(inputs['pixel_values'].shape)
        with torch.no_grad():
            outputs = model(**inputs)
        # print(outputs)
        last_hidden_states = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        # print(last_hidden_states.shape)
        data.append(last_hidden_states)
        index.append(file.stem)
    except ValueError:
        continue

df = pd.DataFrame(data)
df.index = index
print(df.head())
df.to_csv(f'data/{model_ckpt.split("/")[1]}.csv')





