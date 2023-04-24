from PIL import Image
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel


model_ckpt = 'google/vit-base-patch16-224-in21k'
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
data = []
index = []
path = Path('data/img')

for file in path.rglob('*.jpg'):
    print(file)
    image = Image.open(file)
    try:
        inputs = extractor(image, return_tensors="pt")
        print(inputs['pixel_values'].shape)
        with torch.no_grad():
            outputs = model(**inputs)
        # print(outputs)
        last_hidden_states = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        print(last_hidden_states.shape)
        data.append(last_hidden_states)
        index.append(file.stem)
    except ValueError:
        continue

   

df = pd.DataFrame(data)
df.index = index
print(df.head())
df.to_csv('data/embedding_vit.csv')


# listing all image files 




