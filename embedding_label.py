import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, T5EncoderModel

if __name__ == "__main__":
    print('getting embedding of product names using t5-small')
    df = pd.read_csv('data/label_df.csv', index_col=0)
    with open('data/images_downloaded.txt', 'r') as f:
        lines = f.read().strip().split('\n')
    selected_data = df.loc[lines]
    print(selected_data.head())
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5EncoderModel.from_pretrained("t5-small")

    embedding = []
    for i, row in enumerate(selected_data.itertuples()):
        if type(row.product_name) != 'str':
            print('nan is replaced with random values')
            hidden_state = np.random.rand(512)
            embedding.append(torch.from_numpy(hidden_state))
        else:
            input_ids = tokenizer(
                row.product_name, return_tensors="pt"
            ).input_ids  
            outputs = model(input_ids=input_ids)

            last_hidden_states = outputs.last_hidden_state.detach() # shape [batch, seq_len, embedding_dim]
            embedding.append(last_hidden_states.mean(1)[0]) # get averaged embedding of sequence
    embedding = torch.stack(embedding)
    embedding = pd.DataFrame(embedding)
    embedding.index = selected_data.index
    embedding.to_csv('data/embeddings/product-name_t5-small.csv')