import os
import tiktoken
import numpy as np
import polars as pl

# Login using e.g. `huggingface-cli login` to access this dataset
# Load the dataset from Hugging Face using Polars
df = pl.read_parquet('hf://datasets/Selvakumarduraipandian/Thirukural/data/train-00000-of-00001.parquet')

# Check the columns of the dataframe to identify the correct text field
print(df.columns)

# Assuming the text column is named "text", adjust as necessary if the column name differs
full_text = "\n".join(df['Transliteration'].to_list())

# Split the data into training and validation (90% train, 10% validation)
n = len(full_text)
train_data = full_text[:int(n * 0.9)]
val_data = full_text[int(n * 0.9):]

# Encode the data using tiktoken (GPT-2 BPE encoding)
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Convert to numpy arrays and save to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# Save the binary files to disk
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
