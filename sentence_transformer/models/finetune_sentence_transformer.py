from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers import SentenceTransformer, models, losses
from torch.utils.data import DataLoader
import pandas as pd
import torch

# Load the data
PATH_COLLECTION_DATA = '../subtask4b_collection_data.pkl'
PATH_QUERY_TRAIN_DATA = '../subtask4b_query_tweets_train.tsv'
PATH_QUERY_DEV_DATA = '../subtask4b_query_tweets_dev.tsv'
PATH_QUERY_TEST_DATA = '../subtask4b_query_tweets_test.tsv'

# Model specification
model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

# device
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model.to(device)
#print(f"Using device: {device}")

# Load the collection and query data
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)

print(df_collection.head())

# create new column combining title and abstract
df_collection['combined'] = df_collection['title'] + ' ' + df_collection['abstract']

# Load your data (list of scientific sentences)
sentences = df_collection['combined'].tolist()

# Truncate extremely long sentences if needed
max_length = 256
sentences = [sentence[:max_length] for sentence in sentences]

# Wrap into dataset and dataloader
train_dataset = DenoisingAutoEncoderDataset(sentences, model.tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define loss
train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path='microsoft/MiniLM-L6-H384-uncased')

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=1000,
    output_path='./fine-tuned-tsdae-model'
)