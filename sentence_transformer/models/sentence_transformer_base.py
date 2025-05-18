import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import faiss
from torch.utils.data import DataLoader

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from util.evaluate import get_performance_mrr
from util.custom_encoder import EncoderSentenceTransformer
from util.dataset import ClaimSourceDataset
from util.loss_functions import contrastive_loss
from util.custom_encoder import encode_papers_sbert
from util.custom_encoder import retrieve_sbert

# Load the data
PATH_COLLECTION_DATA = '../subtask4b_collection_data.pkl'
PATH_QUERY_TRAIN_DATA = '../subtask4b_query_tweets_train.tsv'
PATH_QUERY_DEV_DATA = '../subtask4b_query_tweets_dev.tsv'
PATH_QUERY_TEST_DATA = '../subtask4b_query_tweets_test.tsv'

# Model specification
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' #TODO change name of the model
tokenizer = SentenceTransformer(MODEL_NAME).tokenizer
encoder = EncoderSentenceTransformer(model_name=MODEL_NAME)
optimizer = torch.optim.AdamW(encoder.parameters(), lr=2e-5)

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder.to(device)
print(f"Using device: {device}")

# Load the collection and query data
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
df_query_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep='\t')
df_query_dev = pd.read_csv(PATH_QUERY_DEV_DATA, sep='\t')
df_query_test = pd.read_csv(PATH_QUERY_TEST_DATA, sep='\t')

train_dataset = ClaimSourceDataset(df_query_train, df_collection, tokenizer, 256) # TODO change max_len
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

print(f"Data ready. Number of training samples: {len(train_loader.dataset)}")

# finetune the model
for epoch in range(5):
    encoder.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        tweet_ids = batch['tweet_input_ids'].to(device)
        tweet_mask = batch['tweet_attention_mask'].to(device)
        paper_ids = batch['paper_input_ids'].to(device)
        paper_mask = batch['paper_attention_mask'].to(device)

        tweet_vecs, paper_vecs = encoder(tweet_ids, tweet_mask, paper_ids, paper_mask)

        loss = contrastive_loss(tweet_vecs, paper_vecs)
   
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

print("Training 1 complete.")

# encode the collection
paper_ids, paper_embeddings = encode_papers_sbert(encoder, df_collection, )

# create index for fast retrieval
faiss_index = faiss.IndexFlatIP(paper_embeddings.shape[1])
faiss_index.add(paper_embeddings)
paper_id_map = {i: pid for i, pid in enumerate(paper_ids)}

# Create embeddings for dev queries
retrieve_sbert(encoder, df_query_dev, faiss_index, paper_id_map)
retrieve_sbert(encoder, df_query_test, faiss_index, paper_id_map)

# Get resutls
results_test = get_performance_mrr(df_query_dev, 'cord_uid', 'dense_topk')
print("MRR Results:", results_test)

# Write results to TSV file
df_query_test[['post_id', 'preds']].to_csv('results/predictions_transformer_base.tsv', index=None, sep='\t')
print("Predictions written to predictions_transformer_base.tsv")

# Save model weights
model_save_path = "base_sentence_transformer.pt"
torch.save(encoder.state_dict(), model_save_path)