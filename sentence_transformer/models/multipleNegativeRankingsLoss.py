import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import faiss

from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# Load the data
PATH_COLLECTION_DATA = '../subtask4b_collection_data.pkl'
PATH_QUERY_TRAIN_DATA = '../subtask4b_query_tweets_train.tsv'
PATH_QUERY_DEV_DATA = '../subtask4b_query_tweets_dev.tsv'
PATH_QUERY_TEST_DATA = '../subtask4b_query_tweets_test.tsv'

model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Using device: {device}")

# Load the collection and query data
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
df_query_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep='\t')
df_query_dev = pd.read_csv(PATH_QUERY_DEV_DATA, sep='\t')
df_query_test = pd.read_csv(PATH_QUERY_TEST_DATA, sep='\t')

def encode_papers_sbert(model, df_collection, batch_size=64):
    paper_texts = df_collection.apply(lambda row: f"{row['title']} {row['abstract']}", axis=1).tolist()
    paper_ids = df_collection['cord_uid'].tolist()
    
    # Use the encode method from the SentenceTransformer object directly
    embeddings = model.encode(
        sentences=paper_texts,  # Make sure to use the named parameter
        batch_size=batch_size, 
        convert_to_numpy=True, 
        show_progress_bar=True,
        device=device 
    )
    return paper_ids, embeddings

def retrieve_sbert(model, df_query, faiss_index, paper_id_map, topk=10):
    query_texts = df_query['tweet_text'].tolist()
    
    # Use the encode method to generate embeddings for queries
    query_embeddings = model.encode(query_texts, 
                                    convert_to_numpy=True,     
                                    show_progress_bar=True,
                                    device=device)
    
    predictions = []
    for query_vec in query_embeddings:
        # Make sure query_vec is properly shaped for FAISS search
        query_vec_reshaped = query_vec.reshape(1, -1)
        D, I = faiss_index.search(query_vec_reshaped, topk)
        preds = [paper_id_map[idx] for idx in I[0]]
        predictions.append(preds)

    # Store predictions in the dataframe
    df_query['dense_topk'] = predictions
    df_query['preds'] = df_query['dense_topk'].apply(lambda x: ' '.join(x))
    return df_query    

def get_performance_mrr(data, col_gold, col_pred, list_k = [1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in [i for i in x[col_pred][:k]] else 0), axis=1)
        #performances.append(data["in_topx"].mean())
        d_performance[k] = data["in_topx"].mean()
    return d_performance


# create list of InputExample objects for training
train_examples = []
for _, row in df_query_train.iterrows():
    tweet = row['tweet_text']
    paper_id = row['cord_uid']
    paper_row = df_collection[df_collection['cord_uid'] == paper_id].iloc[0]
    paper = f"{paper_row['title']} {paper_row['abstract']}"
    train_examples.append(InputExample(texts=[tweet, paper], label=1))

# Create DataLoader for training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=10,
    show_progress_bar=True
)

model.save("./positive-only-sbert")

# encode the collection
paper_ids, paper_embeddings = encode_papers_sbert(model, df_collection)

# create index for fast retrieval
faiss_index = faiss.IndexFlatIP(paper_embeddings.shape[1])
faiss_index.add(paper_embeddings)
paper_id_map = {i: pid for i, pid in enumerate(paper_ids)}

# Create embeddings for dev queries
retrieve_sbert(model, df_query_dev, faiss_index, paper_id_map)
retrieve_sbert(model, df_query_test, faiss_index, paper_id_map)

# Get resutls
results_test = get_performance_mrr(df_query_dev, 'cord_uid', 'dense_topk')
print("MRR Results:", results_test)

# Write results to TSV file
df_query_test[['post_id', 'preds']].to_csv('results/predictions_transformer_base.tsv', index=None, sep='\t')
print("Predictions written to predictions_transformer_base.tsv")