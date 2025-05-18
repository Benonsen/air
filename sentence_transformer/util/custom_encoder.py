import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

class EncoderAutoModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)


    def forward(self, tweet_ids, tweet_mask, paper_ids, paper_mask):
        tweet_vec = self.encoder(tweet_ids, attention_mask=tweet_mask).last_hidden_state[:, 0]
        paper_vec = self.encoder(paper_ids, attention_mask=paper_mask).last_hidden_state[:, 0]
        return tweet_vec, paper_vec
    
class EncoderSentenceTransformer(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)

    def forward(self, tweet_ids, tweet_mask, paper_ids, paper_mask):
        # Create feature dictionaries expected by SentenceTransformer
        tweet_features = {'input_ids': tweet_ids, 'attention_mask': tweet_mask}
        paper_features = {'input_ids': paper_ids, 'attention_mask': paper_mask}
        
        # Pass dictionaries to the encoder
        tweet_output = self.encoder(tweet_features)
        paper_output = self.encoder(paper_features)
        
        # Extract the sentence embeddings from the output dictionaries
        tweet_vec = tweet_output['sentence_embedding']
        paper_vec = paper_output['sentence_embedding']
        
        return tweet_vec, paper_vec
    

def encode_papers_auto(model, device, df_collection, tokenizer, batch_size=64):
    model.eval()
    paper_texts = df_collection.apply(lambda row: f"{row['title']} {row['abstract']}", axis=1).tolist()
    paper_ids = df_collection['cord_uid'].tolist()

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(paper_texts), batch_size):
            batch = paper_texts[i:i+batch_size]
            encodings = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            vecs = model.encoder(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            all_embeddings.append(vecs.cpu().numpy())

    return paper_ids, np.vstack(all_embeddings)

def encode_papers_sbert(model, df_collection, batch_size=64):
    paper_texts = df_collection.apply(lambda row: f"{row['title']} {row['abstract']}", axis=1).tolist()
    paper_ids = df_collection['cord_uid'].tolist()
    # Use the encode method which is designed to handle text inputs directly
    embeddings = model.encoder.encode(paper_texts, batch_size=batch_size, 
                                    convert_to_numpy=True, show_progress_bar=True)
    return paper_ids, embeddings

def retrieve_sbert(model, df_query, faiss_index, paper_id_map, topk=10):
    query_texts = df_query['tweet_text'].tolist()
    
    # Use the encode method to generate embeddings for queries
    query_embeddings = model.encoder.encode(query_texts, convert_to_numpy=True, 
                                          show_progress_bar=True)
    
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

def retrieve_auto(model, device, df_query, tokenizer, faiss_index, paper_id_map, topk=10):
    model.eval()
    predictions = []

    with torch.no_grad():
        for text in df_query['tweet_text']:
            enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            tweet_vec = model.encoder(enc['input_ids'], attention_mask=enc['attention_mask']).last_hidden_state[:, 0]
            tweet_vec = F.normalize(tweet_vec, dim=1).cpu().numpy()

            D, I = faiss_index.search(tweet_vec, topk)
            
            preds = [paper_id_map[idx] for idx in I[0]]
            predictions.append(preds)

    df_query['dense_topk'] = predictions
    