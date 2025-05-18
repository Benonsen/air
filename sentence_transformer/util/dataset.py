from torch.utils.data import Dataset

class ClaimSourceDataset(Dataset):
    def __init__(self, df, collection_df, tokenizer, max_len):
        self.df = df
        self.collection = collection_df.set_index('cord_uid')
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tweet = row['tweet_text']
        paper_id = row['cord_uid']
        paper_row = self.collection.loc[paper_id]
        paper = f"{paper_row['title']} {paper_row['abstract']}"

        tweet_enc = self.tokenizer(tweet, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        paper_enc = self.tokenizer(paper, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")

        return {
            'tweet_input_ids': tweet_enc['input_ids'].squeeze(),
            'tweet_attention_mask': tweet_enc['attention_mask'].squeeze(),
            'paper_input_ids': paper_enc['input_ids'].squeeze(),
            'paper_attention_mask': paper_enc['attention_mask'].squeeze()
        }