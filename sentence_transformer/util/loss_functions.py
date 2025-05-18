import torch
import torch.nn.functional as F

def contrastive_loss(tweet_vecs, paper_vecs, temperature=0.05):
    tweet_vecs = F.normalize(tweet_vecs, dim=1)
    paper_vecs = F.normalize(paper_vecs, dim=1)

    logits = torch.matmul(tweet_vecs, paper_vecs.T) / temperature
    labels = torch.arange(len(tweet_vecs)).to(tweet_vecs.device)
    return F.cross_entropy(logits, labels)