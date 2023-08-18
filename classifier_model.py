from transformers import BertForSequenceClassification, BertTokenizer
from preprocessing_data import preprocessing_mytelkom
import torch
import torch.nn.functional as F


model_path = r'bagas10/MyTelkomselSentimentBert'

# Load a trained model and vocabulary that you have fine-tuned
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Copy the model to the GPU.
model.to('cpu')

i2w = {0: 'positive', 1: 'neutral', 2: 'negative'}


def klasifikasi_mytelkom(review):
  text = preprocessing_mytelkom(review)

  subwords = tokenizer.encode(text)
  subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

  logits = model(subwords)[0]
  label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

  return i2w[label], f'{F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%'
    
