from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment").to(device)

def get_sentiment(text):

    inputs = tokenizer(text, return_tensors= "pt", truncation = True,
                    padding = True, max_length = 512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim = 1)

    return torch.argmax(probs).item(), probs.cpu().numpy()