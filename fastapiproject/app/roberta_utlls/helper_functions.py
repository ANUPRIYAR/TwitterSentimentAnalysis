import re
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizer

batch_size = 8
Max_length = 120
pre_trained_Robertamodel = 'distilroberta-base'
Roberta_tokenizer = RobertaTokenizer.from_pretrained(pre_trained_Robertamodel, do_lower_case=True)


class TweetsDataset(Dataset):

    def __init__(self, tweets, targets, tokenizer, max_length):
        self.tweets = tweets
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        targets = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            tweet,
            max_length=Max_length,
            add_special_tokens=True,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        return {
            'tweets_text': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(targets, dtype=torch.long)
        }


class Roberta_SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(Roberta_SentimentClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pre_trained_Robertamodel)
        self.drop = nn.Dropout(p=0.35)
        self.hidden = nn.Linear(self.roberta.config.hidden_size, 128)
        self.out = nn.Linear(128, n_classes)
        # self.softmax=nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = pooled_output['pooler_output']
        output = self.drop(pooled_output)
        output = self.hidden(pooled_output)
        output = self.out(output)

        return output


def preprocessing(df):
    # drop unneeded columns
    df = df.drop(['UserName', 'ScreenName'], axis=1)

    # Replace @,# etc by ' '
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: re.sub(r'[@#]', ' ', x))
    # Excluding html tags
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: re.sub(r'<[^<>]+>]', ' ', x))
    # Replace https links by ' '
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: re.sub(r'http://\S+|https://\S+', ' ', x))
    return df


def data_loader(df, tokenizer, max_length, batch):
    ds = TweetsDataset(
        tweets=df.OriginalTweet.to_numpy(),
        targets=df.Sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_length=Max_length
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,

    )


def train(model, data_loader, loss_fn, optimizer, device, scheduler, n_observations):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"]
        attention_mask = d["attention_mask"]
        targets = d["targets"]
        print(f"targets: {targets}")
        # Feed data to BERT model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to avoid exploding gradient problem
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_observations, np.mean(losses)


def eval_model(model, data_loader, device, loss_fn, n_observations):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            targets = d["targets"]
            # Feed data to  model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_observations, np.mean(losses)
