from collections import defaultdict

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from transformers import AdamW, AdamWeightDecay, get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup

from app.roberta_utils import constants
from app.roberta_utils.helper_functions import preprocessing, pre_trained_Robertamodel, data_loader, \
    Roberta_SentimentClassifier, Max_length, batch_size, train, eval_model


train_path = r'.......\Corona_NLP_train.csv'
test_path = r'.........\Corona_NLP_test.csv'


def train_model():
    train_df = pd.read_csv(train_path, encoding='latin1')
    test_df = pd.read_csv(test_path, encoding='latin1')

    train_df = preprocessing(train_df)
    test_df = preprocessing(test_df)

    le = LabelEncoder()
    train_df['Sentiment'] = le.fit_transform(train_df['Sentiment'])
    test_df['Sentiment'] = le.transform(test_df['Sentiment'])

    Roberta_tokenizer = RobertaTokenizer.from_pretrained(pre_trained_Robertamodel, do_lower_case=True)

    df_train, df_val = train_test_split(train_df, test_size=0.25, random_state=123, stratify=train_df.Sentiment)

    train_DataLoader = data_loader(df_train, Roberta_tokenizer, constants.Max_length, constants.batch_size)
    test_DataLoader = data_loader(test_df, Roberta_tokenizer, constants.Max_length, constants.batch_size)
    valid_DataLoader = data_loader(df_val, Roberta_tokenizer, constants.Max_length, constants.batch_size)

    Roberta_model = Roberta_SentimentClassifier(len(constants.class_names))

    optimizer = AdamW(Roberta_model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_DataLoader) * constants.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()

    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(constants.epochs):
        print(f'Epoch {epoch + 1}/{constants.epochs}')
        print('-' * 10)
        train_acc, train_loss = train(
            Roberta_model,
            train_DataLoader,
            loss_fn,
            optimizer,
            constants.device,
            scheduler,
            len(df_train)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = eval_model(
            Roberta_model,
            valid_DataLoader,
            constants.device,
            loss_fn,
            len(df_val)
        )
        print(f'Validation  loss {val_loss} accuracy {val_acc}')
        print()
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        torch.save(Roberta_model.state_dict(), '../app/model_state.bin')
        if val_acc > best_accuracy:
            torch.save(Roberta_model.state_dict(), '../app/best_model_state.bin')
            print("Model saved")    
            print(f"Best accuracy is: {val_acc}")
            best_accuracy = val_acc

    test_acc, _ = eval_model(Roberta_model, test_DataLoader, constants.device, loss_fn, len(test_df))
    response = {
        "test_acc": test_acc,
        "history": history
    }

    return response


if __name__ == '__main__':
    response = train_model()
    print(response)

