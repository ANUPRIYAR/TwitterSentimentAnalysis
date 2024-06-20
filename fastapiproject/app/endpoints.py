import asyncio
from collections import defaultdict
from typing import Optional

import pandas as pd
import torch
from fastapi import APIRouter, Depends, Form
from fastapi import HTTPException
from fastapi.responses import Response
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from transformers import AdamW, AdamWeightDecay, get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup

from app.llm_service import get_answer
from app.roberta_utils import constants
from app.roberta_utils.helper_functions import preprocessing, pre_trained_Robertamodel, data_loader, \
    Roberta_SentimentClassifier, Max_length, batch_size, train, eval_model, Roberta_tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

router = APIRouter()
train_path = r'.....\Corona_NLP_train.csv'
test_path = r'.....\Corona_NLP_test.csv'


saved_model_path = r'......\sentiment_analysis_lstm.keras'

@router.post("/train")
async def train_api(epochs: Optional[int] = Form(1)):
    print(epochs)
    train_df = pd.read_csv(train_path, encoding='latin1')
    test_df = pd.read_csv(test_path, encoding='latin1')

    train_df = preprocessing(train_df)
    test_df = preprocessing(test_df)

    le = LabelEncoder()
    train_df['Sentiment'] = le.fit_transform(train_df['Sentiment'])
    test_df['Sentiment'] = le.transform(test_df['Sentiment'])
    map = dict(zip(le.classes_, le.transform(le.classes_)))
    print(map)
    print(f"classes : {le.classes_}")
    print(f"test_df: {test_df}")

    Roberta_tokenizer = RobertaTokenizer.from_pretrained(pre_trained_Robertamodel, do_lower_case=True)

    df_train, df_val = train_test_split(train_df, test_size=0.25, random_state=123, stratify=train_df.Sentiment)

    train_DataLoader = data_loader(df_train, Roberta_tokenizer, constants.Max_length, constants.batch_size)
    test_DataLoader = data_loader(test_df, Roberta_tokenizer, constants.Max_length, constants.batch_size)
    print(f"test_DataLoader : {test_DataLoader}")
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
        if val_acc > best_accuracy:
            torch.save(Roberta_model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    test_acc, _ = eval_model(Roberta_model, test_DataLoader, constants.device, loss_fn, len(test_df))
    response = {
        "test_acc" : test_acc,
        "history" : history
    }

    return response


@router.post("/predict")
async def predict(query: str = Form(..., description="The user's query sent to the LLM.")):
    print(query)
    query_df = pd.DataFrame([query], columns=['OriginalTweet'])
    import tensorflow as tf
    saved_model = tf.keras.models.load_model(saved_model_path)
    print(saved_model.summary())
    # Preprocess query
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(query_df['OriginalTweet'])
    sequences = tokenizer.texts_to_sequences(query_df['OriginalTweet'])
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')
    print(padded_sequences)
    prediction = saved_model.predict(padded_sequences)
    print(f"Prediction: {prediction}")
    prediction = prediction.argmax(axis=1)
    sentiment_mapping = {
        'Extremely Negative': 0,
        'Extremely Positive': 1,
        'Negative': 2,
        'Neutral': 3,
        'Positive': 4
    }
    sentiment = list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(prediction)]
    return sentiment


def preprocess_query(query, tokenizer, max_length):
    # Tokenize and encode the query
    encoding = tokenizer.encode_plus(
        query,
        max_length=max_length,
        add_special_tokens=True,
        pad_to_max_length=True,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt'
    )

    # Extract the input IDs and attention mask and convert them to tensors
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    return input_ids, attention_mask

@router.post("/predict_roberta")
async def get_prediction(query: Optional[str] = Form(None, description="The user's query sent to the LLM."), test_file_path: Optional[str] = Form(None)):
    if query:
        print(query)
        max_length = 120
        input_ids, attention_mask = preprocess_query(query, Roberta_tokenizer, max_length)
        model_path = 'app/best_model_state.bin'
        model = Roberta_SentimentClassifier(n_classes=5)
        model.load_state_dict(torch.load(model_path))
        pred = model(input_ids, attention_mask)
        probabilities = torch.nn.functional.softmax(pred, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        print(predicted_class)
        sentiment_mapping = {
            'Extremely Negative': 0,
            'Extremely Positive': 1,
            'Negative': 2,
            'Neutral': 3,
            'Positive': 4
        }
        sentiment = list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(predicted_class)]
        response = sentiment
    elif test_file_path:
        max_length = 120
        test_df = pd.read_csv(test_file_path, encoding='latin1')
        test_df = preprocessing(test_df)
        print(f"test_df :{test_df}")
        predictions = []
        for query in test_df['OriginalTweet']:
            print(query)
            input_ids, attention_mask = preprocess_query(query, Roberta_tokenizer, max_length)
            model_path = 'app/best_model_state.bin'
            model = Roberta_SentimentClassifier(n_classes=5)
            model.load_state_dict(torch.load(model_path))
            pred = model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(pred, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            print(predicted_class)
            sentiment_mapping = {
                'Extremely Negative': 0,
                'Extremely Positive': 1,
                'Negative': 2,
                'Neutral': 3,
                'Positive': 4
            }
            sentiment = list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(predicted_class)]
            predictions.append(sentiment)
        test_df['predictions'] = predictions
        test_df_path = 'app/test_df_predictions.csv'
        test_df.to_csv(test_df_path, index=False)

        response = test_df_path

    return response



@router.post("/queryllm")
async def queryllm(query: str = Form(...)):
    llm_response = get_answer(query)
    response = {"response": llm_response}
    return response



