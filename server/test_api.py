import pytest
import requests
import pandas as pd
import os
import json
import numpy as np

#API_ENDPOINT = os.environ['API_ENDPOINT']
API_ENDPOINT = "http://127.0.0.1:5000/v1/categorize"


def test_error_get_request():
    "GET request to api endpoint returns a 405 error"
    resp = requests.get(API_ENDPOINT)
    assert resp.status_code == 405

def test_error_invalid_json():
    "An invalid JSON String should return HTTP error 400"
    payload = '{ "dds" "some invalid json"}'
    resp = requests.post(API_ENDPOINT, data=payload)
    assert resp.status_code == 400
    assert resp.text.find("Invalid JSON string") > 0

def test_products_missing():
    "An JSON payload without a 'products' variable should return a HTTP 400 error"
    payload = '{ "productss" : [] }'
    resp = requests.post(API_ENDPOINT, data=payload)
    assert resp.status_code == 400
    assert resp.text.find("JSON must contain 'products'") > 0

def test_error_empty_array():
    "Empty but valid json request should return an error message"
    payload = '{ "products": [] }'
    resp = requests.post(API_ENDPOINT, data=payload)
    assert resp.status_code == 400
    assert resp.text.find("JSON array 'products' must be non empty") > 0

def test_error_title_mandatory_single_value_array():
    "If one of the elements from the product array does not contain title, it should return an error message"
    payload = '{ "products": [ {"bla":"blub"} ] }'
    resp = requests.post(API_ENDPOINT, data=payload)
    assert resp.status_code == 400
    assert resp.text.find("Any element of the 'products' array must contain a 'title' value") > 0

def test_error_title_mandatory_multiple_value_array():
    "If one of the elements from the product array does not contain title, it should return an error message"
    payload = '{ "products": [ {"title":"some title"},{"bla":"blub"} ] }'
    resp = requests.post(API_ENDPOINT, data=payload)
    assert resp.status_code == 400
    assert resp.text.find("Any element of the 'products' array must contain a 'title' value") > 0

def test_title_empty():
    "Test if conversion of an empty title works properly"
    payload = u'{ "products": [ {"title":""},{"title":""} ] }'
    resp = requests.post(API_ENDPOINT, data=payload)
    assert resp.status_code == 200
    assert len(resp.json()["categories"]) == 2

def test_title_word_counts_conversion():
    "Test if conversion of title into word_counts works properly"
    payload = u'{ "products": [ {"title":"Lembrancinha"},{"title":"Carinho de Bebe"} ] }'
    resp = requests.post(API_ENDPOINT, data=payload)
    assert resp.status_code == 200
    assert len(resp.json()["categories"]) == 2
    assert resp.json()["categories"][0] == "Lembrancinhas"
    assert resp.json()["categories"][1] == u"Bebê"

def test_concatenated_tags_word_counts_conversion():
    "Test if conversion of concatenated_tags into word_counts works properly"
    payload = u'{ "products": [ {"title":"", "concatenated_tags":"lembrancinhas"},{"title":"", "concatenated_tags":"bebe"} ] }'
    resp = requests.post(API_ENDPOINT, data=payload)
    assert resp.status_code == 200
    assert len(resp.json()["categories"]) == 2
    assert resp.json()["categories"][0] == "Lembrancinhas"
    assert resp.json()["categories"][1] == u"Bebê"

def test_one_hot_encoding_seller_id():
    "Test one hot encoding of seller_id. \
     This is a weak tests, since it will not check for an actual classification, \
     but only try to pass through the one hot encoding section of the code."
    payload = u'{ "products": [ {"title":"", "seller_id": 203002} ] }'
    resp = requests.post(API_ENDPOINT, data=payload)
    assert resp.status_code == 200
    assert len(resp.json()["categories"]) == 1

def test_creation_date_conversion():
    "Test the conversion of the creation date. \
     This is a weak tests, since it will not check for an actual classification, \
     but only try to pass through the one hot encoding section of the code."
    payload = u'{ "products": [ {"title":"", "creation_date": "2015-11-14 19:42:12"} ] }'
    resp = requests.post(API_ENDPOINT, data=payload)
    assert resp.status_code == 200
    assert len(resp.json()["categories"]) == 1


def test_general_performance():
    "Test if the API has an average precision of more than 80vi %."
    test_products_path = os.environ["TEST_PRODUCTS_PATH"]

    df = pd.read_csv(test_products_path)
    df_input = df.drop(columns=['category'])
    df_output = df['category']

    payload = '{ "products" : %s }'%(df_input.to_json(orient='records'))
    print(payload,flush=True)

    resp = requests.post(API_ENDPOINT, data=payload)
    categories = resp.json()["categories"]
    assert resp.status_code == 200
    assert len(categories) == len(df_output)
    assert np.sum(np.where(categories == df_output,1,0))/len(df_output) > 0.8
