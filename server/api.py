from flask import Flask, request, jsonify, abort
import os
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

# Load column nsames of classifier's input shape
colname_path = os.environ['COLNAME_PATH']
colnames = joblib.load(colname_path)

# Load Imputer for missing values
imputer_path = os.environ['IMPUTER_PATH']
imputer = joblib.load(imputer_path)

# Load classifier
model_path = os.environ['MODEL_PATH']
clf = joblib.load(model_path)


app = Flask(__name__)

@app.route('/v1/categorize', methods = ['POST'])
def categorize():
    try:
        json_dict = request.get_json(force=True)
    except:
        abort(400, "Invalid JSON string")

    try:
        products = json_dict['products']
    except:
        abort(400,"JSON must contain 'products'")

    try:
        first_element = products[0]
    except:
        abort(400,"JSON array 'products' must be non empty")

    try:
        for element in products:
            title = element["title"]
    except:
        abort(400,"Any element of the 'products' array must contain a 'title' value")

    try:
        data = pd.DataFrame(products)
    except:
        abort(400,"Can't convert 'products' json array into a dataframe")

    # Transform title and concatenated_tags in word_counts
    for col in ['title','concatenated_tags']:
        if col in data.columns:
            try:
                word_counts = generate_word_count_frame(data[col])
                data = pd.concat([data,word_counts],axis=1)
            except:
                None
            data = data.drop(columns=[col])


    # One hot encode the seller_id
    if 'seller_id' in data.columns:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc_df = pd.DataFrame(enc.fit_transform(data[['seller_id']]).toarray())
        enc_df.columns = ['seller_id_'+str(col) for col in enc_df]

        # Replace seller_id column with one-hot-encoded data frame
        data = data.drop(columns=['seller_id'])
        data = pd.concat([data,enc_df],axis=1)

    # Convert the creatin date in a Unix Timestamp
    if 'creation_date' in data.columns:
        data['creation_date'] = pd.to_datetime(data['creation_date'], format='%Y-%m-%d %H:%M:%S').astype(int) / 10**9

    # Fill missing colums to match the classifiers input shape
    input_data = pd.DataFrame(index=data.index)

    for col in colnames:
        if col in data.columns:
            col_data = list(data[col])
        else:
            col_data = [None for val in range(len(data))]
        additional_column = pd.DataFrame({col:col_data})
        input_data = pd.concat([input_data,additional_column],axis=1)

    # Use imputer in order to remove NULL values
    input_data = imputer.transform(input_data)

    # Run classifier
    output_data = clf.predict(input_data)

    return  jsonify({"categories": list(output_data)})


def generate_word_count_frame(column):
    # Limit the verctorizer to the 1.000 most popular words (for memory & speed reasons)
    cv = CountVectorizer(max_features=1000)

    column = column.fillna('')
    tf = cv.fit_transform(column)

    word_count_frame = pd.DataFrame(tf.toarray(), columns=cv.get_feature_names())
    word_count_frame.columns = [column.name+'_'+str(col) for col in word_count_frame.columns]

    return word_count_frame
