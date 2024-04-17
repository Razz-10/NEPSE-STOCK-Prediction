from flask import Flask, request, jsonify,render_template
from flask import Flask
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
import json

app = Flask(__name__)

@app.route("/",methods=["GET"])
def home():
    URL = "https://merolagani.com/LatestMarket.aspx"

    # Make GET request to fetch the raw HTML content
    conn = requests.get(URL)
    soup = BeautifulSoup(conn.text, 'html.parser')

    table = soup.find('table', class_='table table-hover live-trading sortable')

    headers = [i.text for i in table.find_all('th')]

    data = []
    for row in table.find_all('tr', {"class": ["decrease-row", "increase-row", "nochange-row"]}):
        row_data = {}
        cells = row.find_all('td')
        row_data['row_class'] = row['class'][0]  # Extracting row class for styling
        for index, cell in enumerate(cells):
            row_data[headers[index]] = cell.text
        data.append(row_data)

    table_data ={"headers": headers,"data": data}
    return render_template('home.html', table_data=table_data)


@app.route("/news")
def news():
    
    response = requests.get("https://merolagani.com/NewsList.aspx")
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
    
        news_elements = soup.find_all("div", class_="media-news media-news-md clearfix")
        
        # Extract news titles, links, and images
        news_list = []
        for news in news_elements:
            title = news.find("h4", class_="media-title").text.strip()
            link = news.find("a")["href"]
            image = news.find("img")
            if image:
                image_url = image["src"]
            else:
                # If image is not available, provide a default image URL
                image_url = "https://via.placeholder.com/150"
            news_list.append({"title": title, "link": link, "image_url": image_url})
        
        # Pass the news list to the template for rendering
        context = {"news_list": news_list}
        return render_template('news.html',context=context)
        # return render(request, "Stock/news.html", context)
    return "Failed to fetch news"









@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method == "POST":
        stock_symbol = request.form["stock_symbol"].upper()
        try:
            model = tf.keras.models.load_model(f"{stock_symbol}_model.h5")
        except Exception as e:
            return f"Error loading model: {e}"

        file_path = f"{stock_symbol}.csv"
        try:
            df = pd.read_csv(file_path)
            df=df.set_index('Date')
        except Exception as e:
            return f"Error reading CSV file: {e}"

        scaler, x_train, y_train, x_test, y_test,train_dates,test_dates,dataset,training_data_len = preprocess(df)

        #predicting train data
        train_predictions=model.predict(x_train)
        train_predictions=scaler.inverse_transform(train_predictions)

        y_train_original=y_train.reshape(-1,1)
        y_train_original=scaler.inverse_transform(y_train_original)

        #predicting test data
        test_predictions=model.predict(x_test)
        test_predictions=scaler.inverse_transform(test_predictions)

        y_test_original = y_test.reshape(-1, 1) 


      

        # Calculate MAPE for testing data
        def calculate_mape(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        mape = calculate_mape(y_test, test_predictions)
        print("MAPE for Testing Data:", mape)
        
        window_size =80

        # Select the last 80 days of data
        last_80_days = dataset[-window_size:]

        # Scale the data
        last_80_days_scaled = scaler.transform(last_80_days)

        #reshape the data for model prediction
        # Reshape the data for model prediction
        X_test = []
        X_test.append(last_80_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Predict the next 10 days' stock prices
        predicted_prices = []
        for i in range(10):
            predicted_price = model.predict(X_test)
            predicted_prices.append(predicted_price[0][0])
             # Update X_test for the next prediction
            X_test = np.append(X_test[:, 1:, :], np.expand_dims(predicted_price, axis=1), axis=1)

        # Invert the scaling
        predicted_prices = np.array(predicted_prices).reshape(-1, 1)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        #latest dates
        latest_dates = df.index[-window_size:]# Dates for the latest data

        latest_data=df['Open'].tail(window_size)


        x_labels = [f"Day {i+1}" for i in range(len(predicted_prices))]

        
        predicted_pricess=predicted_prices.flatten().tolist()
        predicted_pricess = [round(price, 2) for price in predicted_pricess]
       
        latest_data= latest_data.tolist()
        latest_dates = df.index[-window_size:].tolist()
        print(latest_dates)

      
        train_dates=df.index[10:training_data_len].tolist()
        test_datess=df.index[training_data_len:].tolist()
        y_test_original = y_test_original.flatten().tolist()
        y_train_original=y_train_original.flatten().tolist()
        train_predictions=train_predictions.flatten().tolist()
       
        
        predictions=test_predictions.flatten().tolist()
        df=df.reset_index()
        test_dates_str = [str(date) for date in test_dates]
       
        new_dates = pd.date_range(start=test_dates_str[-1], periods=11)[1:].strftime('%Y-%m-%d').tolist()
        latest_dates_set=set(latest_dates)
        new_dates_set =set(new_dates)
        merged_dates_set = latest_dates_set.union(new_dates_set)
        merged_dates = list(merged_dates_set)
        merged_dates.sort()
        
        

        data = {
            'train_dates':train_dates,
            'predictions':predictions,
            'train_predictions':train_predictions,
            'y_test_original':y_test_original,
            'y_train_original':y_train_original,
            'test_dates':test_datess,
            'latest_date':latest_dates,
            'latest_data':latest_data,
            'x_labels':x_labels,
            'new_dates':new_dates,
            'predicted_pricess':predicted_pricess,
            'merged_dates':merged_dates,
            
        }
        
    
         
 
        data_json = json.dumps(data) 
        return render_template('predict.html', datas=data_json,predicted_prices=predicted_pricess,symbol=stock_symbol)


        

    else:


        return render_template('predict.html')
            
    
def preprocess(df):
    data_open=df.filter(['Open'])
    dataset=data_open.values
    training_data_len=math.ceil(len(dataset)*.8)
    train_dates=df.index[10:training_data_len]
    test_dates=df.index[training_data_len:]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
        
    #training data set
    train_data=scaled_data[0:training_data_len,:]
    #split data
    x_train=[]
    y_train=[]
    initial_window_size=10
    for i in range(initial_window_size, len(train_data)):
        x_train.append(train_data[i-initial_window_size:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    #convert the x_train and y_train numpy arrays
    x_train,y_train=np.array(x_train),np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], initial_window_size, 1))



        ##create the testing dataset
    initial_window_size = 10
    test_data = scaled_data[training_data_len - initial_window_size:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(initial_window_size, len(test_data)):
        x_test.append(test_data[i - initial_window_size:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], initial_window_size, 1))
        

    x_test=np.array(x_test)
    x_test=np.reshape(x_test, (x_test.shape[0],initial_window_size,1))



    return scaler ,x_train,y_train,x_test,y_test,train_dates,test_dates,dataset,training_data_len

if __name__=="__main__":
    app.run(debug=True)