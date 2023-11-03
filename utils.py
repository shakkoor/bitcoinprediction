import requests
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,LSTM
from keras import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# model = load_model('humanAI.h5') 
def preprocessdata(crypto,curr,days):
    crypto=crypto.lower()
    curr=curr.lower()
    url = f"https://tokeninsight-crypto-api1.p.rapidapi.com/api/v1/history/coins/{crypto}"

    querystring = {"interval":"day","length":days,"vs_currency":curr}

    headers = {
        "TI_API_KEY": "a6853f07b5674e5ab026c71b60eb9851",
        "X-RapidAPI-Key": "83c90e634bmsha596c0631310c5cp156c3djsnaf348aaffe81",
        "X-RapidAPI-Host": "tokeninsight-crypto-api1.p.rapidapi.com"
    }



    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()
        data = pd.DataFrame(data["data"]["market_chart"])
        data = data.dropna()
        x = data.drop('price',axis=1)
        print(x.head())
        sc = StandardScaler()
        sc.fit(x)

        # Transform the data using the scaler
        x = sc.transform(x)
        print(x)
        y = data['price']
        X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
        X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        model_lstm = Sequential()
#         model_lstm.add(GRU(10, input_shape=(1, X_train.shape[1]), activation='linear', kernel_initializer='lecun_uniform', return_sequences=False))
        model_lstm.add(LSTM(50, input_shape=(1, X_train.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=True))
        model_lstm.add(Dense(100,activation="relu"))
        model_lstm.add(Dense(300,activation="relu"))
        model_lstm.add(Dense(400,activation="relu"))
        model_lstm.add(Dense(100))
        model_lstm.add(Dense(1))
        model_lstm.summary()
        model_lstm.compile(loss=tf.keras.metrics.mean_squared_error,
              metrics=[tf.keras.metrics.RootMeanSquaredError(name='mse')], optimizer=tf.keras.optimizers.RMSprop())
        early_stop = EarlyStopping(monitor='loss', patience=30, verbose=1)
        history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])

        y_p = model_lstm.predict(X_tst_t)
        print("evaluation")
        model_lstm.evaluate(X_tst_t,y_test)
        print(y_p[0][0]) 
        return y_p[0][0]
    except requests.exceptions.RequestException as e:
        print(e)
        return f'An error occurred: {str(e)}'


