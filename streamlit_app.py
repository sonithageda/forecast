import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st

#streamlit app
st.title("Forecasting Model")

#Upload Dataset
uploaded_file=st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    df['Date']=pd.to_datetime(df['Date'])
    #set 'date' column as index
    df.set_index('Date',inplace=True)
    st.write(df.head())

    #ask for user input
    target_column = st.selectbox("Select target column for forecasting:",df.columns)
    dependent_columns= st.multiselect("Select dependent columns:",df.columns)

    #Button to start model training and forecasting
    if st.button("Start Forecasting"):
        #Normalize the Data for selected columns
        scaler=MinMaxScaler()
        normalised_data=scaler.fit_transform(df[[target_column]+dependent_columns])

        #Convert the normalized columns into images
        def create_images(data, window_size):
            images =[]
            for i in range(len(data)-window_size):
                image=data[i:i+window_size]
                images.append(image)
            return np.array(images)
        
        window_size=30

        #Create images for each column
        images =[]
        for i in range(normalised_data.shape[1]):
            images.append(create_images(normalised_data[:,1],window_size))

        #combine all image inputs into a single input array
        all_images=np.stack(images,axis=-1)

        #split data into training and testing sets
        split_ratio=0.8
        split_index=int(len(all_images)*split_ratio)
        X_train=all_images[:split_index]
        y_train=normalised_data[window_size:split_index+window_size,0]
        X_test=all_images[split_index:]
        y_test=normalised_data[split_index+window_size:,0]

        #Define LSTM model with two LSTM Layers
        model=Sequential([
            LSTM(units=100,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])),
            LSTM(units=100),
            Dense(units=1)
        ])

        model.compile(optimizer='adam',loss='mean_squared_error')

        #Train the model with more epochs
        model.fit(X_train,y_train,epochs=50,batch_size=32)
        loss=model.evaluate(X_test,y_test)
        st.write(f"Test Loss:{loss}")

        #Test the model
        y_train_pred=model.predict(X_train)
        y_train_pred_inverse=scaler.inverse_transform(np.hstack((np.zeros((len(y_train_pred),3)),y_train_pred.reshape(-1,1))))
        y_test_pred=model.predict(X_test)
        y_test_pred_inverse=scaler.inverse_transform(np.hstack((np.zeros((len(y_test_pred),3)),y_test_pred.reshape(-1,1))))
        y_train_inverse=scaler.inverse_transform(np.hstack((np.zeros((len(y_train),3)),y_train.reshape(-1,1))))
        y_test_inverse=scaler.inverse_transform(np.hstack((np.zeros((len(y_test),3)),y_test.reshape(-1,1))))

        #calculate mean absolute percentage error(MAPE)
        train_mape=np.mean(np.abs((y_train_inverse-y_train_pred_inverse)/y_train_inverse))*100
        test_mape=np.mean(np.abs((y_test_inverse-y_test_pred_inverse)/y_test_inverse))*100

        #display test results
        st.write(f"Train Mean Absolute Percentage Error(MAPE):{train_mape:.2f}%")
        st.write(f"Test Mean Absolute Percentage Error(MAPE):{test_mape:.2f}%")

        #number of future time stamps to predict
        n_future=st.number_input("Enter number of day to predict into the future",min_value=1,step=1)
        future_predictions=[]
        last_window=X_test[-1]

        #iterate to predict future stamps
        for i in range(n_future):
            next_prediction_scaled=model.predict(last_window.reshape((1,window_size,4)))
            future_predictions.append(next_prediction_scaled[0][0])
            last_window=np.roll(last_window,-1,axis=0)
            last_window[-1][-1]=next_prediction_scaled[0][0]
        
        future_predictions=np.array(future_predictions).reshape(-1,1)
        future=scaler.inverse_transform(np.hstack((np.zeros((len(future_predictions),3)),future_predictions.reshape(-1,1))))

        #get the last historical data
        last_data=df.index[-1]

        #convert last historical date to pandas timestamp object
        last_data_timestamp=pd.Timestamp(last_data)

        #calculate  future date starting from the last historical data
        future_dates= [last_data_timestamp + pd.Timedelta(days=i+1) for i in range(n_future)]

        #display predicted values of the targeted column in a table
        st.write("Future Predictions:")
        future_df=pd.DataFrame(future_predictions[:,-1],columns=[target_column],index=future_dates)
        st.write(future_df)

        #plot historical VS predicted data
        fig, ax= plt.subplots()
        df[target_column].plot(ax=ax,label='Historical Data')
        future_df[target_column].plot(ax=ax,label='Predicted Data')
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.title('Historical vs Predicted Data')
        plt.legend()
        st.pyplot(fig)
