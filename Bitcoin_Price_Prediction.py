import numpy as np
import joblib
def Bitcoin_Closing_Price():
    try:
        ss=joblib.load(r"C:\Users\ACER\Documents\Downloads\scaler_bitcoin.pkl")
        gru=joblib.load(r"C:\Users\ACER\Documents\Downloads\gru_model.pkl")
        ann=joblib.load(r"C:\Users\ACER\Documents\Downloads\ann_model.pkl")
        xgboost=joblib.load(r"C:\Users\ACER\Documents\Downloads\xgboost_model.pkl")
        Day=int(input("Enter the Day of the Date: "))
        Month=int(input("Enter the Month of Date: "))
        Year=int(input("Enter the Year of the Date: "))
        Open=float(input("Enter the Open price of Bitcoin: "))
        High=float(input("Enter the Highest price of Bitcoin for the Day: "))
        Low=float(input("Enter the Lowest price of the Bitcoin for the Day: "))
        Volume=float(input("Enter the Volume of the Bitcoin for the Day: "))
        features=np.array([Year,Month,Day,High,Low,Open,Volume]).reshape(1,-1)
        features_scaled=ss.transform(features)
        xgboost_pred=xgboost.predict(features_scaled)
        ann_pred=ann.predict(features_scaled)
        #gru_pred=gru.predict(features_scaled)



        print(f"The XGBOOST Model predict {xgboost_pred} as the closing price of Bitcoin for Date {Day}-{Month}-{Year}")
        print(f"The ANN Model predict {ann_pred} as the closing price of Bitcoin for Date {Day}-{Month}-{Year}")
    except ValueError as e1:
        print("Please provide valid values(numeric) for the inputs.",e1)
    except Exception as e2:
        print("Something wents wrong in the execution of the code", e2)
    finally:
        print("Thanks for using this Bitcoin Closing Price Prediction Model")
if __name__=="__main__":
    Bitcoin_Closing_Price()
        
    
    
