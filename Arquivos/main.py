import pandas as pd
from model import train_model

def main():
   df = pd.read_csv("../Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
   model = train_model(df)


if __name__  =="__main__" :
    import pandas as pd
    df = pd.read_csv("../Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    train_model(df)