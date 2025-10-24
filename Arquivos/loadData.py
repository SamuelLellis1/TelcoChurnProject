import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_data("../Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(df.head())
