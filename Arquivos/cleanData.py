import pandas as pd

def clean_data(df: pd.DataFrame):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors= "coerce")
    df.dropna(subset=["TotalCharges"], inplace= True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1:"Yes", 0:"No"})
    return df

if __name__ == "__main__":
    from loadData import load_data
    df = load_data("../Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_clean = clean_data(df)
    print(df_clean.info())