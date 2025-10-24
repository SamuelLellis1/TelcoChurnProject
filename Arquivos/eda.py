import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def summary (df: pd.DataFrame):
    print("Valores nulos:", df.isna().sum())
    print("Shape:", df.shape)
    print(df.describe())

def churn_distribuiton (df: pd.DataFrame):
    sns.countplot(data = df, x= "Churn")
    plt.title("Distribuição de Churn")
    plt.show()

def cor_heatmap (df: pd.DataFrame):
    corr = df.corr(numeric_only= True)
    sns.heatmap(corr, annot = True, cmap="Blues")
    plt.title("Matriz de Correlação")
    plt.show()
