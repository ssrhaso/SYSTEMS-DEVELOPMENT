#Test File for Data Exploration
import pandas as pd
import matplotlib.pyplot as plt

def read_file(file_path):
    data = pd.read_csv(file_path)
    return data

def data_clean(data):
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'], format= '%d/%m/%Y')
    return data
def plot_data(data):
    headers = data.columns.tolist()
    data.plot(x=headers[0], kind='line')
    plt.ylabel('Number Sold')
    plt.title('Sales Data Over Time')
    plt.show()
    




drinks_data = pd.read_csv('data/raw/Pink CoffeeSales March - Oct 2025.csv')
drinks_clean_data = data_clean(drinks_data)
plot_data(drinks_clean_data)