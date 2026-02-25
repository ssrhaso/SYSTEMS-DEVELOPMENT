import pandas as pd
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import os

file_paths = ['./data/raw/Pink_CroissantSales_March-Oct_2025.csv', './data/raw/Pink_CoffeeSales_March - Oct 2025.csv']
def read_file(file_path):
    type_item = 0
    
    # Coffee file has a 2-row header structure - combine headers from both rows
    if "Coffee" in file_path:
        type_item = 2
        # Read first two rows to extract header information
        header_row1 = pd.read_csv(file_path, nrows=0).columns.tolist()
        header_row2 = pd.read_csv(file_path, skiprows=1, nrows=0).columns.tolist()
        
        # Combine headers intelligently
        combined_headers = []
        for i, (col1, col2) in enumerate(zip(header_row1, header_row2)):
            # First column: use row 1 (Date)
            # Other columns: prefer row 2 if it has meaningful data
            if i == 0 and col1 and not col1.startswith('Unnamed'):
                combined_headers.append(col1)
            elif col2 and not col2.startswith('Unnamed') and col2.strip():
                combined_headers.append(col2)
            elif col1 and not col1.startswith('Unnamed'):
                combined_headers.append(col1)
            else:
                combined_headers.append(f'Column_{i}')  # fallback
        
        # Read data skipping first row, then set proper column names
        data = pd.read_csv(file_path, skiprows=1)
        data.columns = combined_headers
        
    elif "Croissant" in file_path:
        type_item = 1
        data = pd.read_csv(file_path)
        if "Number Sold" in data.columns:
            data = data.rename(columns={"Number Sold": "Croissants"})
    else:
        data = pd.read_csv(file_path)
    
    return data, type_item

def data_clean(data):
    data.dropna(inplace=False)
    data.drop_duplicates(inplace=False)
    data['Date'] = pd.to_datetime(data['Date'], format= '%d/%m/%Y')
    return data

def plot_data(data, output_path='plot.png', drink_name=None):
    headers = data.columns.tolist()
    date_column = headers[0]
    plt.figure(figsize=(14, 6))
    if drink_name:
        if drink_name not in data.columns:
            print(f"Warning: '{drink_name}' not found in data. Available columns: {headers[1:]}")
            plt.close()
            return
        data.plot(x=date_column, y=drink_name, kind='line', marker='o')
        plt.title(f'{drink_name} Sales Over Time')
    else:
        data.plot(x=date_column, kind='line')
        plt.title('Sales Data Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number Sold')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory
    print(f"Plot saved to: {output_path}")

 #testng the functions
if __name__ == "__main__":    
    for i, file in enumerate(file_paths):
        data, type_item = read_file(file)
        clean_data = data_clean(data)
        # Generate output filename based on input file
        filename = os.path.basename(file).replace('.csv', '.png')
        output_path = f'./data/processed/{filename}'
        plot_data(clean_data, output_path)
        if type_item == 2:  # If it's the Coffee file, also plot a single drink
            plot_data(clean_data, f'./data/processed/Cappuccino_{filename}', drink_name='Cappuccino')
            plot_data(clean_data, f'./data/processed/Americano_{filename}', drink_name='Americano')