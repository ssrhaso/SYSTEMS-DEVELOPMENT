# Test File for Data Exploration
import pandas as pd # Import the pandas library for data manipulation and analysis.
import matplotlib.pyplot as plt # Import the matplotlib.pyplot module for plotting/visualization.

# --- Function Definitions ---

def read_file(file_path):
    # Function to read a CSV file into a pandas DataFrame.
    data = pd.read_csv(file_path)
    return data

def data_clean(data):
    # Function to clean the input pandas DataFrame.
    data.dropna(inplace=False) # Remove rows with any missing (NaN) values, modifying the DataFrame in place.
    data.drop_duplicates(inplace=False) # Remove duplicate rows, modifying the DataFrame in place.
    
    
    data['Date'] = pd.to_datetime(data['Date'], format= '%d/%m/%Y')# Convert the 'Date' column to datetime objects using the specified format.
    return data
<<<<<<< HEAD

=======
    
>>>>>>> bf580eb310106130649c4db6f391078bf42f8950
def plot_data(data):
    # Function to create and display a line plot of the sales data.
    headers = data.columns.tolist() # Get a list of all column names (headers).
    # Create a line plot. It uses the first column (headers[0], expected to be 'Date') as the x-axis 
    # and all other columns (sales figures) as y-series by default.
    data.plot(x=headers[0], kind='line') 
    # Set the label for the y-axis.
    plt.ylabel('Number Sold')
    # Set the title of the plot.
    plt.title('Sales Data Over Time')
    # Display the plot window.
    plt.show()
    
# --- Main Execution Block (for Pink Coffee Sales) ---

# Read the coffee sales data into a DataFrame.
drinks_data = pd.read_csv('data/raw/pink_coffee_sales.csv')
# Clean the coffee sales data (drops NA/duplicates, converts 'Date').
drinks_clean_data = data_clean(drinks_data)
# Plot the cleaned coffee sales data.
plot_data(drinks_clean_data)
# --- Main Execution Block (for Pink Croissant Sales) ---
# Read the food sales data into a DataFrame.
food_data = pd.read_csv('data/raw/pink_croissant_sales.csv')
<<<<<<< HEAD
food_clean_data = data_clean(food_data) 
plot_data(food_clean_data)

=======
# Clean the food sales data (drops NA/duplicates, converts 'Date').
food_clean_data = data_clean(food_data)
# Plot the cleaned food sales data.
plot_data(food_clean_data)
>>>>>>> bf580eb310106130649c4db6f391078bf42f8950
