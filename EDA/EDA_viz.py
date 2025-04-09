import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Function to display menu and perform EDA operations
def eda_operations(df):
    while True:
        print("\nChoose an EDA operation:")
        print("1. Find null values")
        print("2. Find empty values")
        print("3. Display dataset info")
        print("4. Visualize dataset (histogram)")
        print("5. Visualize two columns (scatter/bar chart)")
        print("6. Clean dataset (remove null values)")
        print("7. Feature engineering (create new feature)")
        print("8. Basic machine learning model")
        print("9. Exit")

        choice = input("Enter your choice (1/2/3/4/5/6/7/8/9): ")

        if choice == '1':
            # Find null values
            print("\nNull values in the dataset:")
            print(df.isnull().sum())
        elif choice == '2':
            # Find empty values
            print("\nEmpty values in the dataset:")
            print((df == '').sum())
        elif choice == '3':
            # Display dataset info
            print("\nDataset Info:")
            print(df.info())
        elif choice == '4':
            # Visualize dataset (histogram)
            try:
                df.hist(figsize=(10, 8))
                plt.show()
            except Exception as e:
                print(f"Error visualizing dataset: {e}")
        elif choice == '5':
            # Visualize two columns (scatter/bar chart)
            try:
                print("\nAvailable columns:")
                print(df.columns)

                col1 = input("Enter the name of the first column: ")
                col2 = input("Enter the name of the second column: ")

                if col1 in df.columns and col2 in df.columns:
                    print("\nChoose a visualization type:")
                    print("1. Scatter plot")
                    print("2. Bar chart")

                    viz_choice = input("Enter your choice (1/2): ")

                    if viz_choice == '1':
                        plt.figure(figsize=(8, 6))
                        plt.scatter(df[col1], df[col2])
                        plt.xlabel(col1)
                        plt.ylabel(col2)
                        plt.title(f"Scatter Plot of {col1} vs {col2}")
                        plt.show()
                    elif viz_choice == '2':
                        if df[col1].dtype == 'object':
                            plt.figure(figsize=(8, 6))
                            df.groupby(col1)[col2].mean().plot(kind='bar')
                            plt.xlabel(col1)
                            plt.ylabel(col2)
                            plt.title(f"Bar Chart of {col1} vs {col2}")
                            plt.show()
                        else:
                            print("Bar chart is more suitable for categorical data.")
                    else:
                        print("Invalid choice.")
                else:
                    print("One or both columns not found.")
            except Exception as e:
                print(f"Error visualizing two columns: {e}")
        elif choice == '6':
            # Clean dataset (remove null values)
            try:
                df_clean = df.dropna()
                print("\nCleaned dataset info:")
                print(df_clean.info())
                df = df_clean  # Update df with cleaned version
            except Exception as e:
                print(f"Error cleaning dataset: {e}")
        elif choice == '7':
            # Feature engineering (create new feature)
            try:
                print("\nAvailable columns:")
                print(df.columns)

                col1 = input("Enter the name of the first column: ")
                col2 = input("Enter the name of the second column: ")

                if col1 in df.columns and col2 in df.columns:
                    df['NewFeature'] = df[col1] + df[col2]
                    print("\nNew feature added to the dataset:")
                    print(df.head())
                else:
                    print("One or both columns not found.")
            except Exception as e:
                print(f"Error creating new feature: {e}")
        elif choice == '8':
            # Basic machine learning model
            try:
                print("\nAvailable columns:")
                print(df.columns)

                feature_col = input("Enter the name of the feature column: ")
                target_col = input("Enter the name of the target column: ")

                if feature_col in df.columns and target_col in df.columns:
                    X = df[[feature_col]]
                    y = df[target_col]

                    # Check for null values
                    if X.isnull().values.any() or y.isnull().values.any():
                        print("Null values found. Please clean the data first.")
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        print(f"\nMachine learning model MSE: {mse}")
                else:
                    print("One or both columns not found.")
            except Exception as e:
                print(f"Error running machine learning model: {e}")
        elif choice == '9':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


# Main function to load dataset and perform EDA operations
if __name__ == "__main__":
    # File path provided
    file_path = "C:/Users/Blackpearl Computers/Downloads/Bike Prices.csv"

    try:
        # Load the dataset into a DataFrame
        df = pd.read_csv(file_path)
        print(f"\nDataset '{file_path}' loaded successfully!")

        # Perform EDA operations
        eda_operations(df)
    except Exception as e:
        print(f"Error loading dataset: {e}")
