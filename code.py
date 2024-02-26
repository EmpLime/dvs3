from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QLineEdit, QMessageBox, QTextEdit, QGridLayout
from PyQt5.QtGui import QTextCursor, QPixmap
from PyQt5.QtCore import Qt
import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno
from boltzmannclean import clean
import os

def n_dt(dt):
    print("Starting normalization process...")
    # Select only the numerical columns for normalization
    num_dt = dt.select_dtypes(include=['float64', 'int64'])
    print("Numeric data before dropping NaN values:")
    print(num_dt.head())
    
    # Check if there are any NaN values in num_dt
    print("Checking for NaN values...")
    print(num_dt.isnull().sum())
    
    # Drop rows with missing values
    num_dt = num_dt.dropna()  
    print("Numeric data after dropping NaN values:")
    print(num_dt.head())
    
    if num_dt.empty:
        # Handle empty DataFrame here
        print("Numeric DataFrame is empty after dropping NaN values.")
        return pd.DataFrame()
    
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(num_dt)
    normalized_df = pd.DataFrame(normalized_data, columns=num_dt.columns)
    print("Normalization process completed successfully.")
    return normalized_df

def minmax_scale(dt):
    print("Starting MinMax scaling process...")
    # Select only the numerical columns for MinMax scaling
    num_dt = dt.select_dtypes(include=['float64', 'int64'])
    print("Numeric data before dropping NaN values:")
    print(num_dt.head())
    
    # Check if there are any NaN values in num_dt
    print("Checking for NaN values...")
    print(num_dt.isnull().sum())
    
    # Drop rows with missing values
    num_dt = num_dt.dropna()  
    print("Numeric data after dropping NaN values:")
    print(num_dt.head())
    
    if num_dt.empty:
        # Handle empty DataFrame here
        print("Numeric DataFrame is empty after dropping NaN values.")
        return pd.DataFrame()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(num_dt)
    scaled_df = pd.DataFrame(scaled_data, columns=num_dt.columns)
    print("MinMax scaling process completed successfully.")
    return scaled_df

class MainWindow(QMainWindow):
    def calculate_summary_statistics(self):
        if self.nd_dt.empty and self.dt.empty:
            print("No data available for summary statistics.")
            return
        elif self.nd_dt.empty:
            summary_stats = self.num_dt.describe()
            print("Summary Statistics for Data:")
            print(summary_stats)
        else:
            # Calculate summary statistics for self.nd_dt
            summary_stats = self.nd_dt.describe()
            print("Summary Statistics for Normalized Data:")
            print(summary_stats)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(100, 100, 400, 200)
        self.file_label = QLabel("No file chosen")
        self.ftrs_label = QLabel("features to plot:")
        self.ftrs_input = QLineEdit()  # Changed to QLineEdit for user input
        self.choose_file_button = QPushButton("Choose File", self)
        self.choose_file_button.clicked.connect(self.choose_file_dialog)
        self.normalize_button = QPushButton("Normalize Data", self)  # Added normalize button
        self.normalize_button.clicked.connect(self.normalize_data)  # Connect button to normalization method
        self.minmax_button = QPushButton("MinMax Scale Data", self)  # Added MinMax scaling button
        self.minmax_button.clicked.connect(self.minmax_scale_data)  # Connect button to MinMax scaling method
        self.plot_button = QPushButton("Plot", self)
        self.plot_button.clicked.connect(self.plot)
        self.missing_values_button = QPushButton("Show Missing Values Matrix", self)  # Button to show missing values matrix
        self.missing_values_button.clicked.connect(self.show_missing_values_matrix)  # Connect button to method
        self.summary_stats_button = QPushButton("Calculate Summary Statistics", self)
        self.summary_stats_button.clicked.connect(self.calculate_summary_statistics)
        layout = QVBoxLayout()
        layout.addWidget(self.choose_file_button)
        layout.addWidget(self.file_label)
        layout.addWidget(self.ftrs_label)
        layout.addWidget(self.ftrs_input)
        layout.addWidget(self.normalize_button)  # Add normalize button to layout
        layout.addWidget(self.minmax_button)  # Add MinMax scaling button to layout
        layout.addWidget(self.plot_button)
        layout.addWidget(self.missing_values_button)  # Add missing values button to layout
        layout.addWidget(self.summary_stats_button)  # Add summary stats button to layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.dt = pd.DataFrame()
        self.nd_dt = pd.DataFrame()  # Initialize normalized data frame

    def choose_file_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_path:
            self.file_label.setText(f"Chosen File: {file_path}")
            self.dt = pd.read_csv(file_path)
            self.ftrs_input.setText(f"ftrs: {', '.join(self.dt.columns)}")

    def normalize_data(self):
        if not self.dt.empty:
            print("Data before normalization:")
            print(self.dt.head())
            self.nd_dt = n_dt(self.dt)
            print("Data after normalization:")
            print(self.nd_dt.head())
        else:
            print("Please select a CSV file first")
    
    def minmax_scale_data(self):
        if not self.dt.empty:
            print("Data before MinMax scaling:")
            print(self.dt.head())
            self.nd_dt = minmax_scale(self.dt)
            print("Data after MinMax scaling:")
            print(self.nd_dt.head())
        else:
            print("Please select a CSV file first")
    
    def plot(self):
        if self.dt.empty:
            print("Please select a CSV file first")
            return
    
        if self.nd_dt.empty:
            data_to_plot = self.dt
        else:
            data_to_plot = self.nd_dt
    
        ftrs_to_plot = self.ftrs_input.text().split(',')
        valid_ftrs = [ftr.strip() for ftr in ftrs_to_plot if ftr.strip() in data_to_plot.columns]
    
        if not valid_ftrs:
            print("Invalid feature names, please enter valid features")
            return
    
        if len(valid_ftrs) < 2:
            print("Please select at least two numeric features for plotting.")
            return
    
        # Check if selected features are numeric
        if not data_to_plot[valid_ftrs].apply(pd.to_numeric, errors='coerce').notnull().all().all():
            print("Selected features contain non-numeric values. Please select numeric features.")
            return
    
        if len(valid_ftrs) == 2:
            # Create a 2D plot
            plt.figure(figsize=(8, 6))
            plt.scatter(data_to_plot[valid_ftrs[0]], data_to_plot[valid_ftrs[1]], alpha=0.7)
            plt.xlabel(valid_ftrs[0])
            plt.ylabel(valid_ftrs[1])
            plt.grid(True)
            plt.savefig('2D_plot.png')  # Save the plot as a PNG file
            plt.show()
            os.startfile('2D_plot.png')  # Open the saved PNG file
        else:
            # Create a 3D plot
            num_samples, num_features = data_to_plot[valid_ftrs].shape
            n_components = min(num_samples, num_features)
    
            pca = PCA(n_components=n_components)  # Adjust number of components
            principal_components = pca.fit_transform(data_to_plot[valid_ftrs])
    
            # Use the y-coordinate of each point as the parameter for coloring
            color_param = principal_components[:, 1]  # Use the second column of principal components
    
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
    
            # Use the y-coordinate values to assign colors to the points
            scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2],
                                 c=color_param, cmap='viridis', alpha=0.7)
    
            ax.set_xlabel(valid_ftrs[0])  # Set X-axis label
            ax.set_ylabel(valid_ftrs[1])  # Set Y-axis label
            ax.set_zlabel(valid_ftrs[2])  # Set Z-axis label
    
            # Add a color bar to the plot
            cbar = fig.colorbar(scatter, ax=ax, label='Color Map')
            cbar.set_label(valid_ftrs[1])
    
            plt.savefig('3D_plot.png')  # Save the plot as a PNG file
            plt.show()
            os.startfile('3D_plot.png')  # Open the saved PNG file

    
    def show_missing_values_matrix(self):
        if self.dt.empty:
            print("Please select a CSV file first")
            return
    
        if self.nd_dt.empty:
            msno.matrix(self.dt)
        else:
            msno.matrix(self.nd_dt)
    
        plt.savefig('missing_values_matrix.png')  # Save the missing values matrix plot as a PNG file
        plt.show()
        os.startfile('missing_values_matrix.png')  # Open the saved PNG file

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
