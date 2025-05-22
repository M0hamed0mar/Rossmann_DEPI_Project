# AIS4_S1_3 – DEPI-Team Project

Welcome to the official repository for the **AIS4_S1_3** project, developed by the **DEPI-Team**. This project demonstrates a blend of data analysis, visualization, and web development using Python and Flask.

## 🚀 Project Overview

AIS4_S1_3 is a data-driven application built to analyze and visualize structured data for extracting meaningful insights. It uses Python's data science ecosystem (NumPy, Pandas, Matplotlib, etc.) for backend processing and provides a clean, interactive user interface through a Flask web application.

## 🧰 Features

- 📊 **Exploratory Data Analysis** (EDA) with Jupyter Notebooks
- 🌐 **Flask Web Interface** for visualization and interaction
- 🧠 **Modular Design** separating analysis, backend, and frontend logic
- 📁 **Support for Preloaded Dataset** (compressed in `data.rar`)
- 📷 **Image Display and Analysis** options in the web interface

## 📁 Project Structure

```
AIS4_S1_3/
├── Notebooks/          # Jupyter Notebooks for data exploration
├── models/             # Any ML or data models used in the project
├── static/             # Static files (CSS, JS, images)
├── templates/          # HTML templates for Flask frontend
├── app.py              # Main Flask application script
├── eda_utils.py        # Utility functions for data analysis
├── requirements.txt    # Python dependencies
├── data.rar            # Compressed dataset (extract before use)
└── README.md           # Project documentation
```

## 🛠️ Installation

Follow these steps to set up and run the project locally:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/AbdalrahmanOthman01/AIS4_S1_3.git
   cd AIS4_S1_3
   ```

2. **(Optional) Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Extract the dataset**:

   Unzip or extract the contents of `data.rar` into the project directory if applicable.

## 🚀 Usage

Run the Flask application locally:

```bash
python app.py
```

Then open your browser and visit:

```
http://localhost:5000
```

## 📊 Data Analysis

The `Notebooks/` folder contains Jupyter Notebooks used for analyzing the dataset. These notebooks showcase how data is processed, visualized, and interpreted before being integrated into the web interface.

Use them to:
- Understand data distribution
- Plot charts and graphs
- Explore feature relationships
- Test preprocessing methods

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork this repository.
2. Create your feature branch:

   ```bash
   git checkout -b feature/YourFeature
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add Your Message"
   ```

4. Push to the branch:

   ```bash
   git push origin feature/YourFeature
   ```

5. Open a pull request with a clear explanation of your changes.

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this software with proper attribution.


---

*This README was crafted to provide a clear and professional overview of the AIS4_S1_3 project. Feel free to update any section to reflect new features or changes.*
