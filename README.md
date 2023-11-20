# YouTube Hate Speech Detection

![YouTube](img/youtubecomment.jpg)

## Project Description

YouTube has been facing a growing challenge related to the increase in hate speech messages within video comments. In an effort to address this issue, we have developed a Machine Learning model using advanced techniques to automatically detect hate speech messages. This project focuses on providing a practical and efficient solution, prioritizing implementation over model precision.

## Folder Structure

- **img:** Contains images related to the project.
- **notebook:** Includes the Jupyter notebook (MVP.ipynb) where the model is developed and trained.
- **.gitignore:** Configuration file to ignore specific files and folders in version control.
- **README.md:** This document providing detailed information about the project.
- **app.py:** Main file containing the implementation of the hate speech detection solution.

## Implementation Levels

### Essential Level
- ML model that recognizes hate speech messages.
- Overfitting below 5%.
- Solution to operationalize the model through an interface, API, or scraper to check if a message is hate speech.

### Intermediate Level
- ML model with ensemble techniques to improve hate speech detection.
- Solution allowing the recognition of hate speech messages given a link to a specific video.

### Advanced Level
- Model based on neural networks to significantly enhance results.
- Solution enabling real-time hate speech detection by monitoring a video.

### Expert Level
- Model based on transformers or RNN (GRU or LSTM) for advanced detection.
- Real-time comment processing using queue servers and database access.

## Usage Instructions

1. Clone this repository to your local machine.
2. Explore the Jupyter notebook in the "notebook" folder to understand the model's development.
3. Run the `app.py` file to utilize the implemented practical solution with `streamlit run app.py` .

Contributions and improvements are welcome! Feel free to reach out to discuss ideas or make enhancements to the project.
