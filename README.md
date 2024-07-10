# Introduction
Conceptualized as an exhaustive project, I'm trying to predict the box office performance of upcoming movies based on a diverse feature set like posters, trailers, videos, etc. The current version of this project predicts movie revenue based on poster images only. Thanks to Rounak (https://github.com/rounakbanik) and TMDB for the data source. Rounak's project led me to TMDB whose API I used to gather further data.

# Data Collection
The dataset was sourced from TMDB which was written to a CSV file. The data included fields such as 'adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id', 'imdb_id', 'original_language', 'original_title', 'overview', 'popularity', 'poster_path', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 'spoken_languages', 'status', 'tagline', 'title', 'video', 'vote_average', 'vote_count'. I included the 'download_successful' field to indicate whether the poster image was successfully downloaded as some posters weren't available on TMDB. Additionally, the revenue values were normalized between 1 and 10 to standardize the input range for the neural network. The downloaded posters were transformed to ensure consistency.

# Dataset Class
I created a custom MovieDataset class to handle the loading and transformation of images. This class is designed to handle the loading and transformation of images for this project.

# Neural Network Architecture
The network architecture consists of a simple convolutional neural network (CNN) with two convolutional layers and three fully connected layers. The model was trained using the mean squared error (MSE) loss function and the Adam optimizer. Early stopping was implemented to prevent overfitting. The model was able to predict the outcome of a movie with 89% accuracy.

# Future Work
This project is part of a larger ambition to build a comprehensive movie prediction system. Future work involves incorporating additional features such as trailers, teasers, music among others and combining these results with the current model to enhance prediction accuracy.
