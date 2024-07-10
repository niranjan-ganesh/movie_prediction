import pandas as pd
import requests
import os
from PIL import Image
import time

f = open("api_key.txt", "r")
api = f.read()

start_time = time.perf_counter() # Capturing start_time
# Path to your CSV file containing movie_id and poster_path
file_path = "file_path"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path, dtype={10: str})

# Function to validate image
def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Verify that it is, in fact, an image
        return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid image: {file_path} - {e}")
        return False

# Function to download movie posters
def download_poster(id, poster_path, api_key, download_dir='posters/'):
    url = f"https://image.tmdb.org/t/p/original/{poster_path}"
    response = requests.get(url)
    if response.status_code == 200:
        # Successful download
        os.makedirs(os.path.join(download_dir, str(id)), exist_ok=True)
        file_path = os.path.join(download_dir, str(id), f"{id}.jpg")  # Save in folder named after movie_id
        with open(file_path, 'wb') as f:
            f.write(response.content)
        if is_valid_image(file_path):
            return "Yes"
        else:
            os.remove(file_path)  # Remove invalid image
            return "No"
    else:
        # Failed to download
        return "No"

# Replace with your TMDb API key
api_key = api

# Download posters and save results
df['download_successful'] = df.apply(lambda row: download_poster(row['id'], row['poster_path'], api_key), axis=1)

# Save movie ID, poster path, and download success status to a CSV file
output_file_path = 'movie_id_poster.csv'
df.to_csv(output_file_path, index=False)

print(f"Download complete. Results saved to {output_file_path}.")
end_time = time.perf_counter() # Capturing end_time
elapsed_time = end_time - start_time # To calculate the time taken to run this program

print(f"Time:{elapsed_time // 3600} hours, {elapsed_time % 60} minutes")
