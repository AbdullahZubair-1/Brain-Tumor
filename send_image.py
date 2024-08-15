import requests

# Specify the image file path
image_path = "C:\\Users\\onlyw\\Desktop\\Abdullah\ATS--INTERNSHIP\\5-brain tumor\\yes\\Y1.jpg"
# C:\\Users\\onlyw\\Desktop\\Abdullah\\ATS--INTERNSHIP\\brain tumor\\no\\2 no.jpeg
# Specify the URL
url = "http://127.0.0.1:8000/predict/"

# Open the image file in binary mode
with open(image_path, "rb") as file:
    files = {"file": file}
    # Send the POST request
    response = requests.post(url, files=files)

# Print the response
print(response.json())
