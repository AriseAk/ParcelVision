import requests

url = "http://127.0.0.1:5001/detect"

files = {
    "image": open("test.png", "rb")  # put a box image here
}

res = requests.post(url, files=files)

print(res.json())