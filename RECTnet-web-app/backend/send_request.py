import requests

pic = open('./dog.jpg', 'rb')
print(pic)

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('./dog.jpg','rb')})
                    # rb reads binary format of file
print(resp.json())