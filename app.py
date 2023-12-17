import pickle
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote

app = FastAPI()

pickle_file = open('classifier.pkl','rb')
classifier = pickle.load(pickle_file)

@app.get('/')
def home():
    return {'message': 'Hello! Welcome'}

@app.get('/predict')
def predict(data:BankNote):
    data = data.dict()
    variance = data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']

    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])

    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)