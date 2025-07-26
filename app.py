import uvicorn
from fastapi import FastAPI
import fastai
import fastbook
from fastai.imports import *
from fastai.tabular.all import *
from BioValues import Biovalue
app  = FastAPI()
model = load_learner('nn_92-8_no_age.pkl')
@app.get('/')
def index():
    return {'message':'Hi there.'}
@app.get('/Welcome')
def get_name(name:str):
    return {'Welcome to NTA-ELISA Model :': name}
@app.post('/predict')
def predict_type(data:Biovalue):
    data = data.dict()
    print(data)
    Gender      = data['Gender']
    NTA_Scatter = data['NTA_Scatter']
    ELISA       = data['ELISA']
    df = pd.DataFrame([[Gender, float(NTA_Scatter), float(ELISA)]],
                      columns=['Gender', 'NTA-Scatter', 'ELISA'])
    pred_class, pred_idx, probs = model.predict(df.iloc[0])
    class_names = model.dls.vocab
    # Format results for table display
    prob_df = pd.DataFrame({
        "Class": class_names,
        "Confidence": [f"{p:.4f}" for p in probs]
    })

    return {'prediction' : (str(pred_class), prob_df)}


if  __name__ == '__main__' :
    uvicorn.run(app,host = '127.0.0.1' , port = 8000)