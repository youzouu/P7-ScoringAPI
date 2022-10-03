from flask import Flask, jsonify, request
import pandas as pd
import pickle as pc
from flask_restful import Api, Resource



#from dashboard import df_client

app = Flask(__name__)
api = Api(app)

#import the model and the file
df_api = pd.read_csv('df_api.csv')

model_pip = pc.load(open('model_pip.pkl', 'rb'))


class Score(Resource):
    def get(self, id_curr):

        df_client = df_api[df_api['SK_ID_CURR'] == id_curr].iloc[:,3:]

        #calcul de la proba
        id_proba = model_pip.predict_proba(df_client)


        return {"SK_ID_CURR": id_curr,"data": list(df_client.T.to_dict().values())[0], 'proba':id_proba[0][1]}


api.add_resource(Score, "/score/<int:id_curr>/")
if __name__ == "__main__":
    app.run()
