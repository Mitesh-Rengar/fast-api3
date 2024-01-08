from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import sklearn, fastapi,pydantic


# print(fastapi.__version__)
# print(pydantic.__version__)
# print(sklearn.__version__)
# print(pd.__version__)





class MatchData(BaseModel):
    batting : str
    bowling : str
    city : str
    total_run : int
    score : int
    over : int
    wickets :  int

with open('pipe.pkl','rb') as f:
    model = pickle.load(f)


app = FastAPI()

@app.post('/')
async def fun(item:MatchData):
    
    rl = item.total_run - item.score
    bl = 120 - (item.over*6)
    wl = 10 - item.wickets
    crr = item.score/item.over
    rrr = rl / (bl/6)
    if item.score >= item.total_run:
        return {item.batting : '= '+ str(100)+'%',
                item.bowling : '= '+ str(0)+'%',
                }

    else:
        df=pd.DataFrame({'batting_team':[item.batting], 'bowling_team':[item.bowling], 'city':[item.city],'run_l':[rl], 'ball_l': [bl], 'wickets_left':[wl],'total_runs_x':[item.total_run],'crr':[crr],'rrr':[rrr]})
        result = model.predict_proba(df)

    # print(result)
    # return {'pre':y_hat}
    return {item.batting:round(result[0][0],2),
            item.bowling:round(result[0][1],2)}
