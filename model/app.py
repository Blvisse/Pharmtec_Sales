import mlflow
from flask import Flask,request,render_template
import logging
logged_model = 'runs:/cdda265c77d14d70a74c818df770ddd2/Best Model'
app=Flask(__name__)

logging.basicConfig(filename='logs/deploying.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)


logging.debug('------ Importing Libraries ----')
# Load model as a PyFuncModel.
# loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pickle 
import pandas as pd
from datetime import datetime
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def homepage():
    logging.debug('Calling Homepage route ')
    return render_template('homepage.html')

@app.route('/prediction',methods=['POST'])

def predict():
    print("Recieving request from users...")
    logging.debug("------------ Received request --------")
    store=request.form.get('store')
    Date=request.form.get('date')

    dateobject=datetime.strptime(Date,"%Y-%m-%d")
    
    
    Month=dateobject.month
    Year=dateobject.year
    DayOfweek=dateobject.weekday()
    Week=dateobject.isocalendar()[1]
    # DayOfWeek=DayOfweek.strftime('%Y-%m-%d') 
    Promo=request.form.get('promo') 
    SalesCustomer=request.form.get('salespercust') 
    CompDistance= request.form.get('compdistance')
    StateHoliday= request.form.get('StateHoliday')
    SchoolHoliday= request.form.get('school')
    Promo2=request.form.get('prom2open')
    Promo2Open=request.form.get('prom2open')
    assortment= request.form.get('assortment')
    competition_open= request.form.get('compopen')
    prom2month= request.form.get('prom2mon')
    storetype=request.form.get('storetype')
    weekend=request.form.get('weekend')
    daysafter= request.form.get('daysafter')
    daysbefore= request.form.get('daysbefore')
    vals=[store,Week,Year,Month,DayOfweek,Promo,SalesCustomer,CompDistance,StateHoliday,SchoolHoliday,Promo2Open,Promo2,assortment,competition_open,prom2month,storetype,daysafter,daysbefore,weekend]
    vals=pd.DataFrame(vals).T  
    # print(data)
    
    # vals= [str(x) for x in request.form.values()]
    # data=request.get_json(force=True)
    # data.update((X,[y] for X,y in data.items()))
    print("Calling Model")
    logging.debug(" --------------------------------Initlaizing Machine model --------------------------------")
    print(vals)
    prediction=model.predict(vals)
    print(prediction)
    print("Completed predictions")
    logging.debug(" -------- Completed predictions --------")

    return render_template('output.html', predictions=prediction)
    



if(__name__=='__main__'):
    # app.run(debug=True)
    app.run(debug=True)



# loaded_model.predict(pd.DataFrame(data))