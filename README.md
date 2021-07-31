# Pharmtec_Sales
## Summary
A Pharmtec company wants to forecast sales in their stores acrros several cities six weeks ahead of time. This repo uses the given data to extract the features and use them to help predict store sales accross the stores

## Data
The data will be stored in a dvc repo. The data's given features include

**Id** - an Id that represents a (Store, Date) duple within the test set

**Store** - a unique Id for each store

**Sales** - the turnover for any given day (this is what you are predicting)

**Customers** - the number of customers on a given day

**Open** - an indicator for whether the store was open: 0 = closed, 1 = open

**StateHoliday** - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

**SchoolHoliday** - indicates if the (Store, Date) was affected by the closure of public schools

**StoreType** - differentiates between 4 different store models: a, b, c, d

**Assortment** - describes an assortment level: a = basic, b = extra, c = extended. Read more about assortment here (https://en.wikipedia.org/wiki/Retail_assortment_strategies)

**CompetitionDistance** - distance in meters to the nearest competitor store

**CompetitionOpenSince[Month/Year]** - gives the approximate year and month of the time the nearest competitor was opened

**Promo** - indicates whether a store is running a promo on that day

**Promo2** - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating

**Promo2Since[Year/Week]** - describes the year and calendar week when the store started participating in Promo2

**PromoInterval** - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

## Code Structure 
1. .dvc folder contains references to data stored in the dvc repository 
2. .gihub/workflows contains cml yaml file thate uses github actions to generate report when new changes are pushed to the repository
3. data older houses our data.dvc tracking files 
4. log folder contain running logs of all the sripts, scripts for various models are preffixed with the script they log eg dvc.log logs data from the dvc script
5. mlruns is a folder that contains mlflow data on models and data, it pushed and tacks models trained 
6. notebooks contain all the notebooks used 
     * pharmtech- this notebook contains eda and sumary of  the intial data initial data 
     * Preprocessing notebook, involves preprocessing scripts to clean and feature engineer our data 
 7. scripts folder contains all of our scripts that we reference in the notebook a brief descrition is made at the begining of a script expalaing what it does
 8. templates are rendered tamplataes for our deployed model, This are basic html files that style our front end 
 9. test folder contains unit tests carried out on programs in thie repo
 10. The .png files are all logged metrics from our training models 
 11. The procfile is an essential document for deployment of our model to heroku 
 12. model.pkl is our trained model hyperparameter tuned and ready for deserialization 
