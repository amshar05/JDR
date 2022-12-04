import pandas as pd
from datetime import timedelta
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
#import tkinter
import sys
import matplotlib
from matplotlib import dates
#matplotlib.use('TkAgg')





total_cases = pd.read_csv("https://api.covid19india.org/csv/latest/districts.csv")
total_cases.fillna(0)
#print(total_cases.columns)
total_cases_har = total_cases[total_cases["State"] == 'Haryana']
#print(total_cases[total_cases["State"] == 'Haryana'])
###list of all districts###
districts = []
for i in total_cases_har.District.unique():
	districts.append(i)

print(districts)

unwanted_num = {'Italians','Unknown', 'Foreign Evacuees'}

districts = [ele for ele in districts if ele not in unwanted_num]
print(districts)
dict_info = {}
for j in districts:
	total_cases_har_amb = total_cases_har[total_cases_har['District'] == j]
	total_cases_har_amb = total_cases_har_amb.iloc[-180:]
	total_cases_har_amb["Date"] = pd.to_datetime(total_cases_har_amb['Date'])
	total_cases_har_amb["Date"] = total_cases_har_amb["Date"].dt.date
	#total_cases_har_amb = total_cases_har_amb.iloc[90:]
	new = total_cases_har_amb.set_index("Date")
	

	#deleting unwanted columns
	del new['State']
	del new['District']
	del new["Other"]
	del new['Tested']

	values = []
	values.append(new.iloc[[-1]].index.values[0]) #last date avaiable
	values.append(new["Confirmed"].iloc[[-1]].values[0])
	values.append(new["Recovered"].iloc[[-1]].values[0])
	values.append(new["Deceased"].iloc[[-1]].values[0])
	
	dict_info[str(j)] = values


	new = new.pct_change()
	####info dictionary

	i = 0
	while i <= 13:
		last_date = new.iloc[[-1]].index.values[0]
		last_date1 = last_date + timedelta(days=1)
		new_2 = pd.DataFrame([((new.iloc[-1]+new.iloc[-2]+new.iloc[-3]+new.iloc[-4]+new.iloc[-5]+new.iloc[-6]+new.iloc[-7]+
			new.iloc[-8]+new.iloc[-9]+new.iloc[-10]+new.iloc[-11]+new.iloc[-12]+new.iloc[-13]+new.iloc[-14])/14)],index=[last_date1])
		globals()[f"dist_{j}"] = new.append(new_2)
		i= i + 1
	plt.xlabel("Date")
	plt.ylabel("Rate")
	plt.title(str(j))

	plt.plot( "Confirmed" ,data = globals()[f"dist_{j}"].iloc[:-14], color='orange', linewidth=1.3, label= "Confirmed Cases")
	plt.plot( "Confirmed",data = globals()[f"dist_{j}"].iloc[-14:], color='orange', linewidth=1.3, linestyle = 'dotted', label="Predicted Cases")
	plt.plot( "Recovered",data = globals()[f"dist_{j}"].iloc[:-14], color='green', linewidth=1.3, label="Recovered")
	plt.plot( "Recovered",data = globals()[f"dist_{j}"].iloc[-14:], color='green', linewidth=1.3, linestyle = 'dotted', label= "Predicted Recovered")
	plt.plot( "Deceased",data = globals()[f"dist_{j}"].iloc[:-14], color='blue', linewidth=1.3, label = "Deceased")
	plt.plot( "Deceased",data = globals()[f"dist_{j}"].iloc[-14:], color='blue', linewidth=1.3, linestyle = 'dotted', label = "Predicted Deceased")
	#plt.xticks(rotation=45)
	plt.legend(loc="upper left")
	plt.savefig("static/"+str(j)+".png")
	plt.clf()



data_df = pd.DataFrame.from_dict(dict_info, orient='index')
data_df.to_excel('data_info.xlsx')


