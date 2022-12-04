import pandas as pd 
from bs4 import BeautifulSoup
from datetime import datetime

#Wed, Feb 24, 09:43 AM 

def parser(file_name,start_date,end_date):
	trip_date=[]
	trip_fair=[]
	trip_cab_type=[]
	trip_from=[]
	trip_id=[]
	objDate=[]
	soup = BeautifulSoup(file_name.read(),"html.parser")
	all_data = soup.find_all(class_="card cust-card")
	for i in all_data:
		if i.find(class_="journey-info right"):
			trip_date.append(i.find("div", {"class":"journey-info left"}).getText().split(",")[1].replace(" ",""))
			trip_fair.append(i.find("div",{"class":"journey-info right"}).getText())
			trip_cab_type.append(i.find("div",{"class":"left cab-type"}).getText())
			trip_id.append(i.find("div",{"class":"left crn"}).getText())
			for j in i.find_all("div",{"class":"destination truncate"}):
				trip_from.append(j.getText().replace(" ",""))

		else:
			pass
	for i in trip_date:
		objDate.append(datetime.strptime(i, '%b%d'))

	data = {
	"Date": trip_date,
	"Trip-ID": trip_id,
	"Cab-type": trip_cab_type, 
	"From":trip_from[::2],
	"To":trip_from[1::2], 
	"Fair": trip_fair
	

	}
	df = pd.DataFrame(data)
	df_out = df[(df['Date'] <= end_date) & (df['Date'] >= start_date)]
	df_out = df_out.reset_index(drop=True)
	#print(trip_date)
	return df_out
if __name__ == '__main__':
	parser("Ola Support.html","Mar01","Mar31")

#print(trip_date)
