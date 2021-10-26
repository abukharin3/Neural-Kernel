import numpy as np

daily_covid = np.load("data/mat/DailyCases_1-17.npy")
weeks = np.load('data/mat/weeks_1-17.npy')

counties = list(np.load("data/mat/counties.npy"))

n_days = daily_covid.shape[0]
n_counties = daily_covid.shape[1]

hotspot_covid = np.zeros((n_days,n_counties))


for county in range (0,n_counties):
	for day in range (31,n_days-1):
		if ((daily_covid[day-6:day, county].sum() > 100) 
			and (daily_covid[day-6:day+1, county].sum()>daily_covid[day-13:day-6, county].sum())
			and (daily_covid[day-2:day+1, county].sum()>.4*daily_covid[day-5:day-2, county].sum())
			and (daily_covid[day-6:day+1, county].sum()>.31*daily_covid[day-29:day+1, county].sum())
			and (daily_covid[day-2:day+1, county].sum()>1.6*daily_covid[day-5:day-2, county].sum() or
				 daily_covid[day-6:day+1, county].sum()>1.6*daily_covid[day-13:day-6, county].sum())
			) :
			hotspot_covid[day, county] = 1

print(np.sum(hotspot_covid))
np.save("data/mat/daily_hotspot.npy", hotspot_covid)

weekly_hotspots = np.zeros((weeks.shape[0], n_counties))

for county in range (0,n_counties):
	for day in range (0,n_days,7):
		week = day // 7	
		weekly_hotspots[week,county]= min(hotspot_covid[day:day+7, county].sum(),1)

np.save("data/mat/hotspot_1-17.npy", weekly_hotspots)
print(np.shape(weekly_hotspots), np.sum(weekly_hotspots))