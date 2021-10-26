# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:17:41 2020

@author: Khurram Yamin
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as pltx


# daily_covid = np.load("data\mat\DailyCases_1-17.npy")
# n_days = daily_covid.shape[0]
# n_counties = daily_covid.shape[1]

# hotspot_covid = np.zeros((n_days,n_counties))

df = pd.DataFrame(columns = ["Time","FIPS"])

weeks = np.load(r'data\mat\weeks_1-17.npy')
counties = list(np.load("data\mat\counties.npy"))



times_list =[]
counties_list=[]


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
            
np.save("data/mat/daily_hotspot.npy", hotspot_covid)           

# weekly_hotspots = np.zeros((n_days//7,n_counties))

# for county in range (0,n_counties):
#     for day in range (0,n_days,7):
#         weekly_hotspots[day//7,county]= min(hotspot_covid[day:day+6, county].sum(),1)
#         if weekly_hotspots[day//7,county] == 1:
#             times_list.append(day//7)
#             counties_list.append(counties[county])

# #For Woody, CSV of only when hotspot=1 for a given week
# length = weekly_hotspots.shape[0]
# df["Time"] = times_list
# df["FIPS"] = counties_list
# df.to_csv("hotspots_1-17",index=False)

#For Liyan, Dictionary of format Day:List of values

# bigDict={}
# for i in range (0,weekly_hotspots.shape[0]):
#     bigDict[i]=list(weekly_hotspots[i,:])

# f = open(r'C:\Users\Khurram Yamin\covid\hotspotDict_12_12',"wb")
# pickle.dump(bigDict,f)
# f.close()
    


# with open (r"inputs\states-10m.json", "r") as f:
#     state_json = json.load(f)
# with open(r"inputs\counties.json", "r") as f:
#     county_geo = json.load(f)
# with open(r"inputs\us_counties_20m_topo.json", "r") as fgeo:
#     uscountygeo = json.loads(fgeo.read())

# '''
# i =13
# county = 10001
# data=weekly_hotspots[i, counties.index(str(county))]
# print(data)
# '''

# for i in range (0, length):
    
    
#     def style_function(feature):
        
        
        
#         county = int(feature['id'][-5:])
#         try:
#             data=weekly_hotspots[i, counties.index(str(county))]
            
#         except Exception as e:
#             data = 0
#         return {
#             'fillOpacity': 0.5,
#             'weight': 0,
#             'fillColor': '#black' if data is None else colorscale(data)
#         }
    
    
    
#     colorscale = branca.colormap.linear.YlOrRd_09.scale(0, 1)
#     colorscale.caption = "hotspots in week" + str(weeks[i])
    
    
#     m = folium.Map(
#     location=[38, -95],
#     tiles='cartodbpositron',
#     zoom_start=4
#     )
    
#     folium.TopoJson(
#     uscountygeo,
#     'objects.us_counties_20m',
#     style_function=style_function
#     ).add_to(m)
    
#     folium.Choropleth(geo_data=uscountygeo,
#        topojson='objects.us_counties_20m',
#        line_weight=0.1,
#        fill_opacity=0.0).add_to(m)
    
#     folium.Choropleth(
#     geo_data=state_json,
#     topojson='objects.states',
#     line_weight=0.15,
#     fill_opacity=0.0
#     ).add_to(m)
    
#     colorscale.add_to(m)
#     img_data = m._to_png(5)
#     img = Image.open(io.BytesIO(img_data))
#     img.save(r'C:\Users\Khurram Yamin\covid\hotspots\_'+ str(weeks[i])  +'.png')
#     print(i)