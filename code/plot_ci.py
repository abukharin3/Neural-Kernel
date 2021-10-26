import csv
import torch
import numpy as np
import json
import branca
import folium

deltas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 5]
deltas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 5]

d = 1e-4



ub = np.load("Results/insample/predicted_cases_ub2%s.npy" % d)
lb = np.load("Results/insample/predicted_cases_lb2%s.npy" % d)
ci = ub - lb


#--------------------------------------------------------------------------
#
# LOAD DATA MATRICES
#
#--------------------------------------------------------------------------


# confirmed cases and deaths
C = np.load("data/mat/ConfirmedCases_1-17.npy") # [ T, n_counties ]
D = np.load("data/mat/death_1-17.npy") 
H = np.load("data/mat/hotspot_1-17.npy")       # [ T, n_counties ]

# Load covariates
M      = np.load("data/mat/mobility_1-17.npy").transpose([2,0,1]) # [ n_mobility, T, n_counties ]
pop    = np.load("data/mat/population.npy")
over60 = np.load("data/mat/over60.npy")
cov    = np.array([pop, over60])                                                # [ n_covariates, T, n_counties ]


# Normalize data
C = C / (np.expand_dims(pop, 0)+1) * pop.mean()
D = D / (np.expand_dims(pop, 0)+1) * pop.mean()

# print(ci.shape, C[2:].shape)
# ci = ci / (C[2:] + 1)
T, n_counties = C.shape # 3144
n_mobility    = M.shape[0]
n_covariates  = cov.shape[0]

#--------------------------------------------------------------------------
#
# LOAD META DATA AND CONFIGURATIONS
#
#--------------------------------------------------------------------------

# Distance matrix for counties
distance = np.sqrt(np.load("data/mat/distance.npy"))  # [ n_counties, n_counties ]
adj      = np.load("data/mat/adjacency_matrix.npy")   # [ n_counties, n_counties ]
# FIPS for US counties
I        = np.load("data/mat/counties.npy").tolist()
loc_dict = {}
# Raw file for US counties
with open('data/meta/county_centers.csv', newline='') as csvfile:
	locsreader = list(csv.reader(csvfile, delimiter=','))
	for row in locsreader[1:]:
		if row[1] != "NA":
			fips, lon, lat = int(row[0]), float(row[1]), float(row[2])
			loc_dict[fips] = [lon, lat]
		else:
			print(row)
# Geolocation (longitude and latitude) of US counties
locs = np.array([ loc_dict[int(i)] for i in I ]) # [ n_counties, 2 ]
# Normalization
# print("FIPS", I)
# print("Geolocation matrix shape", locs.shape)


with open (r"data/meta/states-10m.json", "r") as f:
	state_json = json.load(f)
with open(r"data/meta/us_counties_20m_topo.json", "r") as f:
	uscountygeo = json.load(f)


counties = list(np.load("data/mat/counties.npy"))
ci -= ci.min()
ci /= (pop + 1)
ci = np.log(ci + 1e-5)
ci = ci - ci.min()
ci = ci / ci.max() * 300  - 70
ci *= 5
# ci = np.log(ci + 1e-3)

# ci -= ci.min()
# ci = np.log(ci + 1e-3)

for i in range(49):
	print(i)
	colorscale = branca.colormap.linear.Blues_09.scale(0, 1100 / 2)
	#colorscale = branca.colormap.linear.Blues_09.scale(ci.min(), ci.max())
	colorscale.caption = "Confidence Interval (Number of Cases) Week " + str(i) 

	def style_function(feature):
		county = int(feature['id'][-5:])
		try:
			data=ci[i, counties.index(str(county))]
			#print(ci[i, counties.index(str(county))] + 1, data)
		except Exception as e:
			data = ci[i].mean()
		return {
			'fillOpacity': 0.5,
			'weight': 0,
			'fillColor': '#black' if data is None else colorscale(data)
		}

	m = folium.Map(
	location=[38, -95],
	tiles='cartodbpositron',
	zoom_start=4
	)

	folium.TopoJson(
	uscountygeo,
	'objects.us_counties_20m',
	style_function=style_function
	).add_to(m)

	folium.Choropleth(geo_data=uscountygeo,
	   topojson='objects.us_counties_20m',
	   line_weight=0.1,
	   fill_opacity=0.0).add_to(m)

	folium.Choropleth(
	geo_data=state_json,
	topojson='objects.states',
	line_weight=0.15,
	fill_opacity=0.0
	).add_to(m)


	colorscale.add_to(m)
	m.save("ci/ci{}.html".format(i))