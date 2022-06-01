import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from copy import deepcopy
import time
import tensorflow as tf
from tensorflow import keras
import concurrent.futures
import streamlit as st
from io import BytesIO
import pandas as pd
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.patches import Polygon
_lock = RendererAgg.lock


##################################################################################################
class NN:
	def __init__(self, model, out_property_name, out_property_unit):
		self.model = keras.models.load_model(model, compile=False)
		self.out_property_name = out_property_name
		self.out_property_unit = out_property_unit

cdict = {'red': [(0.0, 0.0078, 0.0078),
                 (0.06249, 0.0078, 0.0078),
                 (0.0625, 0.0, 0.0),
                 (0.34375, 1.0, 1.0),
                 (0.46875, 1.0, 1.0),
                 (0.8125, 1.0, 1.0),
                 (0.81251, 0.85, 0.85),
                 (1.0, 0.85, 0.85)],
        'green': [(0.0, 0.0078, 0.0078),
                 (0.06249, 0.0078, 0.0078),
                 (0.0625, 0.58, 0.58),
                 (0.34375, 1.0, 1.0),
                 (0.46875, 0.0, 0.0),
                 (0.8125, 0.0, 0.0),
                 (0.81251, 0.85, 0.85),
                 (1.0, 0.85, 0.85)],
        'blue': [(0.0, 1.0, 1.0),
                 (0.06249, 1.0, 1.0),
                 (0.0625, 0.0, 0.0),
                 (0.34375, 0.0, 0.0),
                 (0.46875, 0.0, 0.0),
                 (0.8125, 0.0, 0.0),
                 (0.81251, 0.85, 0.85),
                 (1.0, 0.85, 0.85)]


        }

cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
path = 'data/'


mat_model = NN(path + 'nn_mat.h5', 'EasyRo', 'EzRo')
history_model = NN(path + 'nn_mat_history.h5', 'EasyRo', 'EzRo')
temp_model = NN(path + 'nn_temp.h5', 'Temperature', 'C')
transform_data = np.load(path + 'transform.npz')
print(transform_data['gradient'])
print(transform_data['intercept'])

f = 'data/STS_ezRo.csv'
ro_sts = pd.read_csv(f, sep=';')

def summary(model_data):
	text = '''
	Interactive 1D basin model using a Neural Network trained for the Gulf of Mexico geological setting.  
	This is for educational purposes only and should not be used for decision making.  

	You can call it programatically (The code snippet below is up to date with the current app values !):  
	```python
	import pydaisi as pyd
	import numpy as np

	gom_model = pyd.Daisi("laiglejm/Interactive GOM 1D Salt")
	One_dim_model = np.array([0,'''
	for i in range(1,16):
		text += str(int(model_data[i])) + ","
	for i in range(16, 29):
		text += str(round(model_data[i], 2)) + ","
	text += str(int(model_data[29])) + ","
	text += "{:.2e}".format(model_data[30]) + ","
	text += str(int(model_data[31])) + ","
	text += "{:.2e}".format(model_data[32]) + ","
	text += str(int(model_data[33])) + "]).reshape((34,1,1))"
	text += '''
	# dims 0 to 15 = depths, dims 16 to 29 = lithos ratios, dim 30 = Crust Thickness, dim 31 = Lower Crust RHP, dim 32 = Upper Mantle thickness, dim 33 = Upper Crust RHP, dim 34 = Present day surface temperature
	# call the prediction with variable='temperature' or variable='maturity'
	result = gom_model.get_predictions(data=One_dim_model, variable='temperature').value.flatten()
	```

	'''
	return text

##################################################################################################
def transform(X, A, B, gradient, intercept):

	return (np.diag(1. / (B.dot(X) + gradient))).dot((A.dot(X) - intercept))


##################################################################################################
def group_transform(largeX, A, B, gradient, intercept):
	return [transform(X, A, B, gradient, intercept) for X in largeX]

##################################################################################################
def split_data_array(data_array, batches):
	nb_samples = data_array.shape[0]
	lim = int(nb_samples / batches)

	largeX_list = [data_array[i*lim:(i+1)*lim] for i in range(batches-1)]
	largeX_list.append(data_array[(batches-1)*lim:nb_samples])

	return largeX_list

##################################################################################################
def prepare_vector(transform_data, data_array):

	u = transform_data
	A = u['A']
	B = u['B']
	gradient = u['gradient']
	intercept = u['intercept']

	batches = 1
	data_array2 = deepcopy(data_array.T)

	if batches < 4:
		final = np.vstack([transform(a, A, B, gradient, intercept) for a in data_array2])
	else:
		largeX_list = split_data_array(data_array2, batches)

		executor = concurrent.futures.ProcessPoolExecutor(10)
		result = [executor.submit(group_transform, largeX, A, B, gradient, intercept) for largeX in largeX_list]
		concurrent.futures.wait(result)

		final = np.vstack([list(r.result()) for r in list(result)])
	print("FINAL SHAPE", final.shape)

	final[final < 0.] = 0.
	final[final > 1.] = 1.
	print("Final vector", final)

	return final

##################################################################################################
def predict(input, model):

	batch_size = 300000
	prediction = model.predict(input, verbose=1, batch_size=batch_size)

	return prediction



##################################################################################################
def compute(input_vectors, mat_model, temp_model):

	
	temperature = predict(input_vectors, temp_model.model)
	maturity = predict(input_vectors, mat_model.model)

	return temperature, maturity


##################################################################################################
def compute_history(input_vectors, history_model):

	history = predict(input_vectors, history_model.model)

	return history

##################################################################################################
def get_predictions(data, variable):

	if len(data.shape) != 3:
		return "Found shape " + str(data.shape) + " Please provide a data array with shape (16, ny, nx)."

	else:
		ny = data.shape[1]
		nx = data.shape[2]
		array_to_compute = data.reshape((data.shape[0], nx*ny))
		input_vectors = prepare_vector(transform_data, data_array = array_to_compute)
		models = {'temperature' : temp_model.model, 'maturity': mat_model.model, 'maturity_history' : history_model.model}

		if variable in models:
			prediction = predict(input_vectors, models[variable])
			return prediction.transpose().reshape((prediction.shape[1], ny, nx))

		else:
			return "Unsupported variable type. Please use temperature, maturity or maturity_history"
	

##################################################################################################
def load_GOM_data(path):
	mat_model = NN(path + 'trained_model_EzRo.h5', 'EasyRo', 'EzRo')
	history_model = NN(path + 'nn_mat_history.h5', 'EasyRo', 'EzRo')
	temp_model = NN(path + 'trained_model_Temperature.h5', 'Temperature', 'C')
	transform_data = np.load(path + 'transform.npz')
	
	project_affine = np.load(path + 'project_affine.npy')

	d = np.load(path + 'mapstack.npz')
	data_array = d['mapstack']


	data_array[data_array<-9000] = np.nan

	for i in range(5):
		thickness = data_array[i+1] - data_array[i]
		thickness[thickness < 1] = 10.0
		data_array[i+1] = data_array[i] + thickness
	
	f = 'data/STS_ezRo.csv'
	ro_sts = pd.read_csv(f, sep=';')

	depths = data_array[:6]
	lithos = data_array[6:11]
	
	return transform_data, mat_model, temp_model, history_model, project_affine, data_array, ro_sts, depths, lithos

##################################################################################################
def get_predictions(data, variable):
	'''
	Compute the present day physical state (temperature or maturity) of a geological column.
	Computation is performed with a neural network trained in the context of the Gulf of Mexico.
	Results are identical to those that one could obtain with a 1D full physics basin simulator.

	Parameters:
	- data (Numpy array) : the geological column, with shape (34, 1, 1)
	- varialbe (str) :  the desired output, 'temperature' or 'maturity'

	Returns : a Numpy array with the prediction

	Check the Readme for details on the underlying hypotheses.
	'''

	if len(data.shape) != 3:
		return f"Found shape {data.shape} Please provide a data array with shape (34, ny, nx)."

	else:
		ny = data.shape[1]
		nx = data.shape[2]
		array_to_compute = data.reshape((data.shape[0], nx*ny))
		input_vectors = prepare_vector(transform_data, data_array = array_to_compute)
		models = {'temperature' : temp_model.model, 'maturity': mat_model.model}

		if variable in models:
			prediction = predict(input_vectors, models[variable])
			return prediction.transpose().reshape((prediction.shape[1], ny, nx))

		else:
			return "Unsupported variable type. Please use temperature or maturity"

##################################################################################################
def st_ui():
	st.set_page_config(layout = "wide")

	# Load NN models and transform

	layers_dict = {0: "Sea Bottom",
						1: "Top Mio./Plioc.",
						2: "Top Miocene",
						3: "Top Paleogene",
						4: "Top Cretaceous",
						5: "Top mid-Cretaceous",
						6: "Top Callovian Salt"
						}

	layers_wrap = {0: [0],
					1: [1, 2, 3],
					2: [4, 5, 6],
					3: [7, 8],
					4: [9, 10],
					5: [11, 12, 13],
					6: [14]
	}

	st.title('Interactive 1D model (GOM setting with salt)')

	geol_column = [0]
	present_day_temperature = st.sidebar.slider("Present Day SWIT (C)", 3, 25, 5)
	# present_day_temperature = 4.0

	with st.sidebar.header("Geometry"):
		thick_layer1 = st.sidebar.slider("Thickness Plio-Pleistocene (m)", 100, 5000, 1000)
		thick_layer2 = st.sidebar.slider("Thickness Allochtonous Salt (m)", 100, 5000, 2000)
		thick_layer3 = st.sidebar.slider("Thickness Miocene (m)", 100, 5000, 1000)
		thick_layer4 = st.sidebar.slider("Thickness Paleogene (m)", 100, 5000, 1000)
		thick_layer5 = st.sidebar.slider("Thickness Late Cretaceous (m)", 100, 5000, 1000)
		thick_layer6 = st.sidebar.slider("Thickness Upper Jurassic to Mid Cretaceous (m)", 100, 5000, 1000)
		thick_layer7 = st.sidebar.slider("Thickness Callovian Salt (m)", 100, 5000, 1000)
	
	for ix, t in enumerate([thick_layer1, thick_layer2, thick_layer3, thick_layer4, thick_layer5, thick_layer6, thick_layer7]):
		top = geol_column[-1]
		bottom = top + t
		intermediate = np.linspace(top, bottom, num=len(layers_wrap[ix])+1)
		for k in range(1,intermediate.shape[0]):
			geol_column.append(intermediate[k])

	with st.sidebar.header("Lithology ratios"):
		lithoratio_layer2 = st.sidebar.slider("Salt vs 80mdst20sst Layer 2 - Alloch. Salt or Mio./Plioc.", 0., 1., 1.0)
		lithoratio_layer3 = st.sidebar.slider("Salt vs 80mdst20sst Layer 3 - Miocene", 0., 1., 0.)
		lithoratio_layer4 = st.sidebar.slider("Salt vs 50mdst50sst Layer 4 - Paleogene", 0., 1., 0.)
		lithoratio_layer5 = st.sidebar.slider("Salt vs 100lst Layer 5 - Late Cretaceous", 0., 1., 0.)
		lithoratio_layer6 = st.sidebar.slider("Salt vs 100lst Layer 6 - Upper Jurassic to Mid Cretaceous", 0., 1., 0.)

	for ix, l in enumerate([lithoratio_layer2, lithoratio_layer3, lithoratio_layer4, lithoratio_layer5, lithoratio_layer6]):
		for k in range(len(layers_wrap[ix + 1])):
			geol_column.append(l)

	st.sidebar.header("Basement parameters")

	crust_dict = dict()
	crust_dict["Free selection"] = {"Crust Thickness" : [10, 40, 15], 
							   "Upper Crust RHP" : [0., 5., 1.5],
							   "Lower Crust RHP" : [0., 2., 0.2],
							   "Mantle Thickness": [50, 120, 70]}
	crust_dict["Oceanic Crust"] = {"Crust Thickness" : [8, 15, 11], 
							   "Upper Crust RHP" : [0., 0.1, 0.],
							   "Lower Crust RHP" : [0., 0.1, 0.],
							   "Mantle Thickness": [50, 70, 60]}
	crust_dict["Transitional Crust"] = {"Crust Thickness" : [10, 20, 15], 
							   "Upper Crust RHP" : [0., 2., 1.5],
							   "Lower Crust RHP" : [0., 0.5, 0.2],
							   "Mantle Thickness": [50, 90, 70]}
	crust_dict["Continental Crust"] = {"Crust Thickness" : [18, 40, 25], 
							   "Upper Crust RHP" : [2., 5., 3.],
							   "Lower Crust RHP" : [0., 2., 1.],
							   "Mantle Thickness": [80, 120, 100]}

	crust_type = st.sidebar.selectbox("Crust type selection", ["Free selection", "Oceanic Crust", "Transitional Crust", "Continental Crust"])
	mmd = crust_dict[crust_type]["Crust Thickness"]
	crust_thickness = st.sidebar.slider("Crust Thickness (Top Basement to Moho) (km)", mmd[0], mmd[1], mmd[2])
	crust_thickness *= 1000
	mmd = crust_dict[crust_type]["Upper Crust RHP"]
	uc_rhp = st.sidebar.slider("Upper Crust RHP (uW/m3)", mmd[0], mmd[1], mmd[2])
	uc_rhp *= 1.0e-6
	mmd = crust_dict[crust_type]["Lower Crust RHP"]
	lc_rhp = st.sidebar.slider("Lower Crust RHP (uW/m3)", mmd[0], mmd[1], mmd[2])
	lc_rhp *= 1.0e-6
	mmd = crust_dict[crust_type]["Mantle Thickness"]
	mantle_thickness = st.sidebar.slider("Upper Mantle Thickness (Moho to Astenosphere) (km)", mmd[0], mmd[1], mmd[2])
	mantle_thickness *= 1000
	
	geol_column.append(crust_thickness)
	geol_column.append(lc_rhp)
	geol_column.append(mantle_thickness)
	geol_column.append(uc_rhp)
	geol_column.append(present_day_temperature)

	print("geol", geol_column, len(geol_column))
	geol_column = np.array(geol_column).reshape((34,1,1))
	
	st.sidebar.header("Graphical options")

	
	onset_oil_window = st.sidebar.slider("Onset oil window (%Ro)", 0.4, 0.7, 0.55)
	floor_oil_window = st.sidebar.slider("Floor oil window (%Ro)", 1.1, 1.3, 1.2)

	max_depth = float(st.sidebar.text_input("Maximum depth (m)", 10000))
	max_temperature = float(st.sidebar.text_input("Maximum temperature (C)", 400))
	max_Ro = float(st.sidebar.text_input("Maximum Ro (%VRo eq.)", 4))

	temperature = get_predictions(data = geol_column, variable = 'temperature')
	maturity = get_predictions(data = geol_column, variable = 'maturity')

	temperature = temperature[:,0,0].flatten()
	print("Temperature", temperature)
	maturity = maturity[:,0,0].flatten()
	temperature = np.insert(temperature, 0, present_day_temperature)
	maturity = np.insert(maturity, 0, 0.2045)
	sts = np.interp(maturity/100, ro_sts['ezRo'], ro_sts['sts'])
	# print(ro_sts['ezRo'])


	depths = geol_column[:16].flatten()
	mid_points = 0.5 * (depths[1:] + depths[:-1])
	mid_points = np.insert(mid_points, 0, 0)
	
	top_oil_window = np.interp(onset_oil_window, maturity, mid_points)
	lim1 = np.interp(0.8, maturity, mid_points)
	lim2 = np.interp(1.0, maturity, mid_points)
	bottom_oil_window = np.interp(floor_oil_window, maturity, mid_points)
	lim3 = np.interp(1.5, maturity, mid_points)
	lim4 = np.interp(2.0, maturity, mid_points)
	lim5 = np.interp(4.0, maturity, mid_points)

	# print(temperature)
	# data = np.vstack((temperature, mid_points))

	with st.expander("Summary"):
		st.markdown(summary(geol_column.flatten()))

	user_data = st.file_uploader("Load your calibration data ! (Excel format, 3 columns : [depth, temperature, easyRo])")
	data = None
	if user_data is not None:
		xls = pd.read_excel(user_data)
		data = xls.values
	salt_color = "#FF39E7"
	colors = ['#FFECA6', '#FFF102', '#FFF102', '#EEA26D', '#A7CD6B', '#ABDBEC', '#C32E92']
	for i, l in enumerate([lithoratio_layer2, lithoratio_layer3, lithoratio_layer4, lithoratio_layer5, lithoratio_layer6]):
		if l > 0.5:
			colors[i+1] = salt_color

	with _lock:
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,12))
		ax1.plot(temperature, mid_points, 'o-', lw = 3)
		ax1.set_ylim([0,max_depth])

		ax1.invert_yaxis()
		ax1.set_xlabel('Temperature (C)')
		ax1.set_ylabel('Burial Depth (m)')
		ax1.set_xlim([0,max_temperature])
		if data is not None:
			ax1.plot(data[:,1], data[:,0], 'ko')

		ax1.grid()
		# markers = [([0,max_temperature],[depths[i], depths[i]]) for i in range(8)]
		markers = [([0,max_temperature],[depths[0], depths[0]])]
		idx=0
		for i in range(7):
			nb_marks = len(layers_wrap[i])
			idx += nb_marks
			markers.append(([0,max_temperature],[depths[idx], depths[idx]]))


		for ii,m in enumerate(markers):
			skip = False
			ax1.plot(m[0], m[1], 'k--', lw=0.8)
			x1 = m[0][0]
			y1 = m[1][0]
			x2 = m[0][1]
			y2 = m[1][1]
			if ii < len(markers) - 1:
				y3 = markers[ii+1][1][1]
				y4 = markers[ii+1][1][0]
			else:
				y3 = y2
				y4 = y1

			
			y = np.array([[x1,y1], [x2, y2], [x2, y3], [x1, y4]])
			if ii < 7:
				alpha = 0.5
				if colors[ii] == salt_color:
					alpha = 1
				p = Polygon(y, facecolor = colors[ii], alpha = alpha)
				ax1.add_patch(p)
				ax1.annotate(layers_dict[ii], (max_temperature, m[1][0] + 150), ha = 'right')
					

		ax2.plot(maturity, mid_points, 'o-', c='black')
		ax2.plot(sts/100, mid_points, 'o--', c='white')
		ax2.set_ylim([0,max_depth])

		if data is not None:
			ax2.plot(data[:,2], data[:,0], 'ko')

		ax2.invert_yaxis()
		ax2.set_xlabel('Easy Ro (%Ro eq.) and STS/100')
		ax2.set_ylabel('Burial Depth (m)')
		ax2.set_xlim([0,max_Ro])

		ax2.grid()

		# markers = [([0,max_Ro],[depths[i], depths[i]]) for i in range(6)]
		markers = [([0,max_Ro],[depths[0], depths[0]])]
		idx=0
		for i in range(7):
			nb_marks = len(layers_wrap[i])
			idx += nb_marks
			markers.append(([0,max_Ro],[depths[idx], depths[idx]]))
		
		for ii,m in enumerate(markers):
			skip = False
			ax2.plot(m[0], m[1], 'k--', lw=1.5)
			x1 = m[0][0]
			y1 = m[1][0]
			x2 = m[0][1]
			y2 = m[1][1]
			if ii < len(markers) - 1:
				y3 = markers[ii+1][1][1]
				y4 = markers[ii+1][1][0]
			else:
				y3 = y2
				y4 = y1

			
			y = np.array([[x1,y1], [x2, y2], [x2, y3], [x1, y4]])
			if ii < 7:
				p = Polygon(y, facecolor = '#A6A6A6', alpha = 0.5)
				ax2.add_patch(p)
			#Oil window display
			y = np.array([[0,top_oil_window], [max_Ro, top_oil_window], [max_Ro, lim1], [0, lim1]])
			p = Polygon(y, facecolor = '#336600', alpha = 0.5)
			ax2.add_patch(p)
			y = np.array([[0,lim1], [max_Ro, lim1], [max_Ro, lim2], [0, lim2]])
			p = Polygon(y, facecolor = '#66FF02', alpha = 0.5)
			ax2.add_patch(p)
			y = np.array([[0,lim2], [max_Ro, lim2], [max_Ro, bottom_oil_window], [0, bottom_oil_window]])
			p = Polygon(y, facecolor = '#FFFF99', alpha = 0.5)
			ax2.add_patch(p)
			y = np.array([[0,bottom_oil_window], [max_Ro, bottom_oil_window], [max_Ro, lim3], [0, lim3]])
			p = Polygon(y, facecolor = '#FFCC02', alpha = 0.5)
			ax2.add_patch(p)
			y = np.array([[0,lim3], [max_Ro, lim3], [max_Ro, lim4], [0, lim4]])
			p = Polygon(y, facecolor = '#FF6634', alpha = 0.5)
			ax2.add_patch(p)
			y = np.array([[0,lim4], [max_Ro, lim4], [max_Ro, lim5], [0, lim5]])
			p = Polygon(y, facecolor = '#FF0033', alpha = 0.5)
			ax2.add_patch(p)
			ax2.annotate('Top Oil Window', (max_Ro, top_oil_window - 25), ha = 'right')

		buf = BytesIO()
		fig.savefig(buf, format="png", bbox_inches='tight', transparent = True)
		st.image(buf, use_column_width=False, caption='Temperature and Easy Ro profile')

	


	




if __name__ == "__main__":
	st_ui()