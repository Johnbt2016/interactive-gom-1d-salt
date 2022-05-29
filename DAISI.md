# Interactive 1D basin model

Interactive 1D application using a Neural Network trained in the Gulf of Mexico setting.  
This is for educational purposes only and should not be used for decision making.  

You can also call it programatically:  

```python
import pydaisi as pyd
import numpy as np

gom_model = pyd.Daisi("laiglejm/Interactive GOM 1D Salt")
One_dim_model = np.array([0,1000,1666,2333,3000,3333,3666,4000,4500,5000,5500,6000,6333,6666,7000,8000,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,15000,2.00e-07,70000,1.50e-06,5]).reshape((34,1,1))
result = gom_model.get_predictions(data = One_dim_model, variable = 'temperature').value.flatten()
```

* dims 0 to 15 = depths
* dims 16 to 29 = lithos ratios
* dim 30 = Crust Thickness
* dim 31 = Lower Crust RHP
* dim 32 = Upper Mantle thickness
* dim 33 = Upper Crust RHP
* dim 34 = Present day surface temperature
