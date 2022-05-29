# Interactive 1D basin model

Interactive 1D application using the Neural Network developed for the Gulf of Mexico.  
This is for educational purposes only and should not be used for decision making.  

You can also call it programatically:  

```python
import pydaisi as pyd
import numpy as np

gom_model = pyd.Daisi("Gulf of Mexico regional model")
One_dim_model = np.array([0,1000,2000,3000,4000,5000,0.5,0.5,0.5,0.5,0.5,15000,1e-6,5e-7,60000,10]).reshape((16,1,1))
result = gom_model.get_predictions(data = One_dim_model, variable = 'temperature').value.flatten()
```
