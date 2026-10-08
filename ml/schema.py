FEATURES = ['air_temperature','process_temperature','rotational_speed','torque','tool_wear','product_type']
NUMERIC = FEATURES[:-1]
TARGET = 'machine_failure'
# UCI AI4I original labels. These are NOT inputs to the model.
UCI_MAPPING = {'Air temperature [K]':'air_temperature','Process temperature [K]':'process_temperature','Rotational speed [rpm]':'rotational_speed','Torque [Nm]':'torque','Tool wear [min]':'tool_wear','Type':'product_type','Machine failure':'machine_failure'}
