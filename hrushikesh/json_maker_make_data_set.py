import csv
import json
import pdb

with open('hrushikesh_patient_data.json', 'r') as f_json:
    patients_data = json.load(f_json)

# pid -> sensor reading
# pid -> X
# pid -> Y

# patients_data[PtID] = {'SensorGLU_list':[ float(SensorGLU) ]}

#patient_data[PtID]['X'] -> X
#patient_data[PtID]['Y'] -> Y
#patient_data_set[]
m = 6
T = 20
for PtID,_  in patients_data.iteritems():
    current_SensorGLU_list = patients_data[PtID]['SensorGLU_list'] # sugar_list = [..,s,..]
    patients_data[PtID]['X'] = []
    patients_data[PtID]['Y'] = []
    for i,sugar in enumerate(current_SensorGLU_list):
        if i+m+T+1 > len(current_SensorGLU_list) - 1:
            #patient_data[PtID]['X'] = np.array( patient_data[PtID]['X'] )
            #patient_data[PtID]['Y'] = np.array( patient_data[PtID]['Y'] )
            break
        else:
            x = current_SensorGLU_list[i:i+m] # vector in R^(m)
            y_1 = current_SensorGLU_list[i+m+T]
            y_2 = (current_SensorGLU_list[i+m+T+1] - current_SensorGLU_list[i+m+T-1]) /10
            y = [y_1, y_2]
            patients_data[PtID]['X'].append(x)
            patients_data[PtID]['Y'].append(y)

with open('patient_data_X_Y.json', 'w+') as f_json:
    json.dump(patients_data,f_json,sort_keys=True, indent=2, separators=(',', ': '))
