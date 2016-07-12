import csv
import json

data = {'cvs_row':[],'RecID':[],'PtID':[],'ReadingDt':[],'ReadingTm':[],'MeterBG':[],'SensorGLU':[]}
patients_data = {}
# data[PtID]['X'].append(x)
# data[PtID]['Y'].append(y)
# X should be N by D_in
# Y should be N by D_out
with open('hrushikesh.csv','rU') as csvfile:
    cvsreader = csv.reader(csvfile, delimiter=',')
    current_PtID = None
    i = 0
    for row in cvsreader:
        if i > 1:
            data['cvs_row'].append(row)
            (RecID,PtID,ReadingDt,ReadingTm,MeterBG,SensorGLU) = row
            data['RecID'].append(RecID)
            data['PtID'].append(PtID)
            data['ReadingDt'].append(ReadingDt)
            data['ReadingTm'].append(ReadingTm)
            data['MeterBG'].append(MeterBG)
            data['SensorGLU'].append(SensorGLU)
            if current_PtID != PtID:
                current_PtID = PtID
                patients_data[PtID] = {'cvs_row_list':[ cvs_row ]}
                patients_data[PtID] = {'RecID_list':[ float(RecID) ]}
                patients_data[PtID] = {'PtID_list':[ PtID ]}
                patients_data[PtID] = {'ReadingDt_list':[ float(ReadingDt) ]}
                patients_data[PtID] = {'ReadingTm_list':[ float(ReadingTm) ]}
                patients_data[PtID] = {'MeterBG_list':[ float(MeterBG) ]}
                patients_data[PtID] = {'SensorGLU_list':[ float(SensorGLU) ]} #<-reading to make data
            else:
                patients_data[PtID]['SensorGLU_list'].append( float(SensorGLU) )
                patients_data[PtID]['cvs_row_list'].append( cvs_row ]
                patients_data[PtID]['RecID_list'].append( float(RecID) )
                patients_data[PtID]['PtID_list'].append(PtID)
                patients_data[PtID]['ReadingDt_list'].append( float(ReadingDt) )
                patients_data[PtID]['ReadingTm_list'].append( float(ReadingTm) )
                patients_data[PtID]['MeterBG_list'].append( float(MeterBG) )
                patients_data[PtID]['SensorGLU_list'].append( float(SensorGLU) ) #<-reading to make data
        i+=1

with open('hrushikesh_raw_data.json', 'w+') as f_json:
    json.dump(data,f_json,sort_keys=True, indent=2, separators=(',', ': '))
with open('hrushikesh_patient_data.json', 'w+') as f_json:
    json.dump(patients_data,f_json,sort_keys=True, indent=2, separators=(',', ': '))
