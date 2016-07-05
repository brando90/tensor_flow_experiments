import csv
import json

data = {'cvs_data':[],'RecID':[],'PtID':[],'ReadingDt':[],'ReadingTm':[],'MeterBG':[],'SensorGLU':[]}
with open('hrushikesh.csv','rU') as csvfile:
    cvsreader = csv.reader(csvfile, delimiter=',')
    for row in cvsreader:
        data['cvs_data'].append(row)
        (RecID,PtID,ReadingDt,ReadingTm,MeterBG,SensorGLU) = row
        data['RecID'].append(RecID)
        data['PtID'].append(PtID)
        data['ReadingDt'].append(ReadingDt)
        data['ReadingTm'].append(ReadingTm)
        data['MeterBG'].append(MeterBG)
        data['SensorGLU'].append(SensorGLU)


with open('hrushikesh_data.json', 'w+') as f_json:
    json.dump(data,f_json,sort_keys=True, indent=2, separators=(',', ': '))
