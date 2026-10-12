import math

def predict_one(vector, artifact):
    scales=artifact['scales']; q=[float(v)/s for v,s in zip(vector,scales)]
    rows=artifact['samples']; k=min(int(artifact.get('k',3)),len(rows))
    nearest=sorted(rows,key=lambda row:sum((a-b)**2 for a,b in zip(q,row['x'])))[:k]
    votes={name:0 for name in ('normal','detected')}
    for row in nearest: votes[row['label']]+=1
    return 'detected' if votes['detected']>votes['normal'] else 'normal'

def normalize_rows(rows, scales):
    return [[float(v)/s for v,s in zip(row,scales)] for row in rows]
