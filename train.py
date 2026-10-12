from pathlib import Path
import argparse, json, random
from theft_detection.analyzer import analyze_video
from theft_detection.classifier import normalize_rows, predict_one

EXTS={'.mp4','.mov','.avi','.mkv'}
FEATURE_NAMES=['person_tracks','events_per_frame','risk1_per_frame','risk2_per_frame','risk3_per_frame','max_risk']
def features(path):
    report=analyze_video(str(path),stride=5)
    events=report['events']; n=max(1,report['frames_analyzed']); risks=[e['risk'] for e in events]
    return [len(report['person_tracks']),len(events)/n,sum(r==1 for r in risks)/n,sum(r==2 for r in risks)/n,sum(r==3 for r in risks)/n,max(risks,default=0)]
def split(paths, seed):
    rng=random.Random(seed); parts={'train':[],'validation':[],'test':[]}
    for label in ('normal','detected'):
        items=list(paths[label]); rng.shuffle(items); n=len(items)
        ntrain=int(n*.70); nval=int(n*.15)
        parts['train'] += [(p,label) for p in items[:ntrain]]
        parts['validation'] += [(p,label) for p in items[ntrain:ntrain+nval]]
        parts['test'] += [(p,label) for p in items[ntrain+nval:]]
    for part in parts.values(): rng.shuffle(part)
    return parts
def metrics(actual,pred):
    tp=sum(a=='detected' and p=='detected' for a,p in zip(actual,pred)); tn=sum(a=='normal' and p=='normal' for a,p in zip(actual,pred))
    fp=sum(a=='normal' and p=='detected' for a,p in zip(actual,pred)); fn=sum(a=='detected' and p=='normal' for a,p in zip(actual,pred))
    accuracy=(tp+tn)/max(1,len(actual)); precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn); f1=2*precision*recall/max(1e-12,precision+recall); specificity=tn/max(1,tn+fp)
    print(f'Accuracy:    {accuracy:.4f}'); print(f'Precision:   {precision:.4f} (detected)'); print(f'Recall:      {recall:.4f} (detected)'); print(f'F1 score:    {f1:.4f} (detected)'); print(f'Specificity: {specificity:.4f} (normal)')
    print('Confusion matrix (rows=true, columns=predicted; normal, detected):')
    print(f'                 predicted normal  predicted detected\ntrue normal              {tn:5d}             {fp:5d}\ntrue detected            {fn:5d}             {tp:5d}')
    per_class={}
    for cls in ('normal','detected'):
        ctp=sum(a==cls and p==cls for a,p in zip(actual,pred)); cfp=sum(a!=cls and p==cls for a,p in zip(actual,pred)); cfn=sum(a==cls and p!=cls for a,p in zip(actual,pred))
        pr=ctp/max(1,ctp+cfp); rc=ctp/max(1,ctp+cfn); cf=2*pr*rc/max(1e-12,pr+rc)
        per_class[cls]={'precision':pr,'recall':rc,'f1':cf,'support':actual.count(cls)}
        print(f'{cls:10s} precision={pr:.4f} recall={rc:.4f} f1={cf:.4f} support={actual.count(cls)}')
    return {'accuracy':accuracy,'precision_detected':precision,'recall_detected':recall,'f1_detected':f1,'specificity_normal':specificity,'confusion':{'true_normal_pred_normal':tn,'true_normal_pred_detected':fp,'true_detected_pred_normal':fn,'true_detected_pred_detected':tp},'per_class':per_class,'test_count':len(actual)}

def main():
    ap=argparse.ArgumentParser(description='Train a clip-level baseline from normal/ and detected/ folders')
    ap.add_argument('--data',default='data/videos'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--model',default='models/theft_knn.json'); args=ap.parse_args(); root=Path(args.data)
    paths={label:sorted(p for p in (root/label).glob('**/*') if p.suffix.lower() in EXTS) for label in ('normal','detected')}
    if min(map(len,paths.values()))<10:
        raise SystemExit(f"Not enough labeled videos. Found normal={len(paths['normal'])}, detected={len(paths['detected'])}. Add at least 10 video files to each folder: data/videos/normal/ and data/videos/detected/. Pexels parking clips can only be reviewed as normal candidates; they do not provide theft labels.")
    parts=split(paths,args.seed)
    print(f"Split sizes: train={len(parts['train'])} validation={len(parts['validation'])} test={len(parts['test'])} (70/15/15 approx)")
    vectors={}
    for name,items in parts.items():
        vectors[name]=[]
        for i,(p,label) in enumerate(items,1):
            print(f'Extracting {name} {i}/{len(items)}: {p}',flush=True)
            vectors[name].append((features(p),label))
    xtrain=[x for x,y in vectors['train']]; ytrain=[y for x,y in vectors['train']]
    scales=[max(1.0,max(abs(x[j]) for x in xtrain)) for j in range(len(FEATURE_NAMES))]
    nx=normalize_rows(xtrain,scales)
    best_k=1; best_score=-1
    for k in (1,3,5):
        artifact={'scales':scales,'samples':[{'x':x,'label':y} for x,y in zip(nx,ytrain)],'k':k}
        pred=[predict_one(x,artifact) for x,y in vectors['validation']]; actual=[y for x,y in vectors['validation']]
        score=sum(a==b for a,b in zip(actual,pred))/max(1,len(actual))
        if score>best_score: best_k,best_score=k,score
    print(f'Validation accuracy for selected k={best_k}: {best_score:.4f}')
    artifact={'scales':scales,'samples':[{'x':x,'label':y} for x,y in zip(nx,ytrain)],'k':best_k,'feature_names':FEATURE_NAMES}
    test=vectors['test']
    actual=[y for x,y in test]; pred=[predict_one(x,artifact) for x,y in test]
    print('\n=== HELD-OUT TEST METRICS ==='); evaluation=metrics(actual,pred)
    artifact['evaluation']={'split':'held-out test set','metrics':evaluation}
    Path(args.model).parent.mkdir(parents=True,exist_ok=True); Path(args.model).write_text(json.dumps(artifact),encoding='utf-8')
    print('Saved',args.model); print('For valid results, videos from the same camera/session must stay in one split; this starter splits individual files.')
if __name__=='__main__': main()
