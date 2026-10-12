from pathlib import Path
import tempfile
import json
import uuid
import streamlit as st
from theft_detection.classifier import predict_one
from theft_detection.analyzer import analyze_and_annotate_video

st.set_page_config(page_title='Vehicle Theft Monitor',page_icon='🚘',layout='wide')
st.title('Vehicle Theft Monitor')
st.caption('Upload a surveillance clip for person/vehicle tracking and reviewable risk events.')
with st.expander('How training and results work'):
    st.markdown('**Prepare labeled clips:** put at least 10 reviewed videos in each folder: `data/videos/normal/` and `data/videos/detected/`. Then run `python train.py` in the terminal. Training prints validation accuracy and saves held-out test accuracy, precision, recall, F1, specificity, confusion matrix, and per-class scores. After training finishes, these metrics appear below the timeline after you analyze an upload.')
    st.markdown('**This page analyzes one uploaded video.** It shows an annotated MP4 with person/vehicle bounding boxes and per-clip tracking IDs, plus NORMAL/DETECTED, peak heuristic risk (0–3), and a timestamped event list. It does not calculate accuracy/precision/recall for one upload because its true label is unknown. If `models/theft_knn.json` exists, the trained clip classifier supplies the label; otherwise the page uses the risk heuristic (DETECTED when peak risk is 2 or 3).')
st.warning('Track association is not proof of ownership or identity. Review every alert; this prototype does not send notifications.')
with st.sidebar:
    model=st.text_input('YOLO model','yolo11n.pt')
    confidence=st.slider('Detection confidence',.1,.9,.35,.05)
    stride=st.slider('Analyze every Nth frame',1,10,3)
    owner_person_id=st.number_input('Registered owner person ID',min_value=0,value=0,step=1,help='Analyze once to see track IDs, then enter the owner person ID and analyze again. Use 0 if not registered.')
    owner_vehicle_id=st.number_input('Registered owner vehicle ID',min_value=0,value=0,step=1,help='Enter the vehicle track ID that belongs to the registered owner person. Use 0 if not registered.')
    st.caption('Register the known owner person and their car as a pair. Matching pair is green; another person near that registered car is red.')
video=st.file_uploader('Choose a video',type=['mp4','mov','avi','mkv'])
if video:
    st.video(video)
    if st.button('Analyze video',type='primary'):
        with tempfile.NamedTemporaryFile(delete=False,suffix=Path(video.name).suffix or '.mp4') as f:
            f.write(video.getvalue()); path=f.name
        try:
            st.session_state.pop('report',None)
            with st.spinner('Detecting, tracking, and rendering video...'):
                output_path=Path(tempfile.gettempdir())/f'vehicle_theft_annotated_{uuid.uuid4().hex}.mp4'
                progress=st.progress(0,text='Starting video analysis...')
                def update_progress(ratio,current,total):
                    progress.progress(ratio,text=f'Tracking frame {current}/{total} and rendering annotated video...')
                st.session_state.report=analyze_and_annotate_video(path,str(output_path),model,confidence,stride,update_progress,owner_person_id=int(owner_person_id) or None,owner_vehicle_id=int(owner_vehicle_id) or None)
        except Exception as e: st.error(f'Analysis failed: {e}. Install requirements; first model run may require internet to download weights.')
        finally:
            Path(path).unlink(missing_ok=True)
            if 'progress' in locals(): progress.empty()
if 'report' in st.session_state:
    r=st.session_state.report; events=r['events']; score=max([e['risk'] for e in events],default=0)
    label='DETECTED' if score>=2 else 'NORMAL'
    model_path=Path('models/theft_knn.json')
    trained_evaluation=None
    if model_path.exists():
        try:
            pack=json.loads(model_path.read_text(encoding='utf-8')); trained_evaluation=pack.get('evaluation'); risks=[e['risk'] for e in events]; n=max(1,r['frames_analyzed'])
            vec=[[len(r['person_tracks']),len(events)/n,sum(v==1 for v in risks)/n,sum(v==2 for v in risks)/n,sum(v==3 for v in risks)/n,score]]
            label=predict_one(vec[0],pack).upper()
        except Exception as e: st.warning(f'Trained model could not be loaded; showing heuristic result: {e}')
    if any(e['type']=='unmatched_person_at_registered_vehicle' for e in events):
        label='DETECTED'
    a,b,c=st.columns(3); a.metric('Classification',label); b.metric('Peak risk',f'{score}/3'); c.metric('Frames analyzed',r['frames_analyzed'])
    if r['person_tracks']:
        st.write('Person track IDs and observed frame counts:',r['person_tracks'])
        st.info('Track IDs are temporary to this clip. Review the video to decide which track is the owner; the prototype cannot verify ownership.')
    if r.get('vehicle_tracks'):
        st.write('Vehicle track IDs and observed frames:',r['vehicle_tracks'])
    if owner_person_id and owner_vehicle_id:
        st.info(f'Registered owner pair: Person #{int(owner_person_id)} ↔ Vehicle #{int(owner_vehicle_id)}. A different person track near this car is highlighted red as DETECTED.')
    annotated=Path(r.get('annotated_path',''))
    if annotated.is_file():
        st.subheader('Annotated tracking video')
        st.caption('Boxes and Person/Vehicle IDs are per-video tracker IDs. IDs are not identity verification across separate videos.')
        annotated_bytes=annotated.read_bytes()
        st.video(annotated_bytes,format='video/mp4')
        st.download_button('Download annotated video',data=annotated_bytes,file_name='vehicle_theft_annotated.mp4',mime='video/mp4')
    st.subheader('Timeline')
    if not events: st.success('No suspicious interaction events detected in sampled frames.')
    for e in events: st.write(f"**{e['time']:.1f}s · Risk {e['risk']}/3 · {e['type'].replace('_',' ').title()}** — {e['details']}")
    if trained_evaluation:
        ev=trained_evaluation['metrics']
        st.subheader('Model evaluation metrics (held-out test set)')
        m1,m2,m3,m4,m5=st.columns(5)
        m1.metric('Accuracy',f"{ev['accuracy']:.1%}")
        m2.metric('Precision · detected',f"{ev['precision_detected']:.1%}")
        m3.metric('Recall · detected',f"{ev['recall_detected']:.1%}")
        m4.metric('F1 · detected',f"{ev['f1_detected']:.1%}")
        m5.metric('Specificity · normal',f"{ev['specificity_normal']:.1%}")
        st.caption(f"Computed from {ev['test_count']} labeled test clips during training. These describe the held-out test set, not this uploaded clip.")
        cm=ev['confusion']
        st.write('Confusion matrix (rows = actual, columns = predicted)')
        st.table([{'Actual':'NORMAL','Predicted NORMAL':cm['true_normal_pred_normal'],'Predicted DETECTED':cm['true_normal_pred_detected']},{'Actual':'DETECTED','Predicted NORMAL':cm['true_detected_pred_normal'],'Predicted DETECTED':cm['true_detected_pred_detected']}])
        st.write('Per-class scores')
        st.table([{'Class':name.upper(),'Precision':f"{values['precision']:.1%}",'Recall':f"{values['recall']:.1%}",'F1':f"{values['f1']:.1%}",'Test clips':values['support']} for name,values in ev['per_class'].items()])
    else:
        st.info('Training metrics will appear here after you add labeled videos and run python train.py. Metrics are not available from a single unlabeled upload.')
    st.caption('Risk 0 routine · 1 proximity · 2 possible entry/contact · 3 vehicle movement without nearby person detected. These are heuristic review prompts, not determinations of theft.')
