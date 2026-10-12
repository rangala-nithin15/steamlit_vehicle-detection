from collections import Counter
import cv2

VEHICLES={2,3,5,7}
def analyze_video(path, model_name='yolo11n.pt', confidence=.35, stride=3):
    from ultralytics import YOLO
    model=YOLO(model_name); cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError('Could not open video')
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0; counts=Counter(); events=[]; seen=set(); centers={}; frame_no=0; analyzed=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if frame_no%max(1,stride): frame_no+=1; continue
        analyzed+=1; result=model.track(frame,persist=True,tracker='bytetrack.yaml',conf=confidence,verbose=False)[0]
        people=[]; vehicles=[]
        if result.boxes is not None and result.boxes.id is not None:
            for box,tid,cls in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.id.int().cpu().tolist(),result.boxes.cls.int().cpu().tolist()):
                x1,y1,x2,y2=map(float,box); item=(int(tid),(x1,y1,x2,y2),((x1+x2)/2,(y1+y2)/2))
                if cls==0: people.append(item); counts[int(tid)]+=1
                elif cls in VEHICLES: vehicles.append(item)
        t=frame_no/fps
        for vid,vb,vc in vehicles:
            speed=((vc[0]-centers[vid][0])**2+(vc[1]-centers[vid][1])**2)**.5 if vid in centers else 0; centers[vid]=vc
            close=[]; scale=max(30,((vb[2]-vb[0])**2+(vb[3]-vb[1])**2)**.5)
            for pid,pb,pc in people:
                if ((pc[0]-vc[0])**2+(pc[1]-vc[1])**2)**.5/scale<1.5: close.append((pid,pb))
            if close:
                pid,pb=close[0]; key=('near',vid,pid,int(t//5))
                if key not in seen: seen.add(key); events.append(dict(time=t,type='person_near_vehicle',risk=1,person_id=pid,details=f'Person track {pid} is near vehicle {vid}.'))
                overlap=max(0,min(pb[2],vb[2])-max(pb[0],vb[0]))*max(0,min(pb[3],vb[3])-max(pb[1],vb[1])); area=max(1,(pb[2]-pb[0])*(pb[3]-pb[1]))
                if overlap/area>.12:
                    key=('entry',vid,pid,int(t//4))
                    if key not in seen: seen.add(key); events.append(dict(time=t,type='possible_entry_or_occlusion',risk=2,person_id=pid,details=f'Person {pid} overlaps the vehicle region; review the frame.'))
            elif speed>18:
                key=('move',vid,int(t//3))
                if key not in seen: seen.add(key); events.append(dict(time=t,type='moving_vehicle_no_nearby_person',risk=3,person_id=None,details=f'Vehicle {vid} moved with no nearby person detected.'))
        frame_no+=1
    cap.release(); events.sort(key=lambda e:e['time'])
    return dict(frames_analyzed=analyzed,person_tracks=dict(counts),events=events)


def analyze_and_annotate_video(path, output_path, model_name='yolo11n.pt', confidence=.35, stride=2, progress_callback=None, owner_person_id=None, owner_vehicle_id=None):
    """Track a sampled frame sequence, render every source frame, and return risk events."""
    from ultralytics import YOLO
    from collections import Counter
    import math
    model=YOLO(model_name)
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError('Could not open input video')
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width<=0 or height<=0: cap.release(); raise ValueError('Could not read input video dimensions')
    writer=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*'avc1'),fps,(width,height))
    if not writer.isOpened():
        writer.release()
        writer=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened(): cap.release(); raise ValueError('Could not create output MP4; this OpenCV installation lacks a compatible video encoder')
    counts=Counter(); vehicle_counts=Counter(); events=[]; seen=set(); centers={}; frame_no=0; peak_risk=0; last_people=[]; last_vehicles=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if frame_no % max(1,stride)==0:
                result=model.track(frame,persist=True,tracker='bytetrack.yaml',conf=confidence,verbose=False)[0]
                people=[]; vehicles=[]
                if result.boxes is not None and result.boxes.id is not None:
                    boxes=result.boxes.xyxy.cpu().numpy(); ids=result.boxes.id.int().cpu().tolist(); classes=result.boxes.cls.int().cpu().tolist()
                    for box,tid,cls in zip(boxes,ids,classes):
                        x1,y1,x2,y2=map(int,box); tid=int(tid); cx=(x1+x2)//2; cy=(y1+y2)//2
                        item={'id':tid,'box':(x1,y1,x2,y2),'center':(cx,cy),'cls':int(cls)}
                        if cls==0:
                            people.append(item); counts[tid]+=1
                        elif cls in VEHICLES:
                            vehicles.append(item); vehicle_counts[tid]+=1
                last_people=people; last_vehicles=vehicles
            else:
                people=last_people; vehicles=last_vehicles
            t=frame_no/fps; frame_risk=0; links=[]; person_states={}; vehicle_states={}
            for v in vehicles:
                vid=v['id']; vb=v['box']; vc=v['center']; x1,y1,x2,y2=vb
                speed=math.hypot(vc[0]-centers[vid][0],vc[1]-centers[vid][1]) if vid in centers else 0
                centers[vid]=vc
                scale=max(30,math.hypot(x2-x1,y2-y1)); near=[]
                for person in people:
                    pc=person['center']; distance=math.hypot(pc[0]-vc[0],pc[1]-vc[1])/scale
                    if distance<1.5: near.append((person,distance))
                if near:
                    registered_vehicle=(owner_person_id is not None and owner_vehicle_id is not None and vid==owner_vehicle_id)
                    people_to_check=near if registered_vehicle else [min(near,key=lambda x:x[1])]
                    for person,_ in people_to_check:
                        pid=person['id']; pb=person['box']
                        overlap=max(0,min(pb[2],vb[2])-max(pb[0],vb[0]))*max(0,min(pb[3],vb[3])-max(pb[1],vb[1])); area=max(1,(pb[2]-pb[0])*(pb[3]-pb[1]))
                        if registered_vehicle and pid==owner_person_id:
                            links.append((person,v,'owner_match')); person_states[pid]='owner_match'; vehicle_states[vid]='owner_match'
                            continue
                        if registered_vehicle:
                            links.append((person,v,'unknown')); person_states[pid]='unknown'; vehicle_states[vid]='unknown'
                            level=3 if overlap/area>.12 else 2
                            frame_risk=max(frame_risk,level); key=('unmatched',vid,pid,int(t//4))
                            if key not in seen:
                                seen.add(key); events.append({'time':t,'type':'unmatched_person_at_registered_vehicle','risk':level,'person_id':pid,'vehicle_id':vid,'details':f'Person track {pid} does not match the manually registered owner track {owner_person_id} for vehicle {vid}.'})
                        else:
                            links.append((person,v,'approach'))
                            key=('near',vid,pid,int(t//5))
                            if key not in seen:
                                seen.add(key); events.append({'time':t,'type':'person_near_vehicle','risk':1,'person_id':pid,'vehicle_id':vid,'details':f'Person track {pid} is near vehicle track {vid}.'})
                            frame_risk=max(frame_risk,1)
                            if overlap/area>.12:
                                frame_risk=max(frame_risk,2); key=('entry',vid,pid,int(t//4))
                                if key not in seen:
                                    seen.add(key); events.append({'time':t,'type':'possible_entry_or_occlusion','risk':2,'person_id':pid,'vehicle_id':vid,'details':f'Person {pid} overlaps vehicle {vid}; review the clip.'})
                elif speed>12:
                    frame_risk=max(frame_risk,3); key=('move',vid,int(t//3))
                    if key not in seen:
                        seen.add(key); events.append({'time':t,'type':'moving_vehicle_no_nearby_person','risk':3,'person_id':None,'vehicle_id':vid,'details':f'Vehicle {vid} moved without a nearby detected person.'})
            peak_risk=max(peak_risk,frame_risk)
            # Draw track boxes and readable per-clip IDs.
            for item in people:
                x1,y1,x2,y2=item['box']; tid=item['id']; state=person_states.get(tid,'normal')
                color=(0,220,0) if state=='owner_match' else ((0,0,255) if state=='unknown' else (255,120,30))
                prefix='OWNER PERSON' if state=='owner_match' else ('DETECTED PERSON' if state=='unknown' else 'PERSON')
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,3 if state!='normal' else 2)
                label=f'{prefix} #{tid}'
                (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.58,2)
                yy=max(22,y1-7); cv2.rectangle(frame,(x1,yy-th-7),(x1+tw+7,yy+3),color,-1)
                cv2.putText(frame,label,(x1+3,yy-2),cv2.FONT_HERSHEY_SIMPLEX,.58,(10,10,10),2,cv2.LINE_AA)
            for item in vehicles:
                x1,y1,x2,y2=item['box']; tid=item['id']; state=vehicle_states.get(tid,'normal')
                color=(0,220,0) if state in ('owner_match','normal') else (0,0,255)
                prefix='OWNER VEHICLE' if state=='owner_match' else ('DETECTED VEHICLE' if state=='unknown' else 'VEHICLE')
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,3 if state=='unknown' else 2)
                label=f'{prefix} #{tid}'
                (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.58,2)
                yy=max(22,y1-7); cv2.rectangle(frame,(x1,yy-th-7),(x1+tw+7,yy+3),color,-1)
                cv2.putText(frame,label,(x1+3,yy-2),cv2.FONT_HERSHEY_SIMPLEX,.58,(10,10,10),2,cv2.LINE_AA)
            for person,vehicle,state in links:
                color=(0,220,0) if state=='owner_match' else ((0,0,255) if state=='unknown' else (0,190,255))
                cv2.line(frame,person['center'],vehicle['center'],color,3 if state!='approach' else 2)
                mx=(person['center'][0]+vehicle['center'][0])//2; my=(person['center'][1]+vehicle['center'][1])//2
                text=f'OWNER MATCH P{person["id"]} -> V{vehicle["id"]}' if state=='owner_match' else (f'DETECTED P{person["id"]} -> V{vehicle["id"]}' if state=='unknown' else f'P{person["id"]} -> V{vehicle["id"]}')
                cv2.putText(frame,text,(mx,my),cv2.FONT_HERSHEY_SIMPLEX,.48,color,2,cv2.LINE_AA)
            # Status overlay similar to the requested reference.
            hud=(frame.copy())
            cv2.rectangle(hud,(0,0),(min(width,560),76),(0,0,0),-1); cv2.addWeighted(hud,.58,frame,.42,0,frame)
            cv2.putText(frame,f'Frame: {frame_no+1}/{total or "?"} | People: {len(people)} | Vehicles: {len(vehicles)}',(10,27),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,255,255),2,cv2.LINE_AA)
            risk_color=(70,230,80) if frame_risk<2 else ((0,200,255) if frame_risk==2 else (0,0,255))
            cv2.putText(frame,f'Risk: {frame_risk}/3 | Peak risk: {peak_risk}/3 | Events: {len(events)}',(10,57),cv2.FONT_HERSHEY_SIMPLEX,.62,risk_color,2,cv2.LINE_AA)
            writer.write(frame); frame_no+=1
            if progress_callback and (frame_no%15==0 or (total and frame_no==total)):
                progress_callback(min(1.0,frame_no/total) if total else 0.0,frame_no,total)
    finally:
        cap.release(); writer.release()
    events.sort(key=lambda e:e['time'])
    if frame_no==0: raise ValueError('The uploaded video contains no readable frames')
    return {'frames_analyzed':frame_no,'person_tracks':dict(counts),'vehicle_tracks':dict(vehicle_counts),'events':events,'annotated_path':str(output_path),'peak_risk':peak_risk}
