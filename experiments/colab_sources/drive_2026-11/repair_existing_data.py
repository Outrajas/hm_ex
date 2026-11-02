# ONE-CELL REINTERHAND REPAIR + FULL ACQUIRED-IMAGE AUDIT (NO TRAINING)
# Safe to rerun after a disconnect. Only three already-acquired captures are in scope.
import os, io, re, csv, json, time, gzip, shutil, hashlib, tarfile, threading, traceback
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
import cv2
from google.colab import drive

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/Persistent Visual-to-MANO Hand Tracker/04_DATASET_STAGING/ReInterHand')
assert (ROOT / '00_archives').is_dir(), 'Existing dataset folder is unavailable; check Drive mount.'
ARC, EXT = ROOT/'00_archives', ROOT/'01_extracted'
OUT = ROOT/'03_training_ready'
OUT.mkdir(exist_ok=True)
WORK = Path('/content/reinterhand_repair_work'); WORK.mkdir(exist_ok=True)
BASE = 'https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/ReInterHand'
CAPS = [
 ('BKS383','m--20220628--1327--BKS383--pilot--ProjectGoliath--ContinuousHandsy--two-hands',22),
 ('HIR112','m--20221007--1215--HIR112--pilot--ProjectGoliathScript--Hands--two-hands',3),
 ('202107','m--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands',38),
]
STEM='envmap_per_segment.tar.gz'
GIB=1024**3
MAX_DOWNLOAD=8*GIB # replacement/repair only; never acquire the remaining HIR capture
NEW_BYTES=0
START=time.monotonic()
STATUS={'phase':'starting','images':0}
HALT=threading.Event()
cv2.setNumThreads(1)

def say(s): print(time.strftime('%H:%M:%S'), s, flush=True)
def heartbeat():
    while not HALT.wait(60):
        say('HEARTBEAT '+json.dumps(STATUS)+'; elapsed=%.1f min; free local=%.1f GiB'%((time.monotonic()-START)/60,shutil.disk_usage('/content').free/GIB))
threading.Thread(target=heartbeat,daemon=True).start()

def retry(fn, attempts=5):
    for a in range(attempts):
        try: return fn()
        except (OSError, requests.RequestException) as e:
            if a==attempts-1: raise
            say('I/O retry %d/%d: %s'%(a+1,attempts,str(e)[:150])); time.sleep(min(20,2**a))

def digest(p, kind='sha256'):
    h=hashlib.new(kind)
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def atomic_bytes(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.repair_tmp')
    def put():
        with open(tmp,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    retry(put)
    if retry(path.read_bytes)!=data: raise IOError('Readback mismatch: '+str(path))

def save_json(path,obj): atomic_bytes(path,json.dumps(obj,indent=2,sort_keys=True).encode())

def persist_file(local,dest):
    expected=digest(local)
    dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.repair_tmp')
    retry(lambda:shutil.copyfile(local,tmp))
    if retry(lambda:digest(tmp))!=expected: raise IOError('Drive copy checksum mismatch: '+str(dest))
    retry(lambda:os.replace(tmp,dest))
    return expected

# Save precisely this executable cell for recovery, alongside its outputs.
try: atomic_bytes(OUT/'repair_existing_data.py',get_ipython().history_manager.input_hist_raw[-1].encode())
except NameError: pass

SESSION=requests.Session()
SESSION.headers.update({'Accept-Encoding':'identity'})
def download(url,dest,expected_md5=None,byte_range=None):
    global NEW_BYTES
    dest=Path(dest)
    for attempt in range(4):
        headers={}
        offset=dest.stat().st_size if dest.exists() and byte_range is None else 0
        if byte_range is not None: headers['Range']='bytes=%d-%d'%byte_range
        elif offset: headers['Range']='bytes=%d-'%offset
        try:
            with SESSION.get(url,headers=headers,stream=True,timeout=(30,90)) as r:
                r.raise_for_status()
                if byte_range is not None:
                    if r.status_code!=206 or not r.headers.get('Content-Range','').startswith('bytes %d-'%byte_range[0]):
                        raise RuntimeError('Server did not honor the bounded repair range')
                    mode='wb'
                elif offset and r.status_code==206:
                    if not r.headers.get('Content-Range','').startswith('bytes %d-'%offset): raise RuntimeError('Wrong resume range')
                    mode='ab'
                else: mode='wb'
                with open(dest,mode) as f:
                    for b in r.iter_content(4*1024**2):
                        if not b: continue
                        if NEW_BYTES+len(b)>MAX_DOWNLOAD: raise RuntimeError('Repair download budget exceeded; no bulk acquisition permitted')
                        f.write(b); NEW_BYTES+=len(b)
                    f.flush(); os.fsync(f.fileno())
            if byte_range is not None and dest.stat().st_size!=byte_range[1]-byte_range[0]+1:
                raise IOError('Short bounded range')
            if expected_md5 and digest(dest,'md5')!=expected_md5:
                dest.unlink(missing_ok=True) # only the private local staging copy
                raise IOError('Source MD5 mismatch; retry full replacement')
            return
        except (OSError,requests.RequestException) as e:
            if attempt==3: raise
            say('Download retry: '+str(e)[:150]); time.sleep(3*(attempt+1))

CHECKS={}; RECEIPTS={}
def metadata(cap):
    dest=OUT/cap/'metadata'; dest.mkdir(parents=True,exist_ok=True)
    for rel in ['CHECKSUM','frame_list.txt','Mugsy_cameras/cam_params.json']:
        local=WORK/(hashlib.sha1((cap+rel).encode()).hexdigest()+'.meta')
        download(BASE+'/'+cap+'/'+rel,local)
        atomic_bytes(dest/rel,local.read_bytes())
    checks={}
    for line in (dest/'CHECKSUM').read_text().splitlines():
        fields=line.split()
        md=[x for x in fields if re.fullmatch('[0-9a-fA-F]{32}',x)]
        if len(md)==1:
            names=[x.lstrip('*./') for x in fields if x!=md[0]]
            if names: checks[names[0]]=md[0].lower()
    if not checks: raise RuntimeError('Cannot parse official CHECKSUM')
    CHECKS[cap]=checks
    frames={}
    for line in (dest/'frame_list.txt').read_text().splitlines():
        if not line.strip(): continue
        seg,fid=line.split()[:2]; fid=int(fid)
        if fid in frames and frames[fid]!=seg: raise RuntimeError('Ambiguous temporal frame ID')
        frames[fid]=seg
    cams=json.loads((dest/'Mugsy_cameras/cam_params.json').read_text())
    for key,c in cams.items():
        R=np.asarray(c['R'],float); t=np.asarray(c['t'],float).reshape(3)
        assert R.shape==(3,3) and np.isfinite(R).all() and np.isfinite(t).all()
        assert np.allclose(R@R.T,np.eye(3),atol=.02) and np.linalg.det(R)>.95
        assert np.asarray(c['focal']).shape==(2,) and min(c['focal'])>0
        assert np.isfinite(c['princpt']).all()
    return frames,cams

# Stage only one part at a time (usually 2 GiB), verify official MD5, then read.
# A repaired archive part is preserved on Drive; a bad original is renamed, never deleted.
def stage_part(cap,rel):
    expected=CHECKS[cap].get(rel)
    if expected is None: raise RuntimeError('No official checksum for '+rel)
    local=WORK/'current_archive_part.bin'
    source=ARC/cap/rel
    STATUS.update(phase='checksum/stage '+cap.split('--')[3]+' '+Path(rel).name)
    if source.exists(): retry(lambda:shutil.copyfile(source,local))
    else: local.unlink(missing_ok=True)
    if not local.exists() or digest(local,'md5')!=expected:
        say('Repairing '+rel)
        # The short source may be resumed, but only an exact official MD5 is accepted.
        download(BASE+'/'+cap+'/'+rel,local,expected)
        tmp=source.with_name(source.name+'.verified_repair')
        persist_file(local,tmp)
        if source.exists():
            backup=source.with_name(source.name+'.before_repair_'+time.strftime('%Y%m%d_%H%M%S'))
            retry(lambda:os.replace(source,backup))
        retry(lambda:os.replace(tmp,source))
    got=digest(local,'md5')
    if got!=expected: raise RuntimeError('Checksum verification failed')
    RECEIPTS[cap+'/'+rel]={'bytes':local.stat().st_size,'md5':got,'official_md5_match':True}
    save_json(OUT/'archive_receipts.json',RECEIPTS)
    return local

class Parts(io.RawIOBase):
    def __init__(self,cap,rels,extension=None):
        self.cap,self.rels,self.extension=cap,list(rels),extension
        self.n=0; self.fp=None; self.used_extension=False
    def readable(self): return True
    def readinto(self,b):
        while True:
            if self.fp is None:
                if self.n<len(self.rels): p=stage_part(self.cap,self.rels[self.n]); self.n+=1
                elif self.extension and not self.used_extension:
                    p=self.extension; self.used_extension=True
                else: return 0
                self.fp=open(p,'rb')
            n=self.fp.readinto(b)
            if n: return n
            self.fp.close(); self.fp=None
    def close(self):
        if self.fp: self.fp.close()
        super().close()

def archive_rels(cap,folder,stem,count=None):
    if count is not None:
        suffixes=[chr(97+i//26)+chr(97+i%26) for i in range(count)]
        return [folder+'/'+stem+s for s in suffixes]
    rels=sorted(k for k in CHECKS[cap] if re.fullmatch(re.escape(folder+'/'+stem)+'[a-z]{2}',k))
    if not rels: raise RuntimeError('Missing annotation archive definitions: '+stem)
    return rels

def tar_open(cap,rels,extension=None):
    raw=Parts(cap,rels,extension)
    return raw,tarfile.open(fileobj=io.BufferedReader(raw,4*1024**2),mode='r|gz',bufsize=1024**2)

def labels(cap,frames):
    out=OUT/cap; out.mkdir(parents=True,exist_ok=True)
    # Validated compact arrays replace slow random access to hundreds of thousands of JSON files.
    ids=np.array(sorted(frames),dtype=np.int64); lookup={int(v):i for i,v in enumerate(ids)}
    pose=np.full((len(ids),2,48),np.nan,np.float32)
    shape=np.full((len(ids),2,10),np.nan,np.float32)
    trans=np.full((len(ids),2,3),np.nan,np.float32)
    kp=np.full((len(ids),2,24,3),np.nan,np.float32)
    bad=[]
    for kind,folder,stem in [('mano','mano_fits','mano_fits.tar.gz'),('kp','orig_fits','orig_fits.tar.gz')]:
        STATUS.update(phase='annotation audit '+kind)
        raw,tar=tar_open(cap,archive_rels(cap,folder,stem))
        try:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.json'): continue
                name=member.name.replace('\\','/')
                if kind=='mano':
                    m=re.search(r'(\d+)_(left|right)\.json$',name)
                    if not m: continue
                    fid,hand=int(m[1]),m[2]
                else:
                    m=re.search(r'(?:^|/)(left|right)/Keypoints/keypoint-(\d+)\.json$',name)
                    if not m: continue
                    hand,fid=m[1],int(m[2])
                if fid not in lookup: continue
                i,h=lookup[fid],['left','right'].index(hand)
                obj=json.load(tar.extractfile(member))
                try:
                    if kind=='mano':
                        vals=[np.asarray(obj[k],np.float32).reshape(-1) for k in ('pose','shape','trans')]
                        assert [len(v) for v in vals]==[48,10,3] and all(np.isfinite(v).all() for v in vals)
                        pose[i,h],shape[i,h],trans[i,h]=vals
                    else:
                        a=np.asarray(obj,np.float32); assert a.shape==(24,3) and np.isfinite(a).all()
                        assert np.max(np.linalg.norm(a-a[:1],axis=-1))>1
                        kp[i,h]=a
                except (AssertionError,ValueError,TypeError) as e:
                    bad.append({'frame':fid,'hand':hand,'source':name,'error':type(e).__name__})
        finally: tar.close(); raw.close()
    mano_valid=np.isfinite(pose).all(-1)&np.isfinite(shape).all(-1)&np.isfinite(trans).all(-1)
    kp_valid=np.isfinite(kp).all((-1,-2))
    local=WORK/'labels.npz'
    np.savez_compressed(local,frame_ids=ids,mano_pose_axis_angle=pose,mano_shape=shape,mano_trans_source=trans,
                        keypoints_world_mm=kp,mano_valid=mano_valid,keypoints_valid=kp_valid,hand_order=np.array(['left','right']))
    sha=persist_file(local,out/'labels.npz')
    save_json(out/'label_audit.json',{'frames':len(ids),'mano_valid_left_right':mano_valid.sum(0).tolist(),
              'keypoints_valid_left_right':kp_valid.sum(0).tolist(),'malformed':bad,'labels_sha256':sha})
    say('Labels verified: %d frames; MANO %s; keypoints %s'%(len(ids),mano_valid.sum(0),kp_valid.sum(0)))
    return ids,lookup,mano_valid,kp_valid,kp

def valid_path(name):
    pp=PurePosixPath(name)
    if pp.is_absolute() or '..' in pp.parts: raise RuntimeError('Unsafe archive member')
    parts=list(pp.parts)
    if 'envmap_per_segment' not in parts: return None
    return PurePosixPath('Mugsy_cameras',*parts[parts.index('envmap_per_segment'):]).as_posix()

def verify_write_image(job):
    cap,rel,data=job
    decoded=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if decoded is None or decoded.size==0: raise ValueError('Undecodable source image: '+rel)
    dst=EXT/cap/rel; sha=hashlib.sha256(data).hexdigest(); changed=False
    def check(): return dst.exists() and dst.stat().st_size==len(data) and digest(dst)==sha
    if not retry(check):
        atomic_bytes(dst,data); changed=True
    return {'path':rel,'sha256':sha,'bytes':len(data),'height':decoded.shape[0],'width':decoded.shape[1],
            'repaired':changed,'mtime_ns':dst.stat().st_mtime_ns}

def image_pass(label,cap,nparts,frames,cams,ann):
    ids,lookup,mv,kv,kp=ann
    folder=OUT/cap
    journal=folder/'verified_images.jsonl'
    previous={}
    if journal.exists():
        for line in retry(journal.read_text).splitlines():
            try:
                r=json.loads(line); previous[r['path']]=r
            except (ValueError,KeyError): pass
    rels=archive_rels(cap,'Mugsy_cameras',STEM,nparts)
    extension=None; extension_info=None
    if label=='HIR112':
        # Only finish the single image cut by the existing ac boundary.
        # A bounded 8 MiB range of ad is enough for normal PNG size; never acquire all 42 missing shards.
        extension=WORK/'HIR_boundary_8MiB.bin'
        tail_rel='Mugsy_cameras/'+STEM+'ad'
        download(BASE+'/'+cap+'/'+tail_rel,extension,byte_range=(0,8*1024**2-1))
        extension_info={'source':tail_rel,'range':[0,8*1024**2-1],'sha256':digest(extension),'purpose':'finish image at acquired-prefix boundary'}
        persist_file(extension,folder/'HIR_boundary_8MiB.bin')
    raw,tar=tar_open(cap,rels,extension)
    rows=[]; pending=[]; seen=set(); overlays={}; started=time.monotonic(); count=0; repaired=0
    errors=[]; end_reason='complete_archive'; boundary_finished=False
    f=open(journal,'a',encoding='utf-8')
    pool=ThreadPoolExecutor(max_workers=4)
    def consume(future,rel,fid,cam,header_extended):
        nonlocal count,repaired
        rec=future.result(); count+=1; repaired+=int(rec.get('repaired',False))
        i=lookup.get(fid)
        reasons=[]
        if i is None: reasons.append('frame_not_in_official_segment_list')
        if cam not in cams: reasons.append('camera_missing')
        if i is not None and not mv[i].any(): reasons.append('no_valid_MANO_target')
        record=dict(rec,frame_id=fid,camera_id=cam,segment=frames.get(fid,''),annotation_index=i,
                    mano_left=bool(mv[i,0]) if i is not None else False,mano_right=bool(mv[i,1]) if i is not None else False,
                    kp_left=bool(kv[i,0]) if i is not None else False,kp_right=bool(kv[i,1]) if i is not None else False,
                    usable=not reasons,issues=reasons)
        rows.append(record); f.write(json.dumps(record)+'\n')
        if count%200==0: f.flush()
        STATUS.update(phase='verify/extract '+label,images=count,repaired=repaired)
        if i is not None and cam in cams and (cam not in overlays) and kv[i].any(): overlays[cam]=(record,kp[i].copy(),kv[i].copy())
    try:
        for member in tar:
            if not member.isfile(): continue
            rel=valid_path(member.name)
            if rel is None or '/images/' not in '/'+rel or Path(rel).suffix.lower() not in ('.png','.jpg','.jpeg'): continue
            # Stop after the first complete image requiring continuation beyond the existing prefix.
            # read-ahead can mark extension slightly early, but we additionally require >=8581 images.
            p=PurePosixPath(rel); cam=p.parent.name
            if not p.stem.isdigit(): raise RuntimeError('Unrecognized image frame: '+rel)
            fid=int(p.stem)
            if rel in seen: raise RuntimeError('Duplicate archive image: '+rel)
            seen.add(rel)
            data=tar.extractfile(member).read()
            if len(data)!=member.size: raise IOError('Incomplete image member: '+rel)
            prior=previous.get(rel); dst=EXT/cap/rel
            skip=False
            if prior:
                try: skip=dst.stat().st_size==prior['bytes'] and dst.stat().st_mtime_ns==prior['mtime_ns'] and hashlib.sha256(data).hexdigest()==prior['sha256']
                except (OSError,KeyError): pass
            if skip:
                from concurrent.futures import Future
                future=Future(); future.set_result(dict(prior,repaired=False))
            else: future=pool.submit(verify_write_image,(cap,rel,data))
            pending.append((future,rel,fid,cam,raw.used_extension))
            if len(pending)>=8: consume(*pending.pop(0))
            if label=='HIR112' and raw.used_extension and len(seen)>=8581:
                boundary_finished=True; end_reason='acquired_prefix_plus_completed_boundary_image'; break
        for job in pending: consume(*job)
        pending=[]
        if label=='HIR112' and not boundary_finished: raise RuntimeError('HIR boundary not completed within 8 MiB allowance')
    finally:
        pool.shutdown(wait=True); f.flush(); os.fsync(f.fileno()); f.close(); tar.close(); raw.close()
    if not rows: raise RuntimeError('No image members found')
    # Build temporal links strictly within capture/camera/segment; preserve frame gaps explicitly.
    groups=defaultdict(list)
    for r in rows:
        if r['usable']: groups[(r['camera_id'],r['segment'])].append(r)
    for group in groups.values():
        group.sort(key=lambda r:r['frame_id'])
        for j,r in enumerate(group):
            r['prev_frame_id']=group[j-1]['frame_id'] if j else None
            r['next_frame_id']=group[j+1]['frame_id'] if j+1<len(group) else None
            r['frame_gap']=r['frame_id']-group[j-1]['frame_id'] if j else None
    # JSONL retains every checked image, including explicit reasons if a source label is unavailable.
    atomic_bytes(folder/'all_images_audit.jsonl', ''.join(json.dumps(r)+'\n' for r in rows).encode())
    usable=[r for r in rows if r['usable']]
    atomic_bytes(folder/'trainable_manifest.jsonl',''.join(json.dumps(r)+'\n' for r in usable).encode())
    seq=[{'camera_id':cam,'segment':seg,'frame_ids':[r['frame_id'] for r in group]} for (cam,seg),group in sorted(groups.items())]
    save_json(folder/'temporal_sequences.json',seq)
    # Camera-frame projection statistics and visual evidence, with real calibration.
    proj={}; tiles=[]
    for cam,(r,world,valid) in sorted(overlays.items()):
        im=cv2.imread(str(EXT/cap/r['path']))
        c=cams[cam]; R=np.asarray(c['R']); t=np.asarray(c['t']).reshape(3)
        focal=np.asarray(c['focal']); pp=np.asarray(c['princpt'])
        stats=[]
        for h in range(2):
            if not valid[h]: continue
            xyz=world[h]@R.T+t; z=xyz[:,2]
            uv=xyz[:,:2]/np.maximum(z[:,None],1e-6)*focal+pp
            inside=(z>0)&(uv[:,0]>=0)&(uv[:,0]<im.shape[1])&(uv[:,1]>=0)&(uv[:,1]<im.shape[0])
            stats.append({'hand':['left','right'][h],'positive_depth':float((z>0).mean()),'inside_image':float(inside.mean())})
            for x,y in uv[inside]: cv2.circle(im,(int(x),int(y)),4,((0,255,0) if h==0 else (0,0,255)),-1)
        proj[cam]=stats
        cv2.putText(im,label+' '+cam+' '+str(r['frame_id']),(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        tiles.append(cv2.resize(im,(334,512)))
    if tiles:
        montage=np.concatenate(tiles,axis=1); ok,png=cv2.imencode('.png',montage); assert ok
        atomic_bytes(folder/'projection_overlay.png',png.tobytes())
    missingkp=sum(1 for r in usable if not (r['kp_left'] and r['kp_right']))
    summary={'label':label,'capture':cap,'images_verified':len(rows),'images_trainable_MANO':len(usable),
      'images_repaired_or_written':repaired,'images_with_some_missing_keypoint_targets':missingkp,
      'image_bytes':sum(r['bytes'] for r in rows),'sequences':len(seq),'segments':len({r['segment'] for r in usable}),
      'source_end':end_reason,'extension':extension_info,'projection_samples':proj,
      'unusable_images':[{k:r[k] for k in ('path','issues')} for r in rows if not r['usable']],
      'elapsed_seconds':round(time.monotonic()-started,1)}
    save_json(folder/'quality_report.json',summary)
    say(label+' RESULT '+json.dumps({k:summary[k] for k in ('images_verified','images_trainable_MANO','images_repaired_or_written','images_with_some_missing_keypoint_targets')}))
    return summary

results=[]
try:
    say('Repair scope: existing 3 captures; CPU; no training; resumable logs on Drive.')
    # Source checksums replace stale sidecar claims. BKS an is repaired during this first pass.
    for label,cap,nparts in CAPS:
        STATUS.update(phase='metadata '+label,images=0)
        frames,cams=metadata(cap)
        ann=labels(cap,frames)
        result=image_pass(label,cap,nparts,frames,cams,ann)
        results.append(result)
        save_json(OUT/'repair_progress.json',{'captures':results,'downloaded_bytes_this_run':NEW_BYTES,'complete':False})
        del ann
    total=sum(r['images_verified'] for r in results)
    ready=sum(r['images_trainable_MANO'] for r in results)
    report={'captures':results,'verified_images':total,'MANO_trainable_images':ready,
      'all_acquired_images_have_MANO_training_targets':ready==total,
      'downloaded_bytes_this_run':NEW_BYTES,'elapsed_seconds':time.monotonic()-START,
      'scope':'All complete image members in acquired 202107 and BKS archives, and acquired HIR aa-ac prefix plus completed boundary image.',
      'assumptions':['Temporal order uses official frame IDs within capture/camera/segment; frame gaps retained; no guessed FPS.',
        'Original 24-joint keypoint order retained; world keypoints and calibration translation are mm.',
        'MANO pose is 48 axis-angle parameters, shape has 10 values; translation retained in source units.',
        'Missing keypoints are masked; finite MANO targets can still supervise these images.',
        'This checks data integrity, joins and projection samples; it does not prove every published fit is geometrically perfect.',
        'The full unacquired HIR capture is outside scope. No model training performed.']}
    save_json(OUT/'FINAL_QUALITY_REPORT.json',report)
    say('FINAL: %d/%d image records have valid MANO supervision; downloaded %.3f GiB.'%(ready,total,NEW_BYTES/GIB))
    say('Inspect FINAL_QUALITY_REPORT.json and projection_overlay.png in each capture folder.')
    if ready!=total: raise RuntimeError('Unmatched source images remain: see explicit unusable_images. NOT claiming 100% ready.')
    say('INTEGRITY + MANO LABEL JOIN COMPLETE. Await independent Drive persistence/overlay verification.')
except BaseException as e:
    say('STOPPED SAFELY: '+type(e).__name__+': '+str(e))
    try: save_json(OUT/'last_interruption.json',{'error':repr(e),'status':STATUS,'completed_captures':results,'time':time.strftime('%Y-%m-%dT%H:%M:%S')})
    except Exception: pass
    raise
finally:
    HALT.set()