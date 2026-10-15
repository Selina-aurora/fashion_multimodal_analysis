"""GT-free fixed first-pass heuristics. Scores are NOT calibrated probabilities."""
import numpy as np

JOINTS = {'left': [5, 7, 9], 'right': [6, 8, 10]}


def iou(a, b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=x*y
    return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def associate(box, people):
    """Associate one predicted garment to a person, using boxes/shoulders only."""
    ranks=[]
    for p in people:
        b=p['bbox']; k=np.asarray(p['xy']); c=np.asarray(p['conf'])
        if min(c[5],c[6])<.25: continue
        ix=max(0,min(box[2],b[2])-max(box[0],b[0]))
        iy=max(0,min(box[3],b[3])-max(box[1],b[1]))
        covered=ix*iy/max(1,(box[2]-box[0])*(box[3]-box[1]))
        mid=(k[5]+k[6])/2; w=box[2]-box[0]; h=box[3]-box[1]
        # The upper garment should surround the shoulders, including loose sleeves.
        plausible=box[0]-.25*w<=mid[0]<=box[2]+.25*w and box[1]-.3*h<=mid[1]<=box[1]+.6*h
        center=np.array([(box[0]+box[2])/2,box[1]+.2*h])
        close=max(0.,1-float(np.linalg.norm(mid-center))/max(1,np.hypot(w,h)))
        score=.7*covered+.3*close
        if plausible and covered>=.5: ranks.append((score,p['person_id']))
    ranks.sort(reverse=True)
    detail=[{'person_id':pid,'score':float(s)} for s,pid in ranks]
    if not ranks or ranks[0][0]<.6: return None,'no_plausible_person',detail
    if len(ranks)>1 and ranks[0][0]-ranks[1][0]<.12: return None,'ambiguous_person',detail
    return ranks[0][1],None,detail


def arm_geometry(person, side, size):
    ids=JOINTS[side]; q=np.asarray(person['xy'],float)[ids]; c=np.asarray(person['conf'])[ids]
    w,h=size; lens=np.linalg.norm(np.diff(q,axis=0),axis=1)
    result={'anatomical_side':side,'image_side':'unknown','points':dict(zip(['shoulder','elbow','wrist'],q.tolist())),
            'joint_confidences':c.tolist(),'arm_length_px':float(sum(lens)),'failure_reason':None}
    dx=person['xy'][5][0]-person['xy'][6][0]
    if abs(dx)>.015*w: result['image_side']='image_left' if (dx<0)==(side=='left') else 'image_right'
    reason=None
    if min(c)<.25: reason='low_joint_confidence'
    elif not np.isfinite(q).all() or (q<0).any() or (q[:,0]>=w).any() or (q[:,1]>=h).any(): reason='joint_outside_image'
    elif min(lens)<max(8,.015*np.hypot(w,h)): reason='arm_too_short'
    elif max(lens)/min(lens)>3: reason='implausible_segment_lengths'
    elif np.dot(q[1]-q[0],q[2]-q[1])/np.prod(lens)<-.5: reason='strongly_folded_arm'
    elif result['image_side']=='unknown': reason='side_ambiguous'
    result['failure_reason']=reason
    return result


def point_at(q,t):
    lengths=np.linalg.norm(np.diff(q,axis=0),axis=1); d=t*sum(lengths)
    if d<=lengths[0]: return q[0]+(q[1]-q[0])*d/lengths[0],q[1]-q[0]
    return q[1]+(q[2]-q[1])*(d-lengths[0])/lengths[1],q[2]-q[1]


def locate_cuff(mask, arm):
    """Find stable garment-to-background transition along an automatically detected arm.

    This is a boundary proposal, not a trained cuff detector. Occlusion/torso boundaries
    can fool it; retain profiles and require visual review of this exploration set.
    """
    base={'auto_cuff':None,'ratio_pred_or_null':None,'cuff_heuristic_score':None,'failure_reason':arm['failure_reason']}
    if base['failure_reason']: return base
    q=np.array(list(arm['points'].values()),float); length=arm['arm_length_px']; h,w=mask.shape
    ts=np.linspace(0,1,101); profiles=[]
    # Three strip widths test whether the proposed boundary is spatially stable.
    for scale in (.015,.025,.035):
        profile=[]
        for t in ts:
            p,v=point_at(q,t); n=np.array([-v[1],v[0]])/np.linalg.norm(v)
            pts=p+np.linspace(-max(2,scale*length),max(2,scale*length),9)[:,None]*n
            xy=np.rint(pts).astype(int)
            if ((xy[:,0]<0)|(xy[:,0]>=w)|(xy[:,1]<0)|(xy[:,1]>=h)).any():
                return dict(base,failure_reason='boundary_search_outside_image')
            profile.append(float(mask[xy[:,1],xy[:,0]].mean()))
        profiles.append(profile)
    base['mask_profiles']=profiles
    cuts=[]
    for profile in profiles:
        a=np.asarray(profile)
        if np.mean(a[:5])<.55: return dict(base,failure_reason='shoulder_not_in_garment_or_sleeveless')
        candidates=[]
        for i in range(5,94):
            if np.mean(a[i-5:i])>=.65 and np.mean(a[i:i+6])<=.35 and np.mean(a[i+6:])<=.30:
                candidates.append(i)
        if not candidates: return dict(base,failure_reason='no_stable_cuff_boundary')
        cuts.append(candidates[0]/100)
    if max(cuts)-min(cuts)>.06: return dict(base,failure_reason='unstable_cuff_across_widths')
    t=float(np.median(cuts)); p,_=point_at(q,t)
    return dict(base,auto_cuff=p.tolist(),ratio_pred_or_null=t,cuff_heuristic_score=float(1-(max(cuts)-min(cuts))/.1),failure_reason=None)


def infer_instances(instances, people, size, mask_loader):
    """No target IDs, GT boxes, sides, points, ratios or classes are accepted."""
    results=[]
    for inst in instances:
        if inst['category'] not in ('top','outerwear','dress'): continue
        pid,reason,ranks=associate(inst['bbox'],people)
        row={**inst,'predicted_person_id':pid,'association_failure':reason,'person_candidates':ranks,'arms':[]}
        if pid is not None:
            person=next(p for p in people if p['person_id']==pid)
            mask=mask_loader(inst['mask_file'])
            for side in JOINTS:
                arm=arm_geometry(person,side,size); arm.update(locate_cuff(mask,arm)); row['arms'].append(arm)
        results.append(row)
    return results
