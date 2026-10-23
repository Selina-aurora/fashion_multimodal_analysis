"""V3 cuff candidate geometry: dev_v1 exploratory, not production or GT.

This module never reads annotation labels, expected endpoints, image-specific IDs, or
fixed GT-side matches. It only uses a predicted garment mask and predicted pose.
Candidates require manual review; unbounded endings cannot become B endpoints.
"""
from __future__ import annotations
import numpy as np
import cv2


def bilinear(img, points):
    h, w = img.shape
    xy = np.asarray(points, float)
    x, y = xy[:, 0], xy[:, 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    xc, yc = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0 = np.floor(xc).astype(int); y0 = np.floor(yc).astype(int)
    x1, y1 = np.minimum(x0 + 1, w-1), np.minimum(y0 + 1, h-1)
    dx, dy = xc - x0, yc - y0
    values = ((1-dx)*(1-dy)*img[y0, x0] + dx*(1-dy)*img[y0, x1]
              + (1-dx)*dy*img[y1, x0] + dx*dy*img[y1, x1])
    values[~valid] = np.nan
    return values


def reject(reason, metrics=None):
    return dict(status='abstain', reason=reason, A_outer_px=None,
                B_inner_px=None, midpoint_px=None, metrics=metrics or {})


def _scan(mask, p, direction, normal, width, along):
    svals = np.arange(-width, width+.01, 1.5)
    uvals = np.arange(-along, along+.01, 1.25)
    out = []
    for s in svals:
        values = bilinear(mask, p + normal*s + uvals[:, None]*direction)
        if np.isnan(values).any():
            out.append(None)
            continue
        v = np.convolve(np.pad(values, (2, 2), mode='edge'), np.ones(5)/5, mode='valid')
        matches = []
        for i in range(7, len(v)-9):
            if (v[i-1] >= .50 and v[i] < .50
                    and np.mean(v[i-6:i-2]) >= .70
                    and np.mean(v[i+3:i+8]) <= .25):
                if np.max(v[i+4:min(i+13, len(v))]) > .55:
                    continue
                alpha = (v[i-1]-.5)/max(1e-8, v[i-1]-v[i])
                matches.append(float(uvals[i-1]+alpha*(uvals[i]-uvals[i-1])))
        out.append(matches[0] if len(matches) == 1 else None)
    good = np.array([x is not None for x in out])
    runs = []
    i = 0
    while i < len(out):
        if not good[i]:
            i += 1
            continue
        j = i
        while (j+1 < len(out) and good[j+1]
               and abs(out[j+1]-out[j]) <= 5.0):
            j += 1
        if (min(abs(svals[i]), abs(svals[j])) <= 9 or svals[i] <= 0 <= svals[j]):
            runs.append((i, j))
        i = j+1
    if not runs:
        return None
    # Reject competing strong trajectories rather than relying on GT to pick one.
    runs = sorted(runs, key=lambda r: r[1]-r[0], reverse=True)
    if len(runs)>1 and (runs[1][1]-runs[1][0]) > .8*(runs[0][1]-runs[0][0]):
        return {'reason': 'ambiguous_competing_boundary_tracks'}
    i, j = runs[0]
    left_pt = p+normal*svals[i]+direction*out[i]
    right_pt = p+normal*svals[j]+direction*out[j]
    us = np.asarray(out[i:j+1], float)
    return dict(i=i,j=j,n=len(svals),svals=svals,points=[left_pt,right_pt],
                span=float(svals[j]-svals[i]),vertical_range=float(np.ptp(us)),
                clipped_left=i==0,clipped_right=j==len(svals)-1,
                count=j-i+1)


def propose_endpoints(mask, arm, torso_x):
    """GT-blind proposals from pose and garment segmentation mask only.

    New vs V0: adaptive cross-section width and relative tilt guard; if a
    candidate runs into the torso, return OUTER POINT ONLY and B=null. This
    specifically avoids converting an unbounded mask/torso join into a GT.
    """
    if arm.get('failure_reason') or arm.get('auto_cuff') is None:
        return reject('no_valid_v2_arm_seed')
    if not np.isfinite(torso_x):
        return reject('no_torso_center')
    p = np.asarray(arm['auto_cuff'], float)
    q = np.array([arm['points'][key] for key in ('shoulder','elbow','wrist')], float)
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        return reject('invalid_pose')
    seglens = np.linalg.norm(np.diff(q, axis=0),axis=1)
    total = float(seglens.sum())
    if total < 45: return reject('short_arm')
    ratio = float(arm['ratio_pred_or_null'])
    if not np.isfinite(ratio): return reject('invalid_ratio')
    segment = 0 if ratio*total <= seglens[0] else 1
    if seglens[segment] < 1: return reject('invalid_tangent')
    direction = (q[segment+1]-q[segment])/seglens[segment]
    normal = np.array([-direction[1],direction[0]])
    m = np.asarray(mask)
    if m.ndim!=2:return reject('mask_dimensions')
    smooth = cv2.GaussianBlur(m.astype('float32'),(5,5),1)
    # Adaptive field uses *predicted arm length only*, fixed for all cases.
    # Wider than V0 to find physical ends when the V2 seed is near one endpoint.
    width = float(np.clip(total*.26,22,80))
    along = float(np.clip(total*.10,22,42))
    track = _scan(smooth, p, direction, normal, width, along)
    if track is None:return reject('no_stable_cross_arm_boundary')
    if 'reason' in track:return reject(track['reason'])
    i,j=track['i'],track['j']
    span,dy=track['span'],track['vertical_range']
    diag=dict(run_span_px=round(span,2),search_half_width_px=round(width,2),
              range_along_px=round(dy,2),relative_tilt=round(dy/max(span,1),3),
              track_count=track['count'],clipped_left=track['clipped_left'],
              clipped_right=track['clipped_right'])
    if span < max(12, .13*total):
        return reject('boundary_span_too_narrow',diag)
    p1,p2=track['points']
    # A = spatially outward from predicted torso; branch never reads GT-side.
    if abs(p1[0]-torso_x)>=abs(p2[0]-torso_x):
        outer,inner=p1,p2
        outer_clipped=track['clipped_left'];inner_clipped=track['clipped_right']
    else:
        outer,inner=p2,p1
        outer_clipped=track['clipped_right'];inner_clipped=track['clipped_left']
    if outer_clipped:
        return reject('outer_endpoint_clipped',diag)
    if inner_clipped:
        # Expose outer candidate but do NOT invent B or midpoint.
        return dict(status='partial_requires_review',reason='inner_endpoint_merges_with_torso',
                    A_outer_px=outer.tolist(),B_inner_px=None,midpoint_px=None,
                    metrics=diag)
    # Relative opening tilt identifies highly slanted seam/neighbor/torso drift;
    # it is geometry on the predicted mask, not GT closeness.
    if dy>26 or dy/max(span,1)>.55:
        return reject('boundary_tilt_or_neighbor',diag)
    if min(abs(float(track['svals'][i])),abs(float(track['svals'][j])))<4 and not (track['svals'][i]<0<track['svals'][j]):
        return reject('boundary_one_sided',diag)
    mid=(outer+inner)/2
    diag['center_to_seed_px']=round(float(np.linalg.norm(mid-p)),2)
    return dict(status='proposal_requires_review',reason=None,A_outer_px=outer.tolist(),
                B_inner_px=inner.tolist(),midpoint_px=mid.tolist(),metrics=diag)
