"""V2 conservative fallbacks; all decisions depend on predicted geometry/masks only.

Developed after reviewing these eight development cases, not an independent test.
Existing valid V1 cuff outputs are preserved exactly. Nulls may be recovered.
"""
import numpy as np
import geometry_v1 as v1

iou = v1.iou
associate = v1.associate
point_at = v1.point_at
JOINTS = v1.JOINTS


def arm_geometry(person, side, size):
    arm = v1.arm_geometry(person, side, size)
    arm['side_method'] = 'shoulder_order' if arm['image_side'] != 'unknown' else 'unresolved'
    if arm['image_side'] != 'unknown':
        return arm
    # For profile views shoulders can overlap. Use upper-arm midpoint order,
    # provided BOTH shoulders/elbows are confident and inside the image.
    xy = np.asarray(person['xy'], float)
    conf = np.asarray(person['conf'], float)
    pts = xy[[5, 6, 7, 8]]
    w, h = size
    usable = (np.isfinite(pts).all() and min(conf[[5, 6, 7, 8]]) >= .25
              and (pts >= 0).all() and (pts[:, 0] < w).all() and (pts[:, 1] < h).all())
    if usable:
        dx = (xy[5, 0] + xy[7, 0] - xy[6, 0] - xy[8, 0]) / 2
        if abs(dx) > .015*w:
            arm['image_side'] = 'image_left' if (dx < 0) == (side == 'left') else 'image_right'
            arm['side_method'] = 'upper_arm_midpoint_fallback'
            if arm['failure_reason'] == 'side_ambiguous':
                arm['failure_reason'] = None
    return arm


def gradual_transition(profiles):
    """Hysteresis around a 50% occupancy crossing, with local/tail safeguards.

The 70%-to-30% transition may span up to 25% of the arm. Three strip widths
must agree within 6%. Neither a class label nor a desired ratio is an input.
"""
    cuts = []
    intervals = []
    for profile in profiles:
        raw = np.asarray(profile, float)
        a = np.convolve(np.pad(raw, (1, 1), mode='edge'), np.ones(3)/3, mode='valid')
        if len(a) != 101 or np.mean(a[:5]) < .55:
            return None, 'insufficient_shoulder_support', []
        candidates = []
        for i in range(5, 94):
            if not (a[i-1] >= .5 and a[i] < .5):
                continue
            highs = [j for j in range(max(0, i-15), i) if a[j] >= .7]
            lows = [j for j in range(i, min(101, i+16)) if a[j] <= .3]
            if not highs or not lows:
                continue
            lo, hi = highs[-1], lows[0]
            if hi-lo > 25 or np.mean(a[hi:]) > .3 or np.mean(a[hi:] >= .65) > .1:
                continue
            if np.maximum(0, np.diff(a[lo:hi+1])).sum() > .25:
                continue
            if np.mean(a[:lo+1]) < .65:
                continue
            frac = (a[i-1]-.5) / max(1e-9, a[i-1]-a[i])
            candidates.append(((i-1+frac)/100, [lo/100, hi/100]))
        if len(candidates) != 1:
            return None, 'no_unique_gradual_transition', []
        cuts.append(candidates[0][0]); intervals.append(candidates[0][1])
    if max(cuts)-min(cuts) > .06:
        return None, 'unstable_gradual_boundary_across_widths', intervals
    return float(np.median(cuts)), None, intervals


def locate_cuff(mask, arm):
    result = v1.locate_cuff(mask, arm)
    result['cuff_method'] = 'v1_strict' if result['auto_cuff'] is not None else None
    if result['failure_reason'] != 'no_stable_cuff_boundary':
        return result
    t, reason, intervals = gradual_transition(result['mask_profiles'])
    result['fallback_attempted'] = True
    result['fallback_failure'] = reason
    result['transition_intervals'] = intervals
    if reason:
        return result
    q = np.asarray([arm['points'][k] for k in ('shoulder', 'elbow', 'wrist')])
    point, _ = point_at(q, t)
    result.update(auto_cuff=point.tolist(), ratio_pred_or_null=t, failure_reason=None,
                  cuff_method='gradual_mask_transition', cuff_heuristic_score=None)
    return result


def infer_instances(instances, people, size, mask_loader):
    rows = []
    for inst in instances:
        if inst['category'] not in ('top', 'outerwear', 'dress'):
            continue
        pid, reason, ranks = associate(inst['bbox'], people)
        row = {**inst, 'predicted_person_id': pid, 'association_failure': reason,
               'person_candidates': ranks, 'arms': []}
        if pid is not None:
            person = next(p for p in people if p['person_id'] == pid)
            mask = mask_loader(inst['mask_file'])
            if mask.shape != (size[1], size[0]):
                raise ValueError('Mask must be in original image coordinates')
            for side in JOINTS:
                arm = arm_geometry(person, side, size)
                arm.update(locate_cuff(mask, arm))
                row['arms'].append(arm)
        rows.append(row)
    return rows
