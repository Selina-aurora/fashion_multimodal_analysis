"""记录互相矛盾的模型输出；不使用 GT，不自动修正类别、框或掩码。

该规则筛查语义冲突，不验证准确率，也不判定部位真实缺席。
原 status/label/confidence 作为候选证据保留，避免把弃权算作正确检出。
"""
from __future__ import annotations

from itertools import combinations
import math
import re


def bbox_iou(a,b):
    if a is None or b is None or len(a)!=4 or len(b)!=4:return 0.
    if not all(math.isfinite(float(v)) for v in [*a,*b]):return 0.
    area=lambda x:max(0.,x[2]-x[0])*max(0.,x[3]-x[1])
    intersection=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
    return intersection/max(1e-12,area(a)+area(b)-intersection)


def pattern_phrase_label(phrase):
    """只解析明确命名的图案；泛称 pattern/printed pattern 不算冲突。"""
    text=re.sub(r'[^a-z ]',' ',str(phrase).lower())
    matches=[]
    for name,words in [
        ('floral',('floral','flower')),('striped',('stripe','striped','stripes')),
        ('checked_plaid',('plaid','checked','checkered')),
        ('polka_dot',('polka',)),('animal_print',('animal','leopard','zebra')),
        ('camouflage',('camouflage',)),('geometric_abstract',('geometric','abstract')),
        ('graphic_logo',('logo','graphic')),
    ]:
        if any(re.search(r'\b'+word+r'\b',text) for word in words):matches.append(name)
    return matches[0] if len(matches)==1 else None


def mark_region(region,reason):
    """改变交付状态但保留模型候选、原坐标和掩码供审核。"""
    region.setdefault('model_status_before_review',region.get('status'))
    region['status']='requires_review'
    region['semantic_review_status']='pending'
    reasons=region.setdefault('consistency_review_reasons',[])
    if reason not in reasons:reasons.append(reason)


def apply_consistency_review(regions,attributes,collision_iou=.90):
    """对本实例模型输出做原位标记，返回可审查的证据列表。

    框几乎相同仅是审核线索，不证明真实部位不能接近；阈值未经校准。
    DINO/CLIP 分歧也不能证明哪一方正确。
    """
    if not 0<collision_iou<=1:raise ValueError('collision_iou must be in (0,1]')
    flags=[]
    local=['collar','cuff','hem','pocket','waist']
    active={k:v for k,v in regions.items() if v.get('status')=='detected'}
    for first,second in combinations(local,2):
        if first not in active or second not in active:continue
        overlap=bbox_iou(active[first].get('bbox'),active[second].get('bbox'))
        if overlap<collision_iou:continue
        reason='different_regions_share_near_identical_box'
        for key in (first,second):mark_region(regions[key],reason)
        flags.append({'reason':reason,'regions':[first,second],'bbox_iou':overlap,
                      'automatic_action':'require_review_keep_geometry',
                      'note':'Near-identical boxes are a review cue, not proof either region is false.'})
    pattern=regions.get('pattern',{})
    attribute=attributes.get('pattern',{})
    if pattern.get('status')=='detected' and attribute.get('status')=='predicted':
        grounded=pattern_phrase_label(pattern.get('model_phrase',''))
        classified=attribute.get('label')
        if grounded and classified and classified!='other_pattern' and grounded!=classified:
            mark_region(pattern,'grounding_and_pattern_attribute_disagree')
            evidence={'reason':'grounding_and_pattern_attribute_disagree','region':'pattern',
                      'grounding_phrase':pattern.get('model_phrase'),'grounding_pattern_candidate':grounded,
                      'attribute_pattern_candidate':classified,'automatic_action':'require_review_no_label_override'}
            flags.append(evidence)
            attribute['consistency_review_status']='requires_review'
            attribute['consistency_review_evidence']=evidence
            # 分歧时对属性弃权，没有证据支持哪一方更可靠。
            # 原属性候选完整保留，不能将弃权计为准确率提升。
            attribute.setdefault('model_candidate_before_review',{'status':attribute.get('status'),
                'label':attribute.get('label'),'confidence':attribute.get('confidence')})
            attribute['status']='requires_review';attribute['label']='';attribute['confidence']=None
    collar=regions.get('collar',{})
    neckline=attributes.get('neckline',{})
    if collar.get('status')=='requires_review' and neckline.get('status')=='predicted':
        neckline['model_candidate_before_review']={'status':neckline['status'],
            'label':neckline.get('label'),'confidence':neckline.get('confidence')}
        neckline.update(status='requires_review',label='',confidence=None,
                        consistency_review_status='requires_review')
        flags.append({'reason':'neckline_depends_on_conflicted_collar','attribute':'neckline',
                      'automatic_action':'require_review_no_label_override'})
    return flags
