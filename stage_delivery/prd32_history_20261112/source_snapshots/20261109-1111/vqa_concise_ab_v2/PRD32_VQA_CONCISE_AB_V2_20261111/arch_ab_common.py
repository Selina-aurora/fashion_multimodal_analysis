"""Pure helpers: deterministic post-processing of baseline Tags for SHADOW comparison only."""
import json,re,hashlib,statistics

CN_CATEGORY={'top':'上衣','pants':'裤装','skirt':'裙装','outerwear':'外套','dress':'连衣裙','shoe':'鞋履','bag':'包袋','accessory':'配饰'}
UNKNOWN={'','不确定','未知','无','无设计','不适用','none','null','unknown','uncertain','n/a','未识别','无明显设计','不可见','看不清'}
DEBUG_TERMS=['红色边框','红框','蓝色边框','bounding box','检测框','标记框','图片中框','被框选','红色矩形','图中标记']

def valid_str(value):
    v=str(value).strip() if isinstance(value,(str,int,float)) else ''
    return v if v.lower() not in UNKNOWN and not re.fullmatch(r'[\s\-—–/]+',v) else ''

def parse_tags(raw):
    s=raw.strip()
    s=re.sub(r'^```(?:json)?\s*','',s,flags=re.I)
    s=re.sub(r'\s*```\s*$','',s)
    obj=json.loads(s)
    if not isinstance(obj,dict):raise ValueError('Tags JSON is not object')
    allowed={'category','primary_color','pattern','visible_design'}
    return {k:obj.get(k) for k in allowed}

def make_copy(tags_text,style='客观',min_chars=20,max_chars=80):
    """Never infer invisible materials/craft; no auto publish. Includes the uncertainty of Tags."""
    if not (10<=min_chars<=max_chars<=120):raise ValueError('template length target out of bounds')
    if style not in {'客观','简洁'}:raise ValueError('unsupported style')
    try: fields=parse_tags(tags_text)
    except Exception as e:
        return {'status':'BLOCKED_TAGS_UNPARSABLE','text':'','reason':str(e)}
    cat=CN_CATEGORY.get(valid_str(fields.get('category')),'商品')
    color=valid_str(fields.get('primary_color'))
    pattern=valid_str(fields.get('pattern'))
    design=valid_str(fields.get('visible_design'))
    if pattern in ('纯色','纯色系') and color:
        details=[f'以{color}为主色调', '整体呈纯色风格']
    else:
        details=[]
        if color:details.append(f'以{color}为主色调')
        if pattern:details.append(f'可见{pattern}图案')
    if design:details.append(f'可见{design}细节')
    if not details:
        return {'status':'BLOCKED_INSUFFICIENT_TAGS','text':'','reason':'no grounded descriptive attribute'}
    if style=='简洁':
        base=f'这款{cat}，'+'，'.join(details)+'。'
    else:
        base=f'这款{cat}的可见特征包括：'+'；'.join(details)+'。'
    if len(base)<min_chars:
        base += '具体细节以商品实物为准。'
    # Truncation could cut factual statements. Reject rather than slice.
    if len(base)>max_chars:
        return {'status':'BLOCKED_LENGTH_CONSTRAINT','text':base,'reason':f'length {len(base)} > max {max_chars}'}
    return {'status':'TEMPLATE_DRAFT_NEEDS_HUMAN_REVIEW','text':base,'reason':'Tags may itself be inaccurate; not PRD quality-accepted'}

def sha_text(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()

def percentile(vals,p):
    if not vals:return None
    s=sorted(float(x) for x in vals); pos=(len(s)-1)*p/100; lo=int(pos);hi=min(lo+1,len(s)-1)
    return round(s[lo]*(hi-pos)+s[hi]*(pos-lo),3)

def stats(vals):
    if not vals:return None
    vv=[float(x) for x in vals]
    return {'n':len(vv),'mean_ms':round(statistics.mean(vv),3),'p50_ms':percentile(vv,50),'p95_ms':percentile(vv,95)}
