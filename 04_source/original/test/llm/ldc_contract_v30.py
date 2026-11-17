"""Standard-library-only, bounded steady PINN contract; no dataset access."""
import hashlib
import json
import math

VERSION=30
CATEGORIES=dict(pressure_weight=[.5,1.,2.],physics_weight=[.0001,.0003,.001,.003,.01],
    bc_weight=[.001,.003,.01,.03,.1],warmup_fraction=[0.,.03125,.0625])
PARAMETERS={'learning_rate',*CATEGORIES}
BASELINE=dict(learning_rate=.001,pressure_weight=1.,physics_weight=.001,bc_weight=.01,warmup_fraction=.03125)
KEYS={'parameters','reason','evidence_ids','expected_effect','risk'}
SYSTEM=(
    'Choose training settings for one shared steady 2D FlowBench LDC PINN: geometry/Re -> u,v,p. '
    'This is a two-epoch technical test, not converged performance evidence. Supplied history is real; no TEST data. '
    'Minimize score: mean unweighted validation u/v/p MSE. Other errors and NS/divergence are diagnostics. '
    'Loss is TRAIN-std-normalized data MSE (pressure_weight on pressure), plus physics_weight times '
    'steady incompressible Navier-Stokes loss (viscosity 1/Re), plus bc_weight times no-slip loss. '
    'Fixed architecture,640TRAIN/160VAL,2epochs,seed,batch4,2048data/64PDE/128BC points per case. '
    'Adam/cosine; warmup_fraction delays PDE, then fixed 0.03125-fraction ramp. '
    'Inference 5ms is a separate fixed-architecture check; it is NOT a training-time or divergence limit. '
    'Return ONE COMPACT SINGLE-LINE JSON object only. Exactly five top-level keys: '
    'parameters,reason,evidence_ids,expected_effect,risk. No Markdown. '
    'parameters has exactly: learning_rate real [0.0001,0.002]; pressure_weight in [0.5,1,2]; '
    'physics_weight in [0.0001,0.0003,0.001,0.003,0.01]; bc_weight in [0.001,0.003,0.01,0.03,0.1]; '
    'warmup_fraction in [0,0.03125,0.0625]. No other settings. '
    'reason,expected_effect,risk are each ONE short string, 1 to 120 characters EACH. '
    'Generation is constrained to a JSON schema for all models; choose ONE NUMBER per parameter, never a range object. '
    'Use a brief public reason, a hypothesis for expected_effect, and one risk. Do not repeat phrases. '
    'evidence_ids is a nonempty list of unique completed history IDs; include latest_required_id. '
    'No tools,code,files,internet or fabricated measurements. Duplicate configurations still consume a trial.'
)
SYSTEM_SHA256=hashlib.sha256(SYSTEM.encode()).hexdigest()
HISTORY_KEYS={'id','parameters','score','velocity_error','pressure_error','momentum_rms','divergence_rms','training_seconds','training_seed','epochs'}


def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)


def validate_parameters(p):
    if not isinstance(p,dict) or set(p)!=PARAMETERS:raise ValueError('exact_parameter_keys')
    for k,v in p.items():
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError(k+'_finite_number')
        if k=='learning_rate':
            if not .0001<=v<=.002:raise ValueError('learning_rate_range')
        elif v not in CATEGORIES[k]:raise ValueError(k+'_category')
    return p


def validate_feedback(f):
    if set(f)!={'contract_version','phase','epochs','remaining_attempts','latest_required_id','history'}:raise ValueError('feedback_keys')
    if f['contract_version']!=VERSION or f['phase']!='technical_preflight' or f['epochs']!=2:raise ValueError('technical_phase_only')
    if type(f['remaining_attempts']) is not int or not 1<=f['remaining_attempts']<=2:raise ValueError('attempt_limit')
    ids=[]
    if not isinstance(f['history'],list) or not f['history']:raise ValueError('empty_real_history')
    for r in f['history']:
        if set(r)!=HISTORY_KEYS:raise ValueError('history_keys')
        if not isinstance(r['id'],str) or not r['id'] or r['id'] in ids:raise ValueError('unique_history_ids')
        ids.append(r['id']);validate_parameters(r['parameters'])
        for k in ['score','velocity_error','pressure_error','momentum_rms','divergence_rms','training_seconds']:
            v=r[k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError('invalid_metric_'+k)
        if r['epochs']!=2 or type(r['training_seed']) is not int:raise ValueError('inconsistent_trial_budget')
    if f['latest_required_id']!=ids[-1]:raise ValueError('latest_measured_feedback_required')
    canonical(f);return set(ids)


def messages(feedback):
    validate_feedback(feedback)
    return [{'role':'system','content':SYSTEM},{'role':'user','content':canonical(feedback)}]


def unique_pairs(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('duplicate_json_key')
        result[k]=v
    return result


def parse_reply(raw,feedback):
    ids=validate_feedback(feedback);text=raw.strip();transport='strict_json'
    if text.startswith('```json\n') and text.endswith('\n```'):
        text=text[8:-4].strip();transport='single_complete_json_fence'
    elif text.startswith('```\n') and text.endswith('\n```'):
        text=text[4:-4].strip();transport='single_complete_json_fence'
    try:
        obj=json.loads(text,object_pairs_hook=unique_pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite_json')))
        if not isinstance(obj,dict) or set(obj)!=KEYS:raise ValueError('exact_reply_keys')
        validate_parameters(obj['parameters'])
        for k in ['reason','expected_effect','risk']:
            if not isinstance(obj[k],str) or not obj[k].strip():raise ValueError('nonempty_'+k)
        for k in ['reason','expected_effect','risk']:
            if len(obj[k])>120:raise ValueError(k+'_max_120_characters')
        ev=obj['evidence_ids']
        if not isinstance(ev,list) or not ev or any(not isinstance(v,str) or v not in ids for v in ev) or len(set(ev))!=len(ev):raise ValueError('evidence_ids')
        if feedback['latest_required_id'] not in ev:raise ValueError('must_cite_latest_actual_trial')
        return obj,[],transport
    except (ValueError,TypeError,KeyError) as e:return None,[str(e)],transport


def repair_message(errors):
    return 'Invalid response: '+','.join(errors)+'. Each parameter must be ONE NUMBER within the original allowed values, not a range, list or object. Return complete JSON with parameters,reason,evidence_ids,expected_effect,risk; each explanation 1..120 characters. Cite latest_required_id.'


def reply_schema(feedback):
    validate_feedback(feedback)
    params={'learning_rate':dict(type='number',minimum=.0001,maximum=.002)}
    params.update({k:dict(type='number',enum=v) for k,v in CATEGORIES.items()})
    properties=dict(parameters=dict(type='object',properties=params,required=list(params),additionalProperties=False),
        reason=dict(type='string',minLength=1,maxLength=120),
        evidence_ids=dict(type='array',items=dict(type='string',enum=[feedback['latest_required_id']]),minItems=1,maxItems=1),
        expected_effect=dict(type='string',minLength=1,maxLength=120),risk=dict(type='string',minLength=1,maxLength=120))
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


def training_config(base,p,epochs=2,seed=270001):
    validate_parameters(p)
    if type(epochs) is not int or epochs<1 or type(seed) is not int:raise ValueError('invalid_budget_or_seed')
    c=dict(base);updates=160*epochs
    c.update(epochs=epochs,seed=seed,assess_epochs=1,lr=p['learning_rate'],pressure_weight=p['pressure_weight'],
        physics_weight=p['physics_weight'],bc_weight=p['bc_weight'],warmup=round(p['warmup_fraction']*updates),ramp=max(1,round(.03125*updates)))
    return c
