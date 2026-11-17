"""Standard-library-only, bounded steady PINN contract; no dataset access."""
import hashlib
import json
import math

VERSION=28
CATEGORIES=dict(pressure_weight=[.5,1.,2.],physics_weight=[.0001,.0003,.001,.003,.01],
    bc_weight=[.001,.003,.01,.03,.1],warmup_fraction=[0.,.03125,.0625])
PARAMETERS={'learning_rate',*CATEGORIES}
BASELINE=dict(learning_rate=.001,pressure_weight=1.,physics_weight=.001,bc_weight=.01,warmup_fraction=.03125)
KEYS={'parameters','reason','evidence_ids','expected_effect','risk'}
SYSTEM=(
    'Configure training of ONE shared steady 2D lid-driven-cavity PINN on FlowBench harmonics. '
    'Inputs are obstacle geometry and Reynolds number; outputs are steady u,v,p, not time rollout. '
    'The fixed geometry-conditioned coordinate model predicts streamfunction/pressure; velocity is its curl. '
    'Loss = mean([du^2,dv^2,pressure_weight*dp^2]/TRAIN_std^2) + physics_weight*steady_NS_loss + bc_weight*no_slip_loss. '
    'The PDE includes convection, pressure gradient, viscosity 1/Re and continuity. '
    'All candidates use the SAME architecture, 640 TRAIN/160 VAL split, optimizer, training seed, '
    '2048 data/64 PDE/128 boundary points per case and 4 cases per update. No TEST values are available. '
    'Only the five permitted parameters may change. Physics weight must remain positive. '
    'Adam uses cosine decay to 5% of its initial learning rate; PDE warmup is warmup_fraction of all updates, '
    'followed by a fixed 0.03125-fraction linear ramp. Normalize each field using TRAIN-only std. '
    'Minimize validation mean unweighted all-cell u/v/p MSE after predicted pressure gauge and known solid masking. '
    'Velocity/pressure relative L2 and finite-difference NS/divergence are additional reported diagnostics; '
    'low divergence alone does not establish physical accuracy. Require full 512x512 p99 inference <=5ms on H200. '
    'This is a TWO-EPOCH TECHNICAL WIRING TEST, not a converged accuracy comparison or an independent repetition. '
    'The supplied history contains genuine measured two-epoch trials; do not invent measurements. '
    'No tools, code execution, internet, files or additional data access. '
    'Return ONE JSON object with exactly keys parameters,reason,evidence_ids,expected_effect,risk. '
    'parameters must contain exactly learning_rate,pressure_weight,physics_weight,bc_weight,warmup_fraction. '
    'learning_rate real [0.0001,0.002]; pressure_weight one of [0.5,1,2]; '
    'physics_weight one of [0.0001,0.0003,0.001,0.003,0.01]; '
    'bc_weight one of [0.001,0.003,0.01,0.03,0.1]; warmup_fraction one of [0,0.03125,0.0625]. '
    'reason,expected_effect,risk must be nonempty brief public statements, at most 120 words combined. '
    'evidence_ids must be unique IDs of completed trials in the supplied history and include latest_required_id. '
    'Use actual supplied measurements in explanations and distinguish expectations from measured results. '
    'Duplicates consume an attempt and recorded resources. No Markdown or text outside JSON.'
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
        if sum(len(obj[k].split()) for k in ['reason','expected_effect','risk'])>120:raise ValueError('explanation_word_limit')
        ev=obj['evidence_ids']
        if not isinstance(ev,list) or not ev or any(not isinstance(v,str) or v not in ids for v in ev) or len(set(ev))!=len(ev):raise ValueError('evidence_ids')
        if feedback['latest_required_id'] not in ev:raise ValueError('must_cite_latest_actual_trial')
        return obj,[],transport
    except (ValueError,TypeError,KeyError) as e:return None,[str(e)],transport


def repair_message(errors):
    return 'Invalid response: '+','.join(errors)+'. Return the complete corrected JSON object with exactly parameters,reason,evidence_ids,expected_effect,risk. Obey all original ranges and cite latest_required_id. Do not change any measurements.'


def training_config(base,p,epochs=2,seed=270001):
    validate_parameters(p)
    if type(epochs) is not int or epochs<1 or type(seed) is not int:raise ValueError('invalid_budget_or_seed')
    c=dict(base);updates=160*epochs
    c.update(epochs=epochs,seed=seed,assess_epochs=1,lr=p['learning_rate'],pressure_weight=p['pressure_weight'],
        physics_weight=p['physics_weight'],bc_weight=p['bc_weight'],warmup=round(p['warmup_fraction']*updates),ramp=max(1,round(.03125*updates)))
    return c
