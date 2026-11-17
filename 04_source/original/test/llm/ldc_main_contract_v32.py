"""Main and integration contracts share allowed settings, never TEST feedback."""
import hashlib
import math
from ldc_contract_v30 import CATEGORIES, BASELINE, validate_parameters, canonical, unique_pairs, repair_message
import json

VERSION = 32
HISTORY_KEYS = {'id', 'parameters', 'status', 'score', 'velocity_error', 'pressure_error',
                'momentum_rms', 'divergence_rms', 'boundary_error', 'training_seconds',
                'training_seed', 'epochs', 'error_codes'}
KEYS = {'parameters', 'reason', 'evidence_ids', 'expected_effect', 'risk'}


def validate_feedback(feedback):
    if set(feedback) != {'contract_version', 'phase', 'epochs', 'candidate_budget', 'remaining_attempts', 'latest_required_id', 'history'}:
        raise ValueError('feedback_keys')
    phase = feedback['phase']
    expected = {'integration': (2, 3), 'main': (400, 8)}
    if feedback['contract_version'] != VERSION or phase not in expected:
        raise ValueError('contract_phase')
    if (feedback['epochs'], feedback['candidate_budget']) != expected[phase]:
        raise ValueError('fixed_training_and_search_budget')
    history = feedback['history']
    if not isinstance(history, list) or not history:
        raise ValueError('empty_real_history')
    if type(feedback['remaining_attempts']) is not int or feedback['remaining_attempts'] != feedback['candidate_budget']-len(history):
        raise ValueError('remaining_attempts')
    if not 1 <= feedback['remaining_attempts'] < feedback['candidate_budget']:
        raise ValueError('attempt_limit')
    ids = []
    for row in history:
        if set(row) != HISTORY_KEYS or not isinstance(row['id'], str) or not row['id'] or row['id'] in ids:
            raise ValueError('history_keys_or_ids')
        ids.append(row['id'])
        if row['parameters'] is not None:
            validate_parameters(row['parameters'])
        if row['status'] not in ['COMPLETE', 'INVALID_PROPOSAL', 'TRAINING_FAILED', 'SPEED_FAILED']:
            raise ValueError('actual_history_status')
        if row['epochs'] != feedback['epochs'] or type(row['training_seed']) is not int:
            raise ValueError('history_training_budget')
        if not isinstance(row['error_codes'], list) or len(row['error_codes']) > 8 or any(not isinstance(x, str) or len(x)>200 for x in row['error_codes']):
            raise ValueError('bounded_error_codes')
        for key in ['score', 'velocity_error', 'pressure_error', 'momentum_rms', 'divergence_rms', 'boundary_error']:
            value = row[key]
            if row['status'] != 'COMPLETE':
                if value is not None:
                    raise ValueError('failed_trial_has_no_fabricated_metrics')
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('finite_real_metric_' + key)
        if isinstance(row['training_seconds'], bool) or not isinstance(row['training_seconds'], (int,float)) or not math.isfinite(row['training_seconds']) or row['training_seconds'] < 0:
            raise ValueError('real_training_time')
    if feedback['latest_required_id'] != ids[-1] or len({r['training_seed'] for r in history}) != 1:
        raise ValueError('latest_feedback_or_matched_search_seed')
    canonical(feedback)
    return set(ids)


def system_prompt(phase):
    if phase not in ['main', 'integration']:
        raise ValueError('phase')
    epochs, budget = (400, 8) if phase == 'main' else (2, 3)
    return (
        'Configure ONE shared steady 2D FlowBench LDC PINN: geometry/Re -> u,v,p. '
        f'This is {phase}: {epochs} full TRAIN epochs per candidate, {budget} total candidates including the supplied fixed start. '
        'Choose the next configuration using ONLY the supplied actual history; no TEST data. '
        'Minimize mean unweighted validation all-cell u/v/p MSE. Velocity/pressure L2, steady NS residual, '
        'divergence and boundary error are additional measured diagnostics. '
        'Loss: TRAIN-std-normalized data MSE (pressure_weight on p), plus physics_weight times steady '
        'incompressible Navier-Stokes loss with viscosity 1/Re, plus bc_weight times no-slip BC loss. '
        'Fixed own architecture,640TRAIN/160VAL,native512,batch4,2048data/64PDE/128BC points per case; '
        'Adam/cosine,gradient clip1. warmup_fraction delays PDE, then fixed0.03125-fraction ramp. '
        'The 5ms full-field GPU p99 requirement concerns inference, NOT training time or physical error. '
        'Failed attempts have null measured metrics and consume a candidate. Duplicate settings also consume '
        'a complete fresh training; explore a change when useful, but no forced change or manual replacement. '
        'Return ONE complete JSON object, no Markdown or commentary. Exactly parameters,reason,evidence_ids,expected_effect,risk. '
        'parameters exactly: learning_rate number[0.0001,0.002]; pressure_weight in[0.5,1,2]; '
        'physics_weight in[0.0001,0.0003,0.001,0.003,0.01]; bc_weight in[0.001,0.003,0.01,0.03,0.1]; '
        'warmup_fraction in[0,0.03125,0.0625]. Each parameter ONE NUMBER, not a range. '
        'reason,expected_effect,risk: each one short public statement of1..120characters. '
        'evidence_ids contains latest_required_id. Identical schema guidance for every model constrains syntax '
        'and routes the evidence label; it does not prove understanding. No tools,code,internet or invented results.'
    )


def messages(feedback):
    validate_feedback(feedback)
    return [{'role': 'system', 'content': system_prompt(feedback['phase'])},
            {'role': 'user', 'content': canonical(feedback)}]


def system_digest(phase):
    return hashlib.sha256(system_prompt(phase).encode()).hexdigest()


def parse_reply(raw, feedback):
    ids = validate_feedback(feedback)
    try:
        obj = json.loads(raw.strip(), object_pairs_hook=unique_pairs,
                         parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite_json')))
        if not isinstance(obj, dict) or set(obj) != KEYS:
            raise ValueError('exact_reply_keys')
        validate_parameters(obj['parameters'])
        for key in ['reason', 'expected_effect', 'risk']:
            if not isinstance(obj[key], str) or not obj[key].strip() or len(obj[key])>120:
                raise ValueError('compact_' + key)
        evidence = obj['evidence_ids']
        if not isinstance(evidence, list) or not evidence or any(not isinstance(x, str) or x not in ids for x in evidence) or len(set(evidence)) != len(evidence) or feedback['latest_required_id'] not in evidence:
            raise ValueError('latest_actual_evidence')
        return obj, [], 'strict_json'
    except (ValueError, TypeError, KeyError) as error:
        return None, [str(error)], 'strict_json'


def reply_schema(feedback):
    validate_feedback(feedback)
    parameters = {'learning_rate': dict(type='number', minimum=.0001, maximum=.002)}
    parameters.update({k: dict(type='number', enum=v) for k,v in CATEGORIES.items()})
    properties = dict(parameters=dict(type='object', properties=parameters, required=list(parameters), additionalProperties=False),
        reason=dict(type='string', minLength=1, maxLength=120),
        evidence_ids=dict(type='array', items=dict(type='string', enum=[feedback['latest_required_id']]), minItems=1, maxItems=1),
        expected_effect=dict(type='string', minLength=1, maxLength=120), risk=dict(type='string', minLength=1, maxLength=120))
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def training_config(base, parameters, phase, seed):
    validate_parameters(parameters)
    if phase not in ['main', 'integration'] or type(seed) is not int:
        raise ValueError('training_phase_seed')
    epochs = 400 if phase == 'main' else 2
    config = dict(base)
    config.update(epochs=epochs, seed=seed, assess_epochs=10 if phase=='main' else 1,
        lr=parameters['learning_rate'], pressure_weight=parameters['pressure_weight'],
        physics_weight=parameters['physics_weight'], bc_weight=parameters['bc_weight'],
        warmup=round(parameters['warmup_fraction']*160*epochs), ramp=max(1,round(.03125*160*epochs)))
    return config
