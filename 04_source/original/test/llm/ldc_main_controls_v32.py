"""Replay sampler state exactly after every actual completed attempt."""
from ldc_controls_v30 import RandomSearch, TPESearch
from ldc_main_contract_v32 import canonical

FAILURE_OBJECTIVE=1e6


def objective(row):
    return row['score'] if row['status']=='COMPLETE' else FAILURE_OBJECTIVE


def propose_control(method, seed, history):
    if not history or history[0]['status']!='COMPLETE':
        raise ValueError('Valid measured common start required')
    sampler=RandomSearch(seed) if method=='random' else TPESearch(seed,[dict(parameters=history[0]['parameters'],score=history[0]['score'])])
    for row in history[1:]:
        generated=sampler.ask()
        if canonical(generated)!=canonical(row['parameters']):
            raise ValueError('Sampler replay does not match actual past proposal')
        sampler.tell(generated,objective(row))
    parameters=sampler.ask()
    return parameters,dict(replayed_trials=len(history),sampler_seed=seed,
        failure_objective=FAILURE_OBJECTIVE,failure_objective_is_not_a_measurement=True)
