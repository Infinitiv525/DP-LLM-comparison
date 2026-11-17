import math
import random
from ldc_contract_v30 import CATEGORIES,validate_parameters


class RandomSearch:
    def __init__(self,seed):self.rng=random.Random(seed)
    def ask(self):
        p=dict(learning_rate=math.exp(self.rng.uniform(math.log(.0001),math.log(.002))))
        p.update({k:self.rng.choice(v) for k,v in CATEGORIES.items()});return validate_parameters(p)
    def tell(self,p,score):validate_parameters(p)


class TPESearch:
    def __init__(self,seed,history):
        import optuna
        self.optuna=optuna;optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=1))
        self.distributions=dict(learning_rate=optuna.distributions.FloatDistribution(.0001,.002,log=True))
        self.distributions.update({k:optuna.distributions.CategoricalDistribution(v) for k,v in CATEGORIES.items()})
        for r in history:
            self.study.add_trial(optuna.trial.create_trial(params=validate_parameters(r['parameters']),distributions=self.distributions,value=r['score']))
        self.pending=None
    def ask(self):
        if self.pending is not None:raise ValueError('TPE pending result')
        self.pending=self.study.ask();p={}
        for k,d in self.distributions.items():
            p[k]=self.pending.suggest_float(k,d.low,d.high,log=True) if k=='learning_rate' else self.pending.suggest_categorical(k,d.choices)
        return validate_parameters(p)
    def tell(self,p,score):
        if self.pending is None or p!=self.pending.params:raise ValueError('TPE ask/tell mismatch')
        if not math.isfinite(score):raise ValueError('Finite score required')
        self.study.tell(self.pending,score);self.pending=None
    def export(self):
        return dict(optuna=self.optuna.__version__,trials=[dict(number=t.number,parameters=t.params,score=t.value,state=t.state.name) for t in self.study.trials])
