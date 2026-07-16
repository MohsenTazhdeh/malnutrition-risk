import optuna
from typing import Callable, Mapping, Any


def build_suggest_fn(search_space: Mapping[str, Any]) -> Callable[[optuna.Trial], dict]:
    params = search_space.get('params', {})

    def suggest_fn(trial: optuna.Trial) -> dict:
        out = dict()
        for name, spec in params.items():
            if spec['type'] == 'int':
                out[name] = trial.suggest_int(name, spec['low'], spec['high'])
            elif spec['type'] == 'float':
                out[name] = trial.suggest_float(name, spec['low'], spec['high'], log=spec.get('log', False))
            elif spec['type'] == 'categorical':
                out[name] = trial.suggest_categorical(name, spec['choices'])
            else:
                raise ValueError(f"Unknown search space type {spec['type']} for {name}")
        return out

    return suggest_fn

def get_warm_start_params(search_space: Mapping[str, Any]) -> dict[str, Any] | None:
    warm_start_params = search_space.get('warm_start')
    return dict(warm_start_params) if warm_start_params else None

