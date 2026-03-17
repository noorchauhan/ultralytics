# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np

from ultralytics.cfg import TASK2DATA, TASK2METRIC, get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, NUM_THREADS, checks, colorstr


RAY_SEARCH_ALG_REQUIREMENTS = {
    "random": None,
    "ax": "ax-platform",
    "bayesopt": "bayesian-optimization==1.4.3",
    "bohb": ["hpbandster", "ConfigSpace"],
    "hebo": "HEBO>=0.2.0",
    "hyperopt": "hyperopt",
    "nevergrad": "nevergrad",
    "optuna": "optuna",
    "zoopt": "zoopt",
}


def _sanitize_tune_value(value):
    """Convert NumPy-backed Tune values into native Python types for YAML serialization."""
    if isinstance(value, dict):
        return {k: _sanitize_tune_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_tune_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_tune_value(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _create_ax_search(space, task):
    """Create an Ax searcher with an initialized experiment."""
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["ax"])

    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties
    from ray.tune.search.ax.ax_search import AxSearch

    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=AxSearch.convert_search_space(space),
        objectives={TASK2METRIC[task]: ObjectiveProperties(minimize=False)},
    )
    return AxSearch(ax_client=ax_client)


def _create_bohb_search(space, task):
    """Create a BOHB searcher using a ConfigSpace definition compatible with current ConfigSpace versions."""
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["bohb"])

    import ConfigSpace
    from ray.tune.search.bohb.bohb_search import TuneBOHB
    from ray.tune.search.sample import Categorical, Float, Integer, LogUniform, Quantized, Uniform
    from ray.tune.search.variant_generator import parse_spec_vars
    from ray.tune.utils import flatten_dict

    resolved_space = flatten_dict(space, prevent_delimiter=True)
    resolved_vars, domain_vars, grid_vars = parse_spec_vars(resolved_space)
    if grid_vars:
        raise ValueError("Grid search parameters cannot be automatically converted to a TuneBOHB search space.")

    cs = ConfigSpace.ConfigurationSpace()
    for path, domain in domain_vars:
        par = "/".join(str(p) for p in path)
        sampler = domain.get_sampler()
        if isinstance(sampler, Quantized):
            raise ValueError("TuneBOHB does not support quantized search spaces with the current ConfigSpace version.")

        if isinstance(domain, Float) and isinstance(sampler, (Uniform, LogUniform)):
            cs.add(
                ConfigSpace.UniformFloatHyperparameter(
                    par, lower=domain.lower, upper=domain.upper, log=isinstance(sampler, LogUniform)
                )
            )
        elif isinstance(domain, Integer) and isinstance(sampler, (Uniform, LogUniform)):
            upper = domain.upper - 1  # Tune integer search spaces are exclusive on the upper bound
            cs.add(
                ConfigSpace.UniformIntegerHyperparameter(
                    par, lower=domain.lower, upper=upper, log=isinstance(sampler, LogUniform)
                )
            )
        elif isinstance(domain, Categorical) and isinstance(sampler, Uniform):
            cs.add(ConfigSpace.CategoricalHyperparameter(par, choices=domain.categories))
        else:
            raise ValueError(
                f"TuneBOHB does not support parameters of type {type(domain).__name__} "
                f"with sampler type {type(domain.sampler).__name__}."
            )

    fixed_param_space = {"/".join(str(p) for p in path): value for path, value in resolved_vars}
    return TuneBOHB(space=cs, metric=TASK2METRIC[task], mode="max"), fixed_param_space


def _create_nevergrad_search(task):
    """Create a Nevergrad searcher with a default optimizer."""
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["nevergrad"])

    import nevergrad as ng
    from ray.tune.search.nevergrad import NevergradSearch

    return NevergradSearch(optimizer=ng.optimizers.OnePlusOne, metric=TASK2METRIC[task], mode="max")


def _create_zoopt_search(space, task, max_samples):
    """Create a ZOOpt searcher with required budget and converted search space."""
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["zoopt"])

    from ray.tune.search.variant_generator import parse_spec_vars
    from ray.tune.search.zoopt import ZOOptSearch
    from ray.tune.utils import flatten_dict

    resolved_space = flatten_dict(space, prevent_delimiter=True)
    resolved_vars, _, _ = parse_spec_vars(resolved_space)
    fixed_param_space = {"/".join(str(p) for p in path): value for path, value in resolved_vars}
    dim_dict = ZOOptSearch.convert_search_space(space)
    return ZOOptSearch(algo="asracos", budget=max_samples, dim_dict=dim_dict, metric=TASK2METRIC[task], mode="max"), fixed_param_space


def _resolve_ray_search_alg(search_alg, task, space, max_samples):
    """Resolve string search algorithm aliases into Ray Tune searcher objects."""
    if search_alg is None or not isinstance(search_alg, str):
        return search_alg, space

    normalized = search_alg.strip().lower()
    if not normalized:
        return None, space

    if normalized not in RAY_SEARCH_ALG_REQUIREMENTS:
        supported = ", ".join(sorted(RAY_SEARCH_ALG_REQUIREMENTS))
        raise ValueError(f"Unsupported Ray Tune search_alg '{search_alg}'. Supported values: {supported}.")

    if normalized == "random":
        return None, space

    try:
        if normalized == "ax":
            return _create_ax_search(space, task), {}
        if normalized == "bohb":
            return _create_bohb_search(space, task)
        if normalized == "nevergrad":
            return _create_nevergrad_search(task), space
        if normalized == "zoopt":
            return _create_zoopt_search(space, task, max_samples)

        requirements = RAY_SEARCH_ALG_REQUIREMENTS[normalized]
        if requirements:
            checks.check_requirements(requirements)

        from ray.tune.search import create_searcher

        return create_searcher(normalized, metric=TASK2METRIC[task], mode="max"), space
    except (ImportError, ModuleNotFoundError) as e:
        raise ModuleNotFoundError(
            f"Ray Tune search_alg '{search_alg}' requires additional dependencies. Original error: {e}"
        ) from e


def run_ray_tune(
    model,
    space: dict | None = None,
    grace_period: int = 10,
    gpu_per_trial: int | None = None,
    max_samples: int = 10,
    search_alg=None,
    **train_args,
):
    """Run hyperparameter tuning using Ray Tune.

    Args:
        model (YOLO): Model to run the tuner on.
        space (dict, optional): The hyperparameter search space. If not provided, uses default space.
        grace_period (int, optional): The grace period in epochs of the ASHA scheduler.
        gpu_per_trial (int, optional): The number of GPUs to allocate per trial.
        max_samples (int, optional): The maximum number of trials to run.
        search_alg (str | ray.tune.search.Searcher | ray.tune.search.SearchAlgorithm, optional): Search algorithm
            to use. Strings are resolved to supported Ray Tune searchers, while objects are passed through as-is.
        **train_args (Any): Additional arguments to pass to the `train()` method.

    Returns:
        (ray.tune.ResultGrid): A ResultGrid containing the results of the hyperparameter search.

    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")  # Load a YOLO26n model

        Start tuning hyperparameters for YOLO26n training on the COCO8 dataset
        >>> result_grid = model.tune(data="coco8.yaml", use_ray=True)
    """
    LOGGER.info("💡 Learn about RayTune at https://docs.ultralytics.com/integrations/ray-tune")
    try:
        checks.check_requirements("ray[tune]")

        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
        from ray.tune import RunConfig
    except ImportError:
        raise ModuleNotFoundError('Ray Tune required but not found. To install run: pip install "ray[tune]"')

    try:
        import wandb

        assert hasattr(wandb, "__version__")
    except (ImportError, AssertionError):
        wandb = False

    checks.check_version(ray.__version__, ">=2.0.0", "ray")
    default_space = {
        # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
        "momentum": tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
        "weight_decay": tune.uniform(0.0, 0.001),  # optimizer weight decay
        "warmup_epochs": tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
        "warmup_momentum": tune.uniform(0.0, 0.95),  # warmup initial momentum
        "box": tune.uniform(0.02, 0.2),  # box loss gain
        "cls": tune.uniform(0.2, 4.0),  # cls loss gain (scale with pixels)
        "hsv_h": tune.uniform(0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        "hsv_s": tune.uniform(0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        "hsv_v": tune.uniform(0.0, 0.9),  # image HSV-Value augmentation (fraction)
        "degrees": tune.uniform(0.0, 45.0),  # image rotation (+/- deg)
        "translate": tune.uniform(0.0, 0.9),  # image translation (+/- fraction)
        "scale": tune.uniform(0.0, 0.9),  # image scale (+/- gain)
        "shear": tune.uniform(0.0, 10.0),  # image shear (+/- deg)
        "perspective": tune.uniform(0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        "flipud": tune.uniform(0.0, 1.0),  # image flip up-down (probability)
        "fliplr": tune.uniform(0.0, 1.0),  # image flip left-right (probability)
        "bgr": tune.uniform(0.0, 1.0),  # swap RGB↔BGR channels (probability)
        "mosaic": tune.uniform(0.0, 1.0),  # image mosaic (probability)
        "mixup": tune.uniform(0.0, 1.0),  # image mixup (probability)
        "cutmix": tune.uniform(0.0, 1.0),  # image cutmix (probability)
        "copy_paste": tune.uniform(0.0, 1.0),  # segment copy-paste (probability)
    }

    # Put the model in ray store
    task = model.task
    model_in_store = ray.put(model)
    base_name = train_args.get("name", "tune")

    def _tune(config):
        """Train the YOLO model with the specified hyperparameters and return results."""
        model_to_train = ray.get(model_in_store)  # get the model from ray store for tuning
        model_to_train.trainer = None
        model_to_train.reset_callbacks()
        config = _sanitize_tune_value(dict(config))
        config.update(train_args)

        # Set trial-specific name for W&B logging
        try:
            trial_id = tune.get_trial_id()  # Get current trial ID (e.g., "2c2fc_00000")
            trial_suffix = trial_id.split("_")[-1] if "_" in trial_id else trial_id
            config["name"] = f"{base_name}_{trial_suffix}"
        except Exception:
            # Not in Ray Tune context or error getting trial ID, use base name
            config["name"] = base_name

        results = model_to_train.train(**config)
        return results.results_dict

    # Get search space
    if not space and not train_args.get("resume"):
        space = default_space
        LOGGER.warning("Search space not provided, using default search space.")

    # Get dataset
    data = train_args.get("data", TASK2DATA[task])
    space["data"] = data
    if "data" not in train_args:
        LOGGER.warning(f'Data not provided, using default "data={data}".')

    resolved_search_alg, tuner_param_space = _resolve_ray_search_alg(search_alg, task, space, max_samples)

    # Define the trainable function with allocated resources
    trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 0})

    # Define the scheduler for hyperparameter search
    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=TASK2METRIC[task],
        mode="max",
        max_t=train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100,
        grace_period=grace_period,
        reduction_factor=3,
    )
    if isinstance(search_alg, str) and search_alg.strip().lower() == "bohb":
        scheduler = HyperBandForBOHB(
            time_attr="epoch",
            metric=TASK2METRIC[task],
            mode="max",
            max_t=train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100,
            reduction_factor=3,
        )

    # Create the Ray Tune hyperparameter search tuner
    tune_dir = get_save_dir(
        get_cfg(
            DEFAULT_CFG,
            {**train_args, **{"exist_ok": train_args.pop("resume", False)}},  # resume w/ same tune_dir
        ),
        name=train_args.pop("name", "tune"),  # runs/{task}/{tune_dir}
    )  # must be absolute dir
    tune_dir.mkdir(parents=True, exist_ok=True)
    if tune.Tuner.can_restore(tune_dir):
        LOGGER.info(f"{colorstr('Tuner: ')} Resuming tuning run {tune_dir}...")
        tuner = tune.Tuner.restore(str(tune_dir), trainable=trainable_with_resources, resume_errored=True)
    else:
        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=tuner_param_space,
            tune_config=tune.TuneConfig(
                search_alg=resolved_search_alg,
                scheduler=scheduler,
                num_samples=max_samples,
                trial_name_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
                trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
            ),
            run_config=RunConfig(storage_path=tune_dir.parent, name=tune_dir.name),
        )

    # Run the hyperparameter search
    tuner.fit()

    # Get the results of the hyperparameter search
    results = tuner.get_results()

    # Shut down Ray to clean up workers
    ray.shutdown()

    return results
