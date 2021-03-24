from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm

class EvalHookOnStart(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.
    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_function):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        
        self._func = eval_function

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()
        
    def before_train(self):
        self._do_eval()
