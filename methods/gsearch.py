"""
Perform a grid search for libfm parameter values.

"""
import scipy as sp

from libfm import libfm_model
from scaffold import *


def eval_params(data, method, std, iter, dim, fbias, gbias, task, r0, r1, r2,
                lrate, target, cvals=None, rvals=None, previous=False):
    """Get total RMSE for all terms using a certain parameter setting.

    This function is for use in grid searches.
    """
    def eval_fm(train, test):
        return libfm_model(
            train, test, method=method, std=std, iter=iter, dim=dim,
            fbias=fbias, gbias=gbias, task=task, lrate=lrate, r0=r0, r1=r1,
            r2=r2, target=target, cvals=cvals, rvals=rvals, previous=previous)

    results = method_error(data, eval_fm, False)
    evaluation = eval_results(results, by='termnum')
    return evaluation.ix['all']['rmse']


def mcmc_objective(params, *args):
    std, dim = params
    data, iter, fbias, gbias, target, cvals, rvals, previous = args
    return eval_params(
        data, 'mcmc', std, iter, dim, fbias, gbias, 'r', 0.0, 0.0, 0.0, 0.0,
        target, cvals, rvals, previous)

def als_objective(params, *args):
    std, dim, r0, r1, r2 = params
    data, iter, fbias, gbias, target, cvals, rvals, previous = args
    return eval_params(
        data, 'als', std, iter, dim, fbias, gbias, 'r', r0, r1, r2, 0.0, target,
        cvals, rvals, previous)


def make_parser():
    parser = base_parser('Grid search for LibFM parameter values')
    return parser


# TODO: replace use of scipy.optimize.brute with custom multiprocessing grid
# search function. Lift code from scipy and refine using `multiprocessing`
# module.
if __name__ == "__main__":
    args = setup(make_parser)
    data = pd.read_csv(args.data_file).sort(['sid', 'termnum'])

    # params = (slice(0, 1, 0.05), slice(1,11,1))
    # args = (data, 100, 1, 1, 'grdpts', None, None, False)
    # result = sp.optimize.brute(mcmc_objective, params, args=args, finish=None)

    reg_grid = slice(0, 1, 0.25)
    params = (slice(0.1,1,0.1),         # std
              slice(1,5,1),             # dim
              slice(0.15, 0.35, 0.05),  # r0
              slice(0.35, 0.7, 0.05),   # r1
              slice(0.6, 1.05, 0.05))   # r2
    # params = (slice(0.45,0.46,0.01),
    #           slice(1,6,1),
    #           slice(0.25, 0.26, 0.01),
    #           slice(0.5, 0.51, 0.01),
    #           slice(0.75, 0.76, 0.01))
    args = (data, 100, 1, 1, 'grdpts', None, None, False)
    result = sp.optimize.brute(als_objective, params, args=args, finish=None)
    print result

