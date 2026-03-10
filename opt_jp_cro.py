import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import metrics
import Rie_prob_2D_rs

def evaluate_jp_pair(jp_lo, jp_hi, grid_dict, control_dict, phys_dict):
    
    # enforce constraint
    if jp_lo >= jp_hi:
        return 1e6
    
    control = control_dict.copy()
    control["jp_cri"] = (jp_lo, jp_hi)
    control["mode"] = "opt"
    control["file_storage"] = False  # no disk writing during tuning
    
    try:
        q_final = main_rs(grid_dict, control, phys_dict)
    except Exception:
        return 1e6  # crash penalty
    
    # extract primitive variables from q_final
    gamma = phys_dict["gamma"]
    nx = grid_dict["nx"]
    ny = grid_dict["ny"]

    rho = q_final[3:3+nx,3:3+ny,0]
    u   = q_final[3:3+nx,3:3+ny,5]
    v   = q_final[3:3+nx,3:3+ny,6]
    p   = q_final[3:3+nx,3:3+ny,7]

    score, _ = metrics.compute_2d_objective(rho,u,v,p)

    print(f"jp_lo={jp_lo:.4f}, jp_hi={jp_hi:.4f}, score={score:.6e}")

    return score


def optimize_jp(grid_dict, control_dict, phys_dict):

    # search space
    space = [
        Real(0.02, 1, name='jp_lo'),
        Real(0.2, 10, name='jp_hi')
    ]

    @use_named_args(space)
    def objective(**params):
        return evaluate_jp_pair(
            params["jp_lo"],
            params["jp_hi"],
            grid_dict,
            control_dict,
            phys_dict
        )

    result = gp_minimize(
        objective,
        space,
        n_calls=30,        # number of simulations
        n_initial_points=8,
        random_state=0,
        acq_func="EI"
    )

    best_jp_lo = result.x[0]
    best_jp_hi = result.x[1]
    best_score = result.fun

    print("\nOptimal jp_cri found:")
    print(f"jp_lo = {best_jp_lo:.6f}")
    print(f"jp_hi = {best_jp_hi:.6f}")
    print(f"score = {best_score:.6e}")

    return best_jp_lo, best_jp_hi, best_score


# ----------------------------------------
# Example run
# ----------------------------------------
if __name__ == "__main__":

    from Rie_prob_2D_rs import grid_dict, control_dict, phys_dict

    control_dict["t_final"] = 0.05   # short run for tuning
    control_dict["file_storage"] = False
    control_dict["mode"] = "opt"

    lo,hi,bs = optimize_jp(grid_dict, control_dict, phys_dict)
    print(lo)
    print(hi)
    print(bs)