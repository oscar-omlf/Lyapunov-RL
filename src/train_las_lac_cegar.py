# train_las_lac_cegar.py
import os
import numpy as np
import torch
import dreal as d

from agents.agent_factory import AgentFactory
from util.metrics_tracker import MetricsTracker
from util.logger_utils import setup_run_directory_and_logging
from util.dreal import extract_ce_from_model
from util.doa_utils import estimate_doa
from config import config_ldp_pendulum, config_ldp_vanderpol

CFG = config_ldp_pendulum
normalize_gradients = CFG["normalize_gradients"]

CFG["beta"] = 0.1
CFG["model_name"] = f"LDP-{CFG['beta']}"

R1_LB, R1_UB = CFG["r1_bounds"]
R2_LB, R2_UB = 2 * R1_LB, 2 * R1_UB

TRAINING_STEPS        = 1000
MAX_OUTER_LOOPS       = 80
MIN_REL_GAIN          = 0.01
PATIENCE              = 3
N_DOA_SAMPLES         = 50_000
CERTIFICATION_LEVEL_C = 0.4        

def bisect_c_star(check_fn, eps=0.25, delta=1e-5,
                  c_lo=0.01, c_hi=0.95, tol=0.01, max_iter=12):
    ce_model_fail, lo, hi = None, c_lo, c_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ok, ce = check_fn(level=mid, eps=eps, delta=delta)
        if ok:
            lo = mid
        else:
            hi = mid
            ce_model_fail = ce
        if hi - lo < tol:
            break
    return lo, ce_model_fail


def run_las_lac_cegar_training():
    tracker = MetricsTracker()
    run_dir, logger = setup_run_directory_and_logging(CFG)
    CFG["run_dir"] = run_dir

    agent  = AgentFactory.create_agent(config=CFG)
    device = agent.device  

    all_counter_examples, doa_hist = [], []
    total_actor_losses, total_critic_losses = [], []
    stagnant = 0

    logger.info("Sampling from R1!!")
    logger.info("Using W_glo only for the actor update.")
    logger.info("Starting CEGAR Training Loop...")
    logger.info(f"Initial LQR c* used by blending function: {agent.blending_function.c_star:.4f}")

    for outer in range(1, MAX_OUTER_LOOPS + 1):
        logger.info(f"CEGAR Iteration {outer}/{MAX_OUTER_LOOPS}")

        # learner phase
        logger.info(f"Starting Learner Phase ({TRAINING_STEPS} steps)...")
        logger.info(f"Current number of counter-examples: {len(all_counter_examples)}")

        for step in range(1, TRAINING_STEPS + 1):
            a_loss, c_loss = agent.update(counter_examples=all_counter_examples, normalize_gradients=normalize_gradients)
            total_actor_losses.append(a_loss)
            total_critic_losses.append(c_loss)

            if step % 10 == 0:
                logger.info(f"  Step {step:4d}: Actor Loss={a_loss:.4f}, Critic Loss={c_loss:.4f}")

        # verifier phase
        logger.info(f"Starting Falsifier Phase for composite system at c = {CERTIFICATION_LEVEL_C}...")
        c_star, ce_model = bisect_c_star(agent.trainer.check_lyapunov_with_ce)
        logger.info(f"    verified c* = {c_star:.3f}  (outer iteration {outer})")

        # DoA estimate in R2
        doa_frac = estimate_doa(
            level=c_star,
            lb=R2_LB, ub=R2_UB,
            critic_model=agent.critic_model,
            device=device,
            n_samples=N_DOA_SAMPLES,
        )
        logger.info(f"  DoA = {100*doa_frac:.2f} % of R2")

        doa_hist.append(doa_frac)
        if len(doa_hist) >= 2:
            rel_gain = (doa_hist[-1] - doa_hist[-2]) / max(doa_hist[-2], 1e-9)
            logger.info(f"  rel. gain {100*rel_gain:.2f} %")

            if c_star >= CERTIFICATION_LEVEL_C:
                stagnant = stagnant + 1 if rel_gain < MIN_REL_GAIN else 0
                agent.save(file_path=run_dir, episode=c_star)
                if stagnant >= PATIENCE:
                    logger.info("   early stop: DoA stalled")
                    break

        # add counter-example if present
        if ce_model:
            ce = extract_ce_from_model(ce_model, CFG["state_space"].shape[0])
            all_counter_examples.append(ce)
            logger.info(f"  added CE {np.round(ce,4)}")
        else:
            logger.info("   no counter-example beyond c*")

    else:
        logger.error("\nTRAINING FAILED: Max CEGAR iterations reached without full verification.")

    logger.info("Training Finished")
    tracker.add_run_losses(CFG["model_name"], total_actor_losses, total_critic_losses)
    tracker.save_top10_losses_plot(folder=run_dir)
    agent.save(file_path=run_dir, episode=outer * TRAINING_STEPS)


if __name__ == "__main__":
    run_las_lac_cegar_training()
