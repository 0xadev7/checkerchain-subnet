from __future__ import annotations
import os, sys, json

from checkerchain.rlhf.db_mongo import RLHFMongo
from checkerchain.rlhf.job import run_training_tick


def main():
    db = RLHFMongo()
    res = run_training_tick(
        db,
        lr=float(os.getenv("RLHF_LR", "0.05")),
        lambda_stability=float(os.getenv("RLHF_LAMBDA", "0.01")),
        huber_delta=float(os.getenv("RLHF_HUBER_DELTA", "0.75")),
        half_life_days=int(os.getenv("RLHF_HALF_LIFE_DAYS", "14")),
        pgd_steps=int(os.getenv("RLHF_PGD_STEPS", "3")),
    )
    print(json.dumps(res))


if __name__ == "__main__":
    main()
