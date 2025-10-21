import numpy as np, json
import bittensor as bt

from checkerchain.rlhf.db_mongo import RLHFMongo
from checkerchain.rlhf.batch_fit import fit_weights_bias_scale


def main():
    bt.logging.set_trace()
    db = RLHFMongo()
    rows = db.get_all_breakdowns_with_targets()  # [(x10, y100, ts)]
    if len(rows) < 50:
        bt.logging.info(json.dumps({"status": "not-enough-data", "n": len(rows)}))
        return
    X = np.stack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])

    w, b0, b1, obj = fit_weights_bias_scale(
        X, y, huber_delta=3.0, lambda_u_l2=1e-3, lambda_u_tv=0.0
    )
    # persist together
    db.save_weights(
        list(w),
        {"reason": "batch_fit", "b0": b0, "b1": b1, "obj": obj, "n": int(len(rows))},
    )
    bt.logging.info(
        json.dumps({"status": "ok", "b0": b0, "b1": b1, "obj": obj, "weights": list(w)})
    )


if __name__ == "__main__":
    main()
