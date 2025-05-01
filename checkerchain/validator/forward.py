# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import bittensor as bt
import numpy as np

from checkerchain.protocol import CheckerChainSynapse

from checkerchain.database.actions import (
    add_prediction,
    get_predictions_for_product,
    delete_a_product,
    db_get_unreviewd_products,
)
from checkerchain.validator.reward import get_rewards
from neurons.validator import Validator
from checkerchain.utils.checker_chain import fetch_products
from checkerchain.utils.config import IS_OWNER, STATS_SERVER_URL, JWT_SECRET
import requests
import json


async def forward(self: Validator):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    # miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    # miner_uids = [5]
    miner_uids = self.metagraph.uids.tolist()
    # The dendrite client queries the network.
    # define product id to get scores for

    # Fetch product data
    data = fetch_products()

    bt.logging.info(f"new products to send to miners: {data.unmined_products}")
    if len(data.reward_items):
        bt.logging.info(f"Products to score: {[r._id for r in data.reward_items]}")

    if len(data.unmined_products):
        queries = data.unmined_products  # Get product IDs from CheckerChain API
    else:
        unmined_db_products = db_get_unreviewd_products()
        queries = {p._id for p in unmined_db_products}
        bt.logging.info(f"Unmined products from DB: {queries}")

    responses = []
    # Query the miners if there are unmined products
    if len(queries):
        responses = await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=CheckerChainSynapse(query=queries),
            timeout=25,
            deserialize=True,
        )
        bt.logging.info(f"Received responses: {responses}")

        # Add all responses to the database predictions table
        for miner_uid, miner_predictions in zip(miner_uids, responses):
            for product_id, prediction in zip(queries, miner_predictions):
                add_prediction(product_id, miner_uid, prediction)
    else:
        bt.logging.info("No any products to send to miners.")

    # Get one product which has been reviewed and is ready to score.
    # Adjust the scores based on responses from miners.
    reward_product = None
    predictions = []
    miner_ids = miner_uids
    rewards = np.zeros_like(miner_uids, dtype=float)

    if data.reward_items:
        prediction_logs = []
        for reward_product in data.reward_items:
            product_predictions = get_predictions_for_product(reward_product._id) or []
            if not product_predictions:
                continue

            predictions = [p.prediction for p in product_predictions]
            prediction_miners = [p.miner_id for p in product_predictions]

            _rewards = get_rewards(self, reward_product, responses=predictions)
            bt.logging.info("Product ID: ", reward_product._id)
            bt.logging.info("Miners: ", prediction_miners)
            bt.logging.info("Rewards: ", _rewards)
            for miner_id, reward in zip(prediction_miners, _rewards):
                if reward is None:
                    continue
                try:
                    index = prediction_miners.index(miner_id)
                    prediction_score = predictions[index]
                    if not prediction_score:
                        bt.logging.warning(
                            f"Prediction score is None for miner {miner_id} and product {reward_product._id}"
                        )
                        continue
                    prediction_logs.append(
                        {
                            "productId": reward_product._id,
                            "productName": reward_product.name,
                            "predictionScore": prediction_score,
                            "actualScore": reward_product.trustScore,
                            "hotkey": self.metagraph.hotkeys[miner_id],
                            "coldkey": self.metagraph.coldkeys[miner_id],
                            "uid": miner_id,
                        }
                    )
                    rewards[miner_id] += reward
                except Exception as e:
                    bt.logging.error(
                        f"Error while processing product {reward_product._id}: {e}"
                    )
                    continue

        try:
            # You don't need to worry about this part of the code, it's for data collection for owners
            if IS_OWNER:
                import jwt

                token = jwt.encode(
                    {"sub": self.metagraph.coldkeys[0]},
                    JWT_SECRET,
                    algorithm="HS256",
                )
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                }
                bt.logging.info(f"{STATS_SERVER_URL}/prediction/create", "url:")
                result = requests.post(
                    f"{STATS_SERVER_URL}/prediction/create",
                    json=prediction_logs,
                    headers=headers,
                )
                if result.status_code != 201:
                    bt.logging.error(
                        f"Error sending data to stats server: {result.status_code}"
                    )
                else:
                    bt.logging.info("Successfully sent data to stats server")
        except Exception as e:
            bt.logging.error("Error while sending data to stats server", e)
            bt.logging.error(prediction_logs)

        bt.logging.info(f"Scored responses: {rewards}")
        bt.logging.info(f"Score ids: {miner_ids}")
    # Ensure update_scores is always called with valid values

    # If a product was processed, delete it from the database
    if len(data.reward_items):
        self.update_scores(rewards, miner_ids)
        for reward_product in data.reward_items:
            delete_a_product(reward_product._id)
    else:
        self.update_to_last_scores()

    # TODO: One hour until next validation ??
    time.sleep(10 * 60)  # 10 minutes
