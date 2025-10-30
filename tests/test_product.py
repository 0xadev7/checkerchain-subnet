import asyncio
import sys
import bittensor as bt

from neurons.miner import Miner
import checkerchain
from checkerchain.miner.llm import generate_complete_assessment
from checkerchain.utils.checker_chain import fetch_product_data_with_score
from checkerchain.validator.reward import reward
from checkerchain.database.model import MinerPrediction


async def main(product_id: str):
    """
    Asynchronously fetch product data and generate complete assessments in parallel.
    Uses a single OpenAI request per product to generate score, review, and keywords.
    """
    bt.logging.info(f"Fetching product data for product_id={product_id}")

    product = fetch_product_data_with_score(product_id)
    if not product:
        bt.logging.error(f"No product found for product_id={product_id}")
        return

    bt.logging.info(f"Product:\n{product}")

    # Generate assessment
    response = await generate_complete_assessment(product, verbose=True)
    bt.logging.info(f"Response:\n{response}")

    # Reward process
    task = await reward(
        None,
        MinerPrediction(
            id=0,
            product_id=product_id,
            miner_id=0,
            prediction=response.get("score"),
            review=response.get("review"),
            keywords=response.get("keywords"),
        ),
        product.trustScore,
        0,
    )

    bt.logging.info(f"Reward:\n{task}")


if __name__ == "__main__":
    bt.logging.set_trace()

    if len(sys.argv) < 2:
        bt.logging.error("Usage: python script.py <product_id>")
        sys.exit(1)

    product_id = sys.argv[1]
    asyncio.run(main(product_id))
