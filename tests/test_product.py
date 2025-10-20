import asyncio

from neurons.miner import Miner
import checkerchain
from checkerchain.miner.llm import (
    generate_complete_assessment,
)
from checkerchain.utils.checker_chain import fetch_product_data_with_score
from checkerchain.validator.reward import reward
import bittensor as bt

miner_preds = {}


async def main():
    """
    Asynchronously fetch product data and generate complete assessments in parallel.
    Uses a single OpenAI request per product to generate score, review, and keywords.
    """
    product_id = "684036c1a26d4eb551b68aff"
    bt.logging.info(f"Fetching product data for {product_id}")
    product = fetch_product_data_with_score(product_id)
    bt.logging.info("Product:\n", product)

    # Generate assessments for found products
    response = await generate_complete_assessment(product)
    bt.logging.info("Response:\n", response)

    task = reward(None, response, product.trustScore, 0)
    bt.logging.info("Reward:\n", task)


if __name__ == "__main__":
    asyncio.run(main())
