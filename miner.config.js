module.exports = {
  apps: [
    {
      name: "checkerchain-miner",
      script:
        "python neurons/miner.py --netuid 87 --wallet.name develop --wallet.hotkey dev0 --logging.debug",
    },
  ],
};
