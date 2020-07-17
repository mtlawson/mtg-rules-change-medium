# mtg-rules-change-medium

Source code for the simulation analysis of a potential rules change in Magic: the Gathering. You can read my post about this analysis [here](https://medium.com/@michael.lawson_96765/would-changing-who-chooses-to-play-first-shorten-magic-the-gathering-tournaments-63247b9c9ad4).

The business end of things is located in simulate_magic_rounds.py. I've also uploaded the figures from the article, just in case you're curious.

This repository is open source. I make no claims for its quality, and in fact I can guarantee you it's pretty bad because it uses copy-paste instead of functionalizing things in quite a few places. You have the right to use and modify it as you see fit. If you use it to do something cool, I'd appreciate it if you let me know!

## Setup

1. Fork and clone the repository to your local machine

    ```bash
    git clone https://github.com/[your username]/mtg-rules-change-medium.git
    ```

2. Navigate to the repo directory

    ```bash
    cd [clone path]/mtg-rules-change-medium
    ```

3. Create conda environment to install required packages

    ```bash
    conda env create --file=environment.yaml
    ```

4. Activate the newly created environment

    ```bash
    conda activate mtg_sim
    ```

5. If you have issues with the environment after creation and want to start over, you can remove it with the following command

    ```bash
    conda env remove -n mtg_sim
    ```
