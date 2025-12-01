# Import the W&B Python Library and log into W&B
import multiprocessing

import wandb


# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score


def main():
    with wandb.init(project="sweep_debug") as run:
        score = objective(run.config)
        run.log({"score": score})


def run_agent(sweep_id: str, count: int = 1):
    wandb.agent(
        sweep_id,
        function=main,
        entity="leosanitt-university-of-cambridge",
        project="sweep_debug",
        count=count,
    )


if __name__ == "__main__":
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "score"},
        "parameters": {
            "x": {"max": 0.1, "min": 0.01},
            "y": {"values": [1, 3, 7]},
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        entity="leosanitt-university-of-cambridge",
        project="sweep_debug",
    )

    num_workers = 10

    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=run_agent, args=(sweep_id,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("All processes finished.")
