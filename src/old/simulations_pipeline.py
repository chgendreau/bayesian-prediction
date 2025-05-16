"""Pipeline function to lauch all comparisons"""
from src.simulations import ExchangeableSimulator, TMixtureSimulator


CONFIG = dict(
    # Simulation parameters
    seed=None,
    n_obs=10000,
    TMixtureConfig=dict(
        weights=[0.3, 0.1, 0.6],
        mus=[-5, 0, 4],
        nus=[3, 3, 3],
    ),
    config_name="april08",
)


def simulations_pipeline():
    #######################
    # Simulating 1D data
    #######################
    # Exchangeable data
    exchangeable_simulator = ExchangeableSimulator(
        k=1,
        n=CONFIG["n_obs"],
        seed=CONFIG["seed"],
    )
    exchangeable_simulator.simulate(save_name=f"data/{CONFIG['config_name']}_exchangeable_data.npy")

    # TMixture data
    tmixture_cofig = CONFIG["TMixtureConfig"]
    tmixture_simulator = TMixtureSimulator(
        weights=tmixture_cofig["weights"],
        mus=tmixture_cofig["mus"],
        nus=tmixture_cofig["nus"],
        n=CONFIG["n_obs"],
        seed=CONFIG["seed"],
    )
    tmixture_simulator.simulate(save_name=f"data/{CONFIG['config_name']}_tmixture_data.npy")


if __name__ == "__main__":
    simulations_pipeline()
