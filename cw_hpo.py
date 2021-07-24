from cw2 import experiment, cw_error
from cw2 import cluster_work
from cw2.cw_data import cw_logging
import hpo

class MyExperiment(experiment.AbstractExperiment):
    
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        hpo.run_study(config)
    
    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass

if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)
    # RUN!
    cw.run()