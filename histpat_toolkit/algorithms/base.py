from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar

ConfigT = TypeVar("ConfigT")


@dataclass
class AlgorithmConfig:
    pass


@dataclass
class AlgorithmInput:
    pass


class DataLoader(ABC, Generic[ConfigT]):
    def __init__(self, config: ConfigT):
        self.config = config

    @abstractmethod
    def load_data(self, input_data: Any) -> Dict[str, Any]:
        pass


class DataSaver(ABC):
    @abstractmethod
    def save_data(self, result: Any, input_data: Any, config: Any) -> Dict[str, Any]:
        pass


class AlgorithmPipeline(Generic[ConfigT]):
    def __init__(self, loader: DataLoader[ConfigT], saver: DataSaver, algorithm_runner: callable):
        self.loader = loader
        self.saver = saver
        self.algorithm_runner = algorithm_runner

    def run(self, input_data: AlgorithmInput) -> Dict[str, Any]:
        loaded_data = self.loader.load_data(input_data)
        result = self.algorithm_runner(**loaded_data)

        return self.saver.save_data(result, input_data, self.loader.config)
