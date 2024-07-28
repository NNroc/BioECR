import torch
from typing import Iterator, Optional, TypeVar, Generic, Sized

T_co = TypeVar('T_co', covariant=True)


class Sampler(Generic[T_co]):
    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError


class CustomRandomSampler(Sampler[int]):
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False, num_samples: Optional[int] = None,
                 generator=None, train_batch_size=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.bs = train_batch_size

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        # making balance
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        random_list = []
        flag = 1
        while flag:
            random_list = torch.randperm(n, generator=generator).tolist()
            chunked_list = [random_list[i:i + self.bs] for i in range(0, len(random_list), self.bs)]
            flag = 0
            for chunk in chunked_list:
                total_lengh = 0
                for c in chunk:
                    total_lengh += len(self.data_source[c]['spans'])
                if total_lengh >= 150:
                    flag = 1
                    break
        yield from random_list

    def __len__(self) -> int:
        return self.num_samples
