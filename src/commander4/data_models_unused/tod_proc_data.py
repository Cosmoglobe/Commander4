from .band import Band

class TodProcData:
    def __init__(self, inp: tuple[Band, ...]):
        self._data = inp

    @property
    def bands(self) -> tuple[Band, ...]:
        return self._data
