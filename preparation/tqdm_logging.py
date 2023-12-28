import logging
from tqdm import tqdm


class logging_tqdm(tqdm):
    def __init__(
            self,
            *args,
            logger: logging.Logger = None,
            mininterval: float = 1,
            bar_format: str = '{desc}{percentage:3.0f}%{r_bar}',
            desc: str = 'progress: ',
            **kwargs):
        self._logger = logger
        super().__init__(
            *args,
            mininterval=mininterval,
            bar_format=bar_format,
            desc=desc,
            **kwargs
        )

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        return logging.getLogger()

    def display(self, msg=None, pos=None):
        if not self.n:
            # skip progress bar before having processed anything
            return
        if not msg:
            msg = self.__str__()
        if self.total is None:
            self.logger.info('%s', msg)
        else:
            percentage = self.n / self.total
            if percentage >= 0.95:  # put in logging if the progress reach 95%
                self.logger.info('%s', msg)
            else:
                super().display(msg, pos)
