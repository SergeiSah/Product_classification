import pandas as pd


class Timer:

    def __init__(self):
        self.start = None
        self.end = None

        self.last_period = None

    def __enter__(self):
        self.start = pd.Timestamp.now()
        return self

    def __exit__(self, *args):
        self.end_timer()

    def start_timer(self):
        self.start = pd.Timestamp.now()

    def end_timer(self):
        self.end = pd.Timestamp.now()
        self.last_period = self.end - self.start
