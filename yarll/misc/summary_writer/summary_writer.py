class SummaryWriter:
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer
        self.running = False

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def add_scalar(self, name: str, value, step, **kwargs):
        raise NotImplementedError

    def flush():
        raise NotImplementedError
