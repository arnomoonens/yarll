from yarll.misc.summary_writer.summary_writer import SummaryWriter as SWBase

class SummaryWriterPytorch(SWBase):
    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def add_scalar(self, name: str, value, step, **kwargs):
        assert self.running, "Summary writer has not been started yet."
        self.summary_writer.add_scalar(name, value, step, **kwargs)

    def flush(self):
        self.summary_writer.flush()
