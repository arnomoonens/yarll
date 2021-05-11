from tensorflow import summary
from yarll.misc.summary_writer.summary_writer import SummaryWriter as SWBase

class SummaryWriterTensorflow(SWBase):
    def start(self):
        self.summary_writer.set_as_default()
        self.running = True

    def stop(self):
        self.summary_writer.close()
        self.running = False

    def add_scalar(self, name: str, value, step, **kwargs):
        assert self.running, "Summary writer has not been started yet."
        summary.scalar(name, value, step, **kwargs)

    def flush(self):
        self.summary_writer.flush()
