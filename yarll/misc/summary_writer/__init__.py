SUMMARY_WRITER = None

def set(sw):
    global SUMMARY_WRITER
    framework = "pytorch" if str(sw).startswith("<torch") else "tensorflow"
    if framework == "pytorch":
        from yarll.misc.summary_writer.summary_writer_pytorch import SummaryWriterPytorch
        SUMMARY_WRITER = SummaryWriterPytorch(sw)
    else:
        from yarll.misc.summary_writer.summary_writer_tensorflow import SummaryWriterTensorflow
        SUMMARY_WRITER = SummaryWriterTensorflow(sw)

def available():
    return SUMMARY_WRITER is not None

def start():
    if available():
        SUMMARY_WRITER.start()

def stop():
    if available():
        SUMMARY_WRITER.stop()

def add_scalar(name: str, value, step, **kwargs):
    if available():
        SUMMARY_WRITER.add_scalar(name, value, step, **kwargs)

def flush():
    if available():
        SUMMARY_WRITER.flush()
