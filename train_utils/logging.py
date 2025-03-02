import sys

class logging:
    def __init__(self, log_file=None):
        self.terminal = sys.stdout  # Store original stdout
        self.log = open(log_file, "w")  # Open log file for writing
        sys.stdout = self  # Redirect stdout

    def write(self, message):
        self.terminal.write(message)  # Print to console
        self.log.write(message)  # Write to file

    def flush(self):
        self.terminal.flush()
        self.log.flush()