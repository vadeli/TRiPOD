from dataclasses import dataclass
import sys
import os

@dataclass
class Missing_mask:
    offset_train: list
    offset_test: list
    offset_validation: list
    pose_train: list
    pose_test: list
    pose_validation: list

class LoggingPrinter:
    def __init__(self, filename):
        self.out_file = open(filename, "a")
        self.old_stdout = sys.stdout
        #this object will take over `stdout`'s job
        sys.stdout = self
    #executed when the user does a `print`
    def write(self, text):
        self.old_stdout.write(text)
        self.out_file.write(text)
    #executed when `with` block begins
    def __enter__(self):
        return self
    #executed when `with` block ends
    def __exit__(self, type, value, traceback):
        #we don't want to log anymore. Restore the original stdout object.
        sys.stdout = self.old_stdout



class LoggerClass(object):
    """
    Lumberjack class - duplicates sys.stdout to a log file and it's okay
    source: https://stackoverflow.com/a/24583265/5820024
    """
    def __init__(self, filename="Red.Wood", mode="a", buff=0):
        self.stdout = sys.stdout
        self.file = open(filename, mode, buff)
        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None

        if self.file != None:
            self.file.close()
            self.file = None
