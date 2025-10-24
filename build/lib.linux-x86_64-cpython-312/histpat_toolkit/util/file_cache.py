from tempfile import TemporaryDirectory
from collections import defaultdict
import os
from threading import Lock
from typing import Any
from uuid import uuid4
from typing import Callable, Hashable


class FileCache:
  def __init__(self, path=None) -> None:
    if not path:
      self.tempdir = TemporaryDirectory()
      self.path = self.tempdir.name
    else:
      self.tempdir = None

    self.files: dict[Any, str | Exception] = {}
    self.locks = defaultdict(Lock)

    self.main_lock = Lock()

  def __del__(self) -> None:
    if self.tempdir:
      self.tempdir.cleanup()
      self.tempdir = None
    else:
      for file in self.files.values():
        os.remove(file)

  def get(self, fn: Callable[[], bytes], key: Hashable):
    with self.main_lock:
      key_lock = self.locks[key]
    
    with key_lock:
      if key not in self.files:
        try:
          content = fn()
          filename = os.path.join(self.path, str(uuid4()))
          with open(filename, 'wb') as f:
            f.write(content)
          self.files[key] = filename
        except Exception as e:
          self.files[key] = e

      result = self.files[key]
      if isinstance(result, Exception):
        raise result
      else:
        with open(result, 'rb') as f:
          return f.read()