
import os, glob

# import eda_read_csv
# print("a", eda_read_csv.__name__)
__all__ = [
    os.path.split(os.path.splitext(file)[0])[1]
    for file in glob.glob(os.path.join(os.path.dirname(__file__), '[a-zA-Z0-9]*.py'))
]