import shutil
import os

dev_path = '/Users/applelaptop/dev/shortcuts'
os.makedirs(dev_path, exist_ok = True)
wd = os.getcwd()

os.chdir(wd)
scripts = os.listdir()
scripts = [s for s in scripts if s.endswith('.py') and s != 'sync.py']
for s in scripts:
    shutil.copyfile(
        s,
        os.path.join(dev_path, s)
    )
    print(f'successfully copied {s} to {dev_path}')