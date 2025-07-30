import sys
import os


def cut(filepath, basepath):
   filepath = os.path.abspath(filepath)
   with open(filepath, 'r') as f:
      text = f.read()

   lines = text.split('\n')
   questions = [line[2:].strip() for line in lines if line.startswith('A:')]
   rpath = filepath[len(basepath):]
   ret = [rpath]
   for q in questions:
      ret.append('"{}"'.format(q.replace('"', '"\'"\'"')))
   print(' '.join(ret))

if __name__ == '__main__':
   basepath = os.path.abspath(sys.argv[1])
   filepaths = sys.argv[2:]
   for f in filepaths:
      cut(f, basepath)
