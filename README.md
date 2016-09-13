Rubicon
=======

Evolutionary Rubik's Cube solver attempt, made as classwork for a Natural
Computing course. Not yet publicly available.

The documentation (in Portuguese) for this project is available at `doc/doc.pdf`.

Requirements
------------

- Python 3.5 or higher
- NumPy

Optionally, for plotting stats:

- Matplotlib
- Seaborn

Usage
-----

`python3 rubicon/ config/some_config.json`

Downloading the experiments
---------------------------

Due to the exceedingly large size of the experiment logs, I was unable to
provide them together with the rest of the code and files. You may still get
these files by downloading and extracting the `.tar.xz` archive containing the
run logs:

```
wget https://dl.dropboxusercontent.com/u/17792073/rubicon_runs.tar.xz
tar xJfv rubicon_runs.tar.xz
```

or, directly,

```
wget -qO- https://dl.dropboxusercontent.com/u/17792073/rubicon_runs.tar.xz | tar xJv
```

Or just download [rubicon\_runs.tar.xz](https://dl.dropboxusercontent.com/u/17792073/rubicon_runs.tar.xz)
directly and extract it manually. You'll need XZ or some other tool capable of
extracting `.xz` archives.
