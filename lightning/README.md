```bash
conda env create -f environment.yml
```

Activación del ambiente:

```bash
conda activate lightning
```

Run tensorboard and view graphs

```bash
tensorboard --logdir lightning/board/
tensorboard --logdir ../../board
```

