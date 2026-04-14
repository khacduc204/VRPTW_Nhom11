# Benchmark Data Layout

Place Solomon benchmark instance files in this folder:

- `data/solomon/C101.txt`
- `data/solomon/R101.txt`
- `data/solomon/RC101.txt`
- `data/solomon/C201.txt`
- `data/solomon/R201.txt`
- `data/solomon/RC201.txt`

You can also put more files (e.g. C102, R102, RC102, ...).

## Run single instance

```bash
python main.py --mode single --instance data/solomon/C101.txt --runs 30 --iters 1000 --seed 42
```

## Run section6 benchmark table

```bash
python main.py --mode section6 --dataset-root data/solomon --instances C101,R101,RC101,C201,R201,RC201 --runs 30 --iters 1000 --seed 42 --output-csv results/section6_results.csv
```

The CSV file can be used to build comparison tables with the paper.
