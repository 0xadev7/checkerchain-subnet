import time
import subprocess

from checkerchain.database.mongo import dataset_col, get_meta, upsert_meta

MIN_NEW_ROWS = 100  # retrain threshold


def count_rows():
    return dataset_col.estimated_document_count()


def main():
    while True:
        last_n = get_meta("last_train_rowcount", 0) or 0
        n = count_rows()
        if n - int(last_n) >= MIN_NEW_ROWS:
            print(f"[watch] New rows {n-last_n}; retraining...")
            subprocess.run(["python", "train_gb.py"], check=False)
            upsert_meta("last_train_rowcount", n)
        time.sleep(900)  # 15 min


if __name__ == "__main__":
    main()
