"""Download MNIST and CIFAR-10 into research/data/ and verify them (same files the original Keras
scripts used). Run once on a new machine:  python scripts/get_data.py"""
import hashlib
import os
import shutil
import tarfile
import urllib.request

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MD5 = {
    "mnist.npz": "8a61469f7ea1b51cbae51d4f78837e45",
    "cifar-10-batches-py/batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
    "cifar-10-batches-py/data_batch_1": "c99cafc152244af753f735de768cd75f",
    "cifar-10-batches-py/data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "cifar-10-batches-py/data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "cifar-10-batches-py/data_batch_4": "634d18415352ddfa80567beed471001a",
    "cifar-10-batches-py/data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "cifar-10-batches-py/test_batch": "40351d587109b95175f43aff81a1287e",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ok(rel):
    p = os.path.join(DATA, rel)
    return os.path.exists(p) and md5(p) == MD5[rel]


def main():
    os.makedirs(DATA, exist_ok=True)
    if not ok("mnist.npz"):
        print("downloading MNIST ...")
        urllib.request.urlretrieve(MNIST_URL, os.path.join(DATA, "mnist.npz"))
    if not all(ok(k) for k in MD5 if k.startswith("cifar")):
        print("downloading CIFAR-10 (163 MB) ...")
        tgz = os.path.join(DATA, "cifar-10-python.tar.gz")
        urllib.request.urlretrieve(CIFAR_URL, tgz)
        shutil.rmtree(os.path.join(DATA, "cifar-10-batches-py"), ignore_errors=True)
        with tarfile.open(tgz) as t:
            t.extractall(DATA)
        os.remove(tgz)
    bad = [k for k in MD5 if not ok(k)]
    if bad:
        raise SystemExit(f"checksum mismatch: {bad}")
    print("data OK:", DATA)


if __name__ == "__main__":
    main()
