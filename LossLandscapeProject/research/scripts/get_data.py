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
# Fallback when cs.toronto.edu is unreachable: fast.ai's lossless PNG copy. It stores images per
# class, so the original ordering is lost; we rebuild the batches in a fixed class-interleaved order.
# Pixels and labels are identical, but the 10k training subset drawn by lls.data differs from the one
# drawn from the original files. Batches built this way are marked with a REORDERED file.
CIFAR_PNG_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
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


def cifar_from_png():
    import pickle
    import numpy as np
    from PIL import Image
    src = os.path.join(DATA, "cifar10")
    if not os.path.isdir(src):
        tgz = os.path.join(DATA, "cifar10.tgz")
        urllib.request.urlretrieve(CIFAR_PNG_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(DATA)
        os.remove(tgz)
    out = os.path.join(DATA, "cifar-10-batches-py")
    os.makedirs(out, exist_ok=True)

    def split(name):
        files = [sorted(os.listdir(os.path.join(src, name, c)), key=lambda f: int(f.split(".")[0])) for c in CLASSES]
        order = [(y, files[y][i]) for i in range(max(map(len, files))) for y in range(10) if i < len(files[y])]
        x = np.stack([np.asarray(Image.open(os.path.join(src, name, CLASSES[y], f)).convert("RGB")).transpose(2, 0, 1).reshape(-1)
                      for y, f in order])
        return x, [y for y, _ in order]

    xtr, ytr = split("train")
    for b in range(5):
        with open(os.path.join(out, f"data_batch_{b + 1}"), "wb") as f:
            pickle.dump({b"data": xtr[b * 10000:(b + 1) * 10000], b"labels": ytr[b * 10000:(b + 1) * 10000]}, f)
    xte, yte = split("test")
    with open(os.path.join(out, "test_batch"), "wb") as f:
        pickle.dump({b"data": xte, b"labels": yte}, f)
    open(os.path.join(out, "REORDERED"), "w").write(CIFAR_PNG_URL + "\n")
    shutil.rmtree(src)


def main():
    os.makedirs(DATA, exist_ok=True)
    if not ok("mnist.npz"):
        print("downloading MNIST ...")
        urllib.request.urlretrieve(MNIST_URL, os.path.join(DATA, "mnist.npz"))
    if os.path.exists(os.path.join(DATA, "cifar-10-batches-py", "REORDERED")):
        print("CIFAR-10 was rebuilt from the PNG mirror (reordered; checksums not applicable)")
    elif not all(ok(k) for k in MD5 if k.startswith("cifar")):
        print("downloading CIFAR-10 (163 MB) ...")
        tgz = os.path.join(DATA, "cifar-10-python.tar.gz")
        shutil.rmtree(os.path.join(DATA, "cifar-10-batches-py"), ignore_errors=True)
        try:
            urllib.request.urlretrieve(CIFAR_URL, tgz)
            with tarfile.open(tgz) as t:
                t.extractall(DATA)
            os.remove(tgz)
        except OSError as e:
            print(f"original CIFAR-10 host failed ({e}); rebuilding from the PNG mirror")
            cifar_from_png()
    reordered = os.path.exists(os.path.join(DATA, "cifar-10-batches-py", "REORDERED"))
    bad = [k for k in MD5 if not ok(k) and not (reordered and k.startswith("cifar"))]
    if bad:
        raise SystemExit(f"checksum mismatch: {bad}")
    print("data OK:", DATA)


if __name__ == "__main__":
    main()
