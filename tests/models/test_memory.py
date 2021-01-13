import time

from neuroblast.models.memory import DTM


def test_dtm():
    start = time.time()
    memory = DTM(5000, 512)
    print('model setup: {:.6f} seconds'.format(time.time() - start))

    start = time.time()
    y = memory.p_y.sample()
    r = memory.read(y)
    y1 = memory.q_y(r).sample()
    r1 = memory.read(y1)
    print('step 1: {:.6f} seconds'.format(time.time() - start))

    start = time.time()
    path = memory.shortest_path(y, y1)
    print('step 2: {:.6f} seconds'.format(time.time() - start))

    start = time.time()
    memory.write(y1, r)
    print('step 3: {:.6f} seconds'.format(time.time() - start))

    start = time.time()
    path1 = memory.shortest_path(y, y1)
    print('step 4: {:.6f} seconds'.format(time.time() - start))

    print(path, path1)
    if path is None and path1 is not None:
        assert path1.size(0) > 0
    else:
        assert path1.size(0) == 2
