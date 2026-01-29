def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y
