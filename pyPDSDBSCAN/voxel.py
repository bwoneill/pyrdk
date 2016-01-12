class Voxel(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __eq__(self, other):
        return self.p1 == other.p1

    def __hash__(self):
        return ('(%s):(%s)' % (','.join(list(self.p1.astype(str))), ','.join(list(self.p2.astype(str))))).__hash__()
