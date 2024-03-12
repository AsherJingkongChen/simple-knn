import torch
from simple_knn._C import distCUDA2

torch.set_printoptions(precision=7)

TestData_distXXX2 = [
    (
        # Minimal Rows: 3 x 3
        torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        torch.Tensor([1.1342745e38, 1.1342745e38, 1.1342745e38]).float(),
    ),
    (
        # A Few Rows: 8 x 3
        torch.Tensor(
            [
                [0.587, 8.188, 0.164],
                [7.323, 0.374, 4.009],
                [0.757, 7.927, 1.040],
                [7.847, 0.275, 1.542],
                [1.086, 1.783, 0.804],
                [4.925, 0.191, 5.166],
                [0.505, 1.821, 0.828],
                [7.399, 1.662, 6.140],
            ]
        ),
        torch.Tensor(
            [
                27.8444424,
                6.5662956,
                25.3895817,
                17.1050186,
                24.8505611,
                12.6780825,
                26.2391891,
                12.9017153,
            ]
        ).float(),
    ),
]

def test_dimensionality():
    "Tests of Dimensionality"

    def check(i, o):
        assert i.size(0) == o.size(0)
        assert len(o.shape) == 1

    for i, e in TestData_distXXX2:
        o = distCUDA2(i.cuda())
        check(i, e)
        check(i, o)

def test_evaluation():
    "Tests of Evaluation"

    def check(e, o):
        assert torch.equal(e.cpu(), o.cpu())

    for i, e in TestData_distXXX2:
        o = distCUDA2(i.cuda())
        check(e, o)

test_dimensionality()
test_evaluation()
