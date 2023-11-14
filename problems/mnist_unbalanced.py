import torch
import torchvision.datasets
import torchvision.transforms
import random


def move_to_type_device(x, y, device):
    print("X:", x.shape)
    print("y:", y.shape)
    x = x.to(torch.get_default_dtype())
    y = y.to(torch.get_default_dtype())
    x, y = x.to(device), y.to(device)
    return x, y


def load_bound_dataset(
    dataset, batch_size, shuffle=False, start=None, end=None, **kwargs
):
    def _bound_dataset(dataset, start, end):
        if start is None:
            start = 0
        if end is None:
            end = len(dataset)
        return torch.utils.data.Subset(dataset, range(start, end))

    dataset = _bound_dataset(dataset, start, end)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, **kwargs)


def fetch_mnist(root, train=False, transform=None, target_transform=None):
    transform = (
        transform if transform is not None else torchvision.transforms.ToTensor()
    )
    dataset = torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
    return dataset


def load_mnist(
    root, batch_size, train=False, transform=None, target_transform=None, **kwargs
):
    dataset = fetch_mnist(root, train, transform, target_transform)
    return load_bound_dataset(dataset, batch_size, **kwargs)


def create_labels(y0):
    labels_dict = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
    y0 = torch.stack([torch.tensor(labels_dict[int(cur_y)]) for cur_y in y0])
    return y0


def get_unbalanced_data(args, data_loader):
    print("CREATING UNBALANCED DATASET...")
    # get unbalanced data
    s = 4
    data = data_loader.dataset.dataset.data
    targets = data_loader.dataset.dataset.targets
    x0, y0 = [], []
    # Add all 0's
    x0.append(data[targets == 0])
    y0.append(targets[targets == 0])
    # Add all other digits (two possible ways (1): uniform, (2): equally sampled). All other digits classes are a minority

    # Start from option (2)
    num_samples_per_label = int((x0[0].shape[0] / 2) / 9)
    for lable in range(1, 10):
        lable_indices = torch.nonzero(targets == lable).squeeze().tolist()
        sample_indices = random.sample(lable_indices, num_samples_per_label)
        x0.append(data[sample_indices])
        y0.append(torch.tensor([1] * num_samples_per_label))

    x0, y0 = torch.vstack(x0), torch.concat(y0)
    x0 = x0.unsqueeze(1)
    return x0, y0


def load_mnist_data(args):
    print("LOADING TRAINSET")
    # Get Train Set
    data_loader = load_mnist(
        root=args.datasets_dir,
        batch_size=100,
        train=True,
        shuffle=False,
        start=0,
        end=50000,
    )
    x0, y0 = get_unbalanced_data(args, data_loader)

    # Get Test Set
    print("LOADING TESTSET")
    assert not args.data_use_test or (
        args.data_use_test and args.data_test_amount >= 2
    ), f"args.data_use_test={args.data_use_test} but args.data_test_amount={args.data_test_amount}"
    data_loader = load_mnist(
        root=args.datasets_dir,
        batch_size=100,
        train=False,
        shuffle=False,
        start=0,
        end=10000,
    )
    x0_test, y0_test = get_unbalanced_data(args, data_loader)

    # move to cuda and double
    x0, y0 = move_to_type_device(x0, y0, args.device)
    x0_test, y0_test = move_to_type_device(x0_test, y0_test, args.device)

    print(f"BALANCE: 0: {y0[y0 == 0].shape[0]}, 1: {y0[y0 == 1].shape[0]}")

    return [(x0, y0)], [(x0_test, y0_test)], None


def get_dataloader(args):
    args.input_dim = 28 * 28
    args.num_classes = 2
    args.output_dim = 1
    args.dataset = "mnist"

    if args.run_mode == "reconstruct":
        args.extraction_data_amount = (
            args.extraction_data_amount_per_class * args.num_classes
        )

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000

    data_loader = load_mnist_data(args)
    return data_loader
