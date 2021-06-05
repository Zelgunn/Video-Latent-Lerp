from datasets.tfrecord_builders.SubwayTFRecordBuilder import SubwayVideo
from protocols.video_protocols import UCSDProtocol, AvenueProtocol, ShanghaiTechProtocol, SubwayProtocol


def main():
    best_weights = {
        "ped2": 16,
        "ped1": 50,
        "avenue": 22,
        "shanghaitech": 14,
        "exit": 11,
        "entrance": 97,
    }

    train = False
    initial_epoch = None
    dataset = "shanghaitech"

    if initial_epoch is None:
        initial_epoch = best_weights[dataset]

    if dataset == "ped2":
        protocol = UCSDProtocol(initial_epoch=initial_epoch, dataset_version=2)
    elif dataset == "ped1":
        protocol = UCSDProtocol(initial_epoch=initial_epoch, dataset_version=1)
    elif dataset == "avenue":
        protocol = AvenueProtocol(initial_epoch=initial_epoch)
    elif dataset == "shanghaitech":
        protocol = ShanghaiTechProtocol(initial_epoch=initial_epoch)
    elif dataset == "exit":
        protocol = SubwayProtocol(initial_epoch=initial_epoch, video_id=SubwayVideo.EXIT)
    elif dataset == "entrance":
        protocol = SubwayProtocol(initial_epoch=initial_epoch, video_id=SubwayVideo.ENTRANCE)
    elif dataset == "mall1":
        protocol = SubwayProtocol(initial_epoch=initial_epoch, video_id=SubwayVideo.MALL1)
    elif dataset == "mall2":
        protocol = SubwayProtocol(initial_epoch=initial_epoch, video_id=SubwayVideo.MALL2)
    elif dataset == "mall3":
        protocol = SubwayProtocol(initial_epoch=initial_epoch, video_id=SubwayVideo.MALL3)
    else:
        raise ValueError

    if train:
        protocol.train_model()
    else:
        protocol.test_model()


if __name__ == "__main__":
    main()
