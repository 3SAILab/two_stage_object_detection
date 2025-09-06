from torchvision.transforms import v2 as T

transform = {
    "train":{
        T.ToTensor(),
        T.Resize()
    },
    "eval":{
        T.Resize()
    }
}