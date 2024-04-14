import numpy as np
from    matplotlib import pyplot as plt

import os, cv2
from skimage import io, transform
from .AE_model import AE_Net

os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

batch_size = 10
weight = 64
height = 64
input_image_channel = 3
epochs = 100
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def maxmin_norm(data, ):
    data = (data-data.min())/(data.max() - data.min())
    return data

from torch.utils.data import Dataset, DataLoader
class RadioUNet_c(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""

    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="/data/RadioUnet/",
                 numTx=80,
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom".
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())

        Output:
            inputs: The RadioUNet inputs.
            image_gain

        """

        # self.phase=phase

        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 601
            self.ind2 = 699
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.thresh = thresh

        self.simulation = simulation

        if simulation == "DPM":
            self.dir_gain = self.dir_dataset + "gain/DPM/"

        elif simulation == "IRT2":
            self.dir_gain = self.dir_dataset + "gain/IRT2/"

        self.IRT2maxW = IRT2maxW

        self.cityMap = cityMap
        self.missing = missing
        if cityMap == "complete":
            self.dir_buildings = self.dir_dataset + "png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset + "png/buildings_missing"  # a random index will be concatenated in the code
        # else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"

        self.transform = transform

        self.dir_Tx = self.dir_dataset + "png/antennas/"


        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map_ind = self.maps_inds[idxr + self.ind1] + 1
        # names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        # names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

        # Load buildings:
        img_name_buildings = os.path.join(self.dir_buildings, name1)
        image_buildings = np.asarray(io.imread(img_name_buildings))

        # Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))

        # Load radio map:

        img_name_gain = os.path.join(self.dir_gain, name2)
        image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)), axis=2) / 255

        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain - self.thresh * np.ones(np.shape(image_gain))
            image_gain = image_gain / (1 - self.thresh)

        image_buildings = cv2.resize(image_buildings, (64,64))
        image_Tx = cv2.resize(image_Tx, (64,64))
        image_gain = cv2.resize(image_gain, (64,64))

        image_buildings = np.reshape(image_buildings, [64, 64])
        image_Tx = np.reshape(image_Tx, [64, 64])
        image_gain = np.reshape(image_gain, [1,64,64])

        mask_rx = image_gain == image_gain.max()
        image_Tx = image_gain*mask_rx
        image_Tx = np.reshape(image_Tx, [64, 64])

        image_buildings = maxmin_norm(image_buildings)
        image_Tx = maxmin_norm(image_Tx)
        image_gain = maxmin_norm(image_gain)


        a = np.zeros(64*64, dtype=int)
        a[:500] = 1
        np.random.shuffle(a)
        mask = np.resize(a,[64,64])
        image_gain_mask = image_gain[0,:,:] * mask
        #make random measured sample

        # inputs to radioUNet
        inputs = np.stack([image_buildings , image_Tx, image_gain_mask], axis=0)


        return [inputs, image_gain]


def RMSE(A,B):
    return np.sqrt(np.mean(np.power((A - B), 2)))


def training(dataset,phase):
    ShowPic = True
    for inputs, targets in dataloaders[phase]:
        inputs = inputs.numpy().astype('float')
        targets = targets.numpy().astype('float')

        input_image = np.reshape(inputs, [batch_size, weight, height, input_image_channel])
        building_img = np.reshape(input_image[:, :, :, 0], [batch_size, weight, height, 1])
        Pint_map = np.reshape(input_image[:, :, :, 1], [batch_size, weight, height, 1])
        target = np.reshape(targets, [batch_size, weight, height, 1])

        # if ShowPic:
        #     plt.imshow(target[0,:, :,0])
        #     plt.colorbar()
        #     plt.show()
        #     plt.imshow(input_image[0,:, :, 0])
        #     plt.colorbar()
        #     plt.show()
        #     plt.imshow(input_image[0,:, :, 2])
        #     plt.colorbar()
        #     plt.show()

        generator = AE_Net()
        prediction = generator(input_image, training=False)

        if ShowPic:
            plt.imshow(prediction[0,:, :,0])
            plt.colorbar()
            plt.show()

        rmse = RMSE(prediction[building_img>0], target[building_img>0])
        print(rmse)


if __name__=='__main__':
    Radio_val = RadioUNet_c(phase="test")

    image_datasets = {
        'val': Radio_val
    }

    batch_size = 1

    dataloaders = {
        'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=1)
    }

    training(dataloaders,'val')
    # print(image_build_ant[:,:,0].max(), image_build_ant[:,:,0].min())
    # print(image_build_ant[:,:,1].max(), image_build_ant[:,:,1].data)
    # print(image_gain[:,:,0].data, image_gain[:,:,0].data)

