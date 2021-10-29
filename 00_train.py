########################################################################
# import default libraries
########################################################################
import os
import sys
import gc
########################################################################


########################################################################
# import additional libraries
########################################################################
import numpy as np
import scipy.stats
# from import
from tqdm import tqdm
try:
    from sklearn.externals import joblib
except:
    import joblib
# original lib
import common as com
import torch
from model import AutoEncoder
import dataloader_dcase
from torch.utils.data import DataLoader
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(7, 5))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################


########################################################################
# get data from the list for file paths
########################################################################
def file_list_to_data(file_list,
                      msg="calc...",
                      n_mels=64,
                      n_frames=5,
                      n_hop_frames=1,
                      n_fft=1024,
                      hop_length=512,
                      power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vectors = com.file_to_vectors(file_list[idx],
                                                n_mels=n_mels,
                                                n_frames=n_frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        vectors = vectors[: : n_hop_frames, :]
        if idx == 0:
            data = np.zeros((len(file_list) * vectors.shape[0], dims), float)
        data[vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :] = vectors

    return data


########################################################################


########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
        
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                     machine_type=machine_type)

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue
        
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"],
                                                                  machine_type=machine_type)
        # pickle file for storing anomaly score distribution
        score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                machine_type=machine_type)

        # generate dataset
        print("============== DATASET_GENERATOR ==============")

        # get file list for all sections
        # all values of y_true are zero in training
        files, y_true = com.file_list_generator(target_dir=target_dir,
                                                section_name="*",
                                                dir_name="train",
                                                mode=mode)

        print(files.shape)
        mask = []
        for i, f in enumerate(files):
            if "section_03" in f or "section_04" in f or "section_05" in f:
                mask.append(i)
        files = np.delete(files, mask)
        print(files.shape)

        data = file_list_to_data(files,
                                 msg="generate train_dataset",
                                 n_mels=param["feature"]["n_mels"],
                                 n_frames=param["feature"]["n_frames"],
                                 n_hop_frames=param["feature"]["n_hop_frames"],
                                 n_fft=param["feature"]["n_fft"],
                                 hop_length=param["feature"]["hop_length"],
                                 power=param["feature"]["power"])

        # number of vectors for each wave file
        n_vectors_ea_file = int(data.shape[0] / len(files))

        training_data = dataloader_dcase.CustomDataset(data)
        train_dataloader = DataLoader(training_data, batch_size=param["fit"]["batch_size"], shuffle=True)

        model = AutoEncoder(param["feature"]["n_mels"] * param["feature"]["n_frames"])
        model.to(device)
        print(model)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=param["fit"]["lr"])

        # train model
        print("============== MODEL TRAINING ==============")
        for epoch in range(param["fit"]["epochs"]):
            running_loss = 0.0
            total_step = 0
            for step, x in enumerate(tqdm(train_dataloader)):
                x = x.to(device=device, dtype=torch.float)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_fn(y_pred, x)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_step=step

            print('[{}] loss: {}'.format(epoch + 1, running_loss / total_step))
            running_loss = 0.0
            total_step = 0

        torch.save(model.state_dict(), model_file_path)
        com.logger.info("save_model -> {}".format(model_file_path))
        print("============== END TRAINING ==============")

        del data
        del model
        del optimizer
        del training_data
        del train_dataloader
        gc.collect()
        