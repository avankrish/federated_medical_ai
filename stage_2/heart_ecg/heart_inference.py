import torch
from stage_2.heart_ecg.model import ECGCNN


# Load trained model
model = ECGCNN()
model.load_state_dict(torch.load("stage_2/heart_ecg/ecg_model.pth"))
model.eval()


def heart_ecg_inference(signal):

    """
    signal : list or array of 187 ECG values
    """

    signal = torch.tensor(signal, dtype=torch.float32)

    # reshape to CNN format
    signal = signal.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        output = model(signal)

        pred = (output > 0.5).int().item()

    return pred