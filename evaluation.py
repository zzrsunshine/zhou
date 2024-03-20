from MantraNet.mantranet2 import pre_trained_model, check_forgery
import os
import glob
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

device = "cup"  # to change if you have a GPU with at least 12Go RAM (it will save you a lot of time !)
MantraNetmodel = pre_trained_model(
    weight_path="MantraNet/MantraNetv4.pt", device=device
)
MantraNetmodel.eval()
root1 = r"D:\bishe\COVERAGE\image"
root2 = r"D:\bishe\COVERAGE\mask"
dir1 = os.listdir(root1)
dir2 = os.listdir(root2)
forged_dir1 = glob.glob(os.path.join(root1,"*t.tif"))
forged_dir2 = glob.glob(os.path.join(root2,"*forged.tif"))
all_score = 0
count = 0
for img_path,forged_path in zip(forged_dir1,forged_dir2):
    fig = check_forgery(MantraNetmodel, img_path=img_path, device=device)
    fig = np.asarray(fig).reshape(-1)

    im = cv2.imread(forged_path)
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imGray = imGray / 255
    imGray = imGray.reshape(-1)

    # fig.savefig(r'F:\wql-Graduate\fuxian\IMD-main\images_out\example4_result.jpg')
    if fig.shape[0]== imGray.shape[0]:
        score = roc_auc_score(imGray, fig)
        print("roc score is",score)
        all_score += score
        count += 1
    else:
        print(img_path,forged_path)
avg_auc_score = all_score / 100
print(avg_auc_score)
