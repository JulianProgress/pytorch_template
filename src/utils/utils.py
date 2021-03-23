import math
import pickle
import smtplib
from email.mime.text import MIMEText

import numpy as np
import scipy
import torch


def load_file(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
    return file


def save_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def move_model_to_cpu(state_dict):
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    return state_dict


def remove_mask(od):
    key_list = list(od.keys())
    masks = []
    mask_list = [key for key in key_list if 'mask' in key]
    for mask in mask_list:
        od.pop(mask)
    return od, masks


def return_std_mean(path):
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    mean = scaler.mean_[12]
    std = scaler.scale_[12]
    return mean, std


def write_txt(save_dir, txt):
    f = open(save_dir, 'w')
    f.write(txt)
    f.close()


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), scipy.stats.sem(a, axis=0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return np.array([m, m - h, m + h])


def print_mask_ratio(mask):
    for m in mask:
        print(np.count_nonzero(m.detach().cpu().numpy() == 0) / len(m))


def print_layer_ratio(net, printing=True):
    name_list = []
    param_list = []
    for name, param in net.named_parameters():
        with torch.no_grad():
            if 'bias' not in name and len(param.detach().cpu().numpy().shape) > 1:
                li = param.detach().cpu().numpy().flatten()
                if printing:
                    print('name: %s, sparse ratio: %.4f' % (name,
                                                            np.count_nonzero(li == 0) / len(li)))
                name_list.append(name)
                param_list.append(np.count_nonzero(li == 0) / len(li))

    return {'name': name_list, 'sparsity': param_list}


def send_email(message, title, email_add='julian8748@gmail.com', passwd='esscrmvyqouyfqkj'):
    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.ehlo()  # say Hello
    smtp.starttls()  # TLS 사용시 필요
    smtp.login(email_add, passwd)

    msg = MIMEText(message)
    msg['Subject'] = title
    msg['To'] = email_add
    smtp.sendmail(email_add, email_add, msg.as_string())

    smtp.quit()
