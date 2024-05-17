import argparse
import warnings
from GmSa.optimizer import MixOptimizer
from models import *
from layers import *
from loss import *

import torch
import scipy.io as sio

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GCL-MVC-Net')
parser.add_argument('--load_model', default=True, help='Testing if True or training.')
parser.add_argument('--save_model', default=True, help='Saving the model after training.')

parser.add_argument('--db', type=str, default='Multi-COIL-10',
                    choices=['MNIST-USPS', 'Multi-COIL-10', 'scene', 'Fashion', 'ORL'],
                    help='dataset name')
parser.add_argument('--seed', type=int, default=10, help='Initializing random seed.')
parser.add_argument("--con_epochs", default=100, help='Number of epochs to fine-tuning.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='Initializing learning rate.')
parser.add_argument('--weight_decay', type=float, default=0., help='Initializing weight decay.')
parser.add_argument("--temperature_l", type=float, default=1.0)
parser.add_argument('--batch_size', default=100, type=int,
                    help='The total number of samples must be evenly divisible by batch_size.')
parser.add_argument('--normalized', type=bool, default=False)
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx.')

args = parser.parse_args(['--db','MNIST-USPS'])
print("==========\nArgs:{}\n==========".format(args))

# torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    if args.db == "MNIST-USPS":
        args.learning_rate = 0.0002
        args.batch_size = 50
        args.seed = 10
        args.con_epochs = 500
        args.normalized = False
        lmd = 0.05
        beta = 0.005
        gamma = 0.1
        dim_high_feature = 1500
        dim_low_feature = 1024
        dims = [256, 512, 1024]

    elif args.db == "Multi-COIL-10":
        args.learning_rate = 0.0001
        args.batch_size = 50
        args.seed = 20
        args.con_epochs = 200
        args.normalized = False
        lmd = 0.01
        beta = 0.01
        gamma = 1
        dim_high_feature = 768
        dim_low_feature = 200
        dims = [256, 512, 1024, 2048]

    elif args.db == "scene":
        args.learning_rate = 0.001
        args.con_epochs = 200
        args.batch_size = 69
        args.seed = 50
        args.normalized = True
        args.temperature_l = 1
        lmd = 0.1
        beta = 0.1
        gamma = 0.1
        dim_high_feature = 1500
        dim_low_feature = 256
        dims = [256, 512, 1024, 2048]

    elif args.db == "ORL":

        args.learning_rate = 0.0001
        args.batch_size = 50
        args.seed = 20
        args.con_epochs = 200
        args.normalized = True
        args.temperature_l = 0.5
        lmd = 0.005
        beta = 0.001
        gamma = 0.01
        dim_high_feature = 1024
        dim_low_feature = 1024
        dims = [256, 512, 1024]

    elif args.db == "Fashion":
        args.learning_rate = 0.0005
        args.batch_size = 100
        args.con_epochs = 200
        args.seed = 20
        args.normalized = True
        args.temperature_l = 0.5
        lmd = 0.005
        beta = 0.005
        gamma = 0.005
        dim_high_feature = 2000
        dim_low_feature = 500
        dims = [256, 512]


    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mv_data = MultiviewData(args.db, device)
    num_views = len(mv_data.data_views)
    num_samples = mv_data.labels.size
    num_clusters = np.unique(mv_data.labels).size
    print(num_clusters)
    input_sizes = np.zeros(num_views, dtype=int)
    for idx in range(num_views):
        input_sizes[idx] = mv_data.data_views[idx].shape[1]

    t = time.time()
    # neural network architecture
    mnw = GCLMVCNetwork(num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters)
    # filling it into GPU
    mnw = mnw.to(device)

    mvc_loss = DeepMVCLoss(args.batch_size, num_clusters)
    optimizer = torch.optim.Adam(mnw.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = MixOptimizer(optimizer)

    if args.load_model:
        state_dict = torch.load('./models/GCLMVC_pytorch_model_%s.pth' % args.db)
        mnw.load_state_dict(state_dict)

    else:
        t = time.time()
        fine_tuning_loss_values = np.zeros(args.con_epochs, dtype=np.float64)
        print("contrastive_train start")
        for epoch in range(args.con_epochs):
            print('epoch:{}'.format(epoch))
            total_loss = contrastive_train(mnw, mv_data, mvc_loss, args.batch_size, lmd, beta, gamma,
                                           args.temperature_l, args.normalized, epoch, optimizer)
            fine_tuning_loss_values[epoch] = total_loss


        print("contrastive_train finished.")
        print("Total time elapsed: {:.2f}s".format(time.time() - t))

        if args.save_model:
            torch.save(mnw.state_dict(), './models/GCLMVC_pytorch_model_%s.pth' % args.db)

    acc, nmi, pur, ari = valid(mnw, mv_data, args.batch_size)
    with open('result_%s.txt' % args.db, 'a+') as f:
        f.write('{} \t {} \t {} \t {} \t {} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.4f} \t {} \t {} \n'.format(
            gamma, args.temperature_l, args.seed, args.batch_size,
            args.learning_rate, lmd, beta, acc, nmi, pur, (time.time() - t), args.con_epochs, num_views))
        f.flush()

