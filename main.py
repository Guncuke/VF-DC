import os
import time
import copy
import argparse
import numpy as np
import torch
from torchvision.utils import save_image
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, TensorDataset, DiffAugment, ParamDiffAug


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='MIMIC', help='dataset')
    parser.add_argument('--model', type=str, default='MLP', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.05, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=512, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--gpu_num', type=int, default=3, help='use witch card')

    args = parser.parse_args()
    args.method = 'DM'

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu_num)
        args.device = torch.device('cuda:{}'.format(args.gpu_num))
    args.dsa_param = ParamDiffAug()
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 1000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    style, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.dsa = False if args.dsa_strategy in ['none', 'None'] or style!='image' else True

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    # there are two party P0 and P1, P0 have labels, P1 have features
    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]

        # images_all = torch.cat(images_all, dim=0).to(args.device)
        # labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all, dtype=torch.long)

        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor(np.concatenate([np.ones(args.ipc, dtype=np.int64) * i for i in range(num_classes)]),
                                 dtype=torch.long, requires_grad=False, device=args.device).view(
            -1)  # [0,0,0, 1,1,1, ..., 9,9,9]        # p1持有特征和生成数据集，p0持有标签
        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())
        acc_std = []
        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool[:]:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, args.device, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    acc_std.append((np.mean(accs), np.std(accs)))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                if style == 'image':
                    ''' visualize and save '''
                    save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                    for ch in range(channel):
                        image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                    image_syn_vis[image_syn_vis<0] = 0.0
                    image_syn_vis[image_syn_vis>1] = 1.0
                    save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.



            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, args.device, im_size).to(args.device) # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            # embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel
            embed = net.embed


            ''' update synthetic data '''
            batch_size = args.batch_real

            for i in range(len(images_all) // batch_size):
                start_index = i * batch_size
                end_index = (i + 1) * batch_size

                """ 
                ================step 1=====================
                P1 拿出一个 batch 的数据（特征），embed得到输出
                ===========================================
                """
                img_real = images_all[start_index: end_index].to(args.device)

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)

                output_real = embed(img_real).detach()

                """
                ================step 2=====================
                            p0计算每一个类的掩码
                ===========================================
                """
                label_real = labels_all[start_index: end_index]
                mask = torch.zeros((num_classes, batch_size), dtype=torch.bool)
                for index, label in enumerate(label_real):
                    label = label.to('cpu')
                    mask[label][index] = True

                """
                ================step 3=====================
                二者加密上传embd结果和掩码，协同计算每一个类的平均embed结果
                        P0获得了avg的embedding
                ===========================================
                """
                # TODO: encrypted & noise
                # output_real_classes = []
                output_real_classes_mean = []
                for j in range(num_classes):
                    output_real_class = output_real[mask[j]]
                    output_real_class_mean = torch.mean(output_real_class, dim=0)
                    # output_real_classes.append(output_real_class)
                    output_real_classes_mean.append(output_real_class_mean)
                output_real_classes_mean = torch.stack(output_real_classes_mean)

                """
                ================step 4=====================
                  p1再次计算浓缩数据集的embedding，传递给P0
                ===========================================
                """
                img_syn = image_syn[:].squeeze()
                if args.dsa:
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                output_syn = embed(img_syn)

                """
                ================step 5=====================
                  p0 根据收集到的原始数据的avg_embed和生成数据的embed
                  计算梯度
                ===========================================
                """
                # 这里偷懒了，直接就类0~n直接按顺序排下来
                loss = torch.sum((output_real_classes_mean - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1)) ** 2)

                """
                ================step 6=====================
                  p0 将梯度传给 p1, p1反向传播生成数据集
                ===========================================
                """
                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()

            print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss.item()/batch_size))

            # 每一次迭代完打乱原始训练集
            perm_indices = torch.randperm(len(images_all))
            images_all = images_all[perm_indices]
            labels_all = labels_all[perm_indices]

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))
        print(acc_std)

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))


if __name__ == '__main__':
    main()

