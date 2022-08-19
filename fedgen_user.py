import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import wandb
from tqdm import tqdm

class UserpFedGen():
    def __init__(self, config, id, model, generative_model, train_loader, available_labels,
                    latent_layer_idx, label_info, device, val_loader=None):
        self.gen_batch_size = config['gen_batch_size']
        self.model = copy.deepcopy(model)
        self.generative_model = generative_model
        self.latent_layer_idx = latent_layer_idx
        self.available_labels = available_labels
        self.label_info = label_info
        self.prior_distribution = np.tile(np.array([1 / len(self.available_labels)]), len(self.available_labels))
        self.label_weights = {k: self.label_info[k] / sum(self.label_info.values()) for k in self.label_info}
        self.sigmoid = nn.Sigmoid()
        self.country = 'Finland'
        self.unique_labels = 19
        #self.country = config['country'][id]
        self.id = id
        self.steps = 0
        self.train_loader = train_loader
        self.train_samples = config['batch_size'] * len(train_loader)
        #self.iter_train_loader = iter(self.train_loader)
        #self.iter_val_loader = iter(self.val_loader)
        self.device = device


        # those parameters are for personalized federated learning.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.prior_decoder = None
        self.prior_params = None
        self.label_counts = {}
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=0.001, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=1e-2, amsgrad=False)
        self.init_loss_fn()

    def init_loss_fn(self):
        self.loss = nn.BCELoss()
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")

    def set_parameters(self, model,beta=1):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
                local_param.data = beta * new_param.data.clone() + (1-beta) * local_param.data.clone()

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label: 1 for label in range(self.unique_labels)}

    def get_labels_and_counts(self, y):
        result = {}
        labels_id = torch.where(y == 1)[1]
        unique_y, counts = torch.unique(labels_id, return_counts=True)
        unique_y = unique_y.detach().numpy()
        counts = counts.detach().numpy()
        result['labels'] = unique_y
        result['counts'] = counts
        return result

    def train(self, glob_iter, personalized=False, early_stop=100, regularization=True, verbose=False):
        self.clean_up_counts()
        self.model.train()
        self.generative_model.eval()
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0
        #epoch_wise_loss = {e: 0 for e in range(self.local_epochs)}
        EPOCH_PRED_LOSS, EPOCH_LATENT_LOSS, EPOCH_TEACHER_LOSS = 0,0,0

        for i, (X, y) in enumerate(tqdm(self.train_loader)):
            print('user {}'.format(self.id), ' is training')
            self.model.train()
            self.optimizer.zero_grad()
            labels_and_counts = self.get_labels_and_counts(y)
            # bce loss expects targets to be in float. Weird
            y = y.to(torch.float32)
            self.update_label_counts(labels_and_counts['labels'], labels_and_counts['counts'])
            X = X.to(self.device)
            y = y.to(self.device)
            model_result = self.model(X, logit=True)
            user_output_sigmoid = model_result['output']
            predictive_loss = self.loss(user_output_sigmoid, y)
            EPOCH_PRED_LOSS+= predictive_loss.detach().clone().item()
            teacher_loss, latent_loss = None, None
            #### latent loss and teacher loss after seeing half of the samples in an epoch
            latent_teacher_loss_condition = regularization and i < early_stop
            if latent_teacher_loss_condition:
                generative_alpha = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                generative_beta = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                ### get generator output(latent representation) of the same label
                gen_output = self.generative_model(y, latent_layer_idx=self.latent_layer_idx)['output']
                target_p = self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['output']
                target_p = target_p.detach().clone()
                # unnecessary -  from the official implementation,
                # target_p=F.sigmoid(logit_given_gen, dim=1).detach().clone()
                # replacing with bce loss from official implementation for multi label case
                user_latent_loss = generative_beta * self.loss(user_output_sigmoid, target_p)

                # self.latent_loss_every_batch.append(user_latent_loss.detach().clone().item())
                EPOCH_LATENT_LOSS += user_latent_loss.detach().clone().item()
                """
                Sampling y for multi label case. At least one label per sample is selected
                """
                sampled_y = sample_multi_label_distribution(len(self.available_labels), self.gen_batch_size)
                sampled_y = torch.tensor(sampled_y)
                gen_result = self.generative_model(sampled_y, self.device, latent_layer_idx=self.latent_layer_idx)
                gen_output = gen_result['output']
                # latent representation when latent = True, x otherwise
                user_output_sigmoid = self.model(gen_output, start_layer_idx=self.latent_layer_idx)['output']

                # implementation's teacher loss uses cross entropy. Replacing it with BCE because of multi label prediction
                teacher_loss = generative_alpha * torch.mean(
                    self.loss(user_output_sigmoid, sampled_y)
                )
                # self.teacher_loss_every_batch.append(teacher_loss.detach().clone().item())
                EPOCH_TEACHER_LOSS += teacher_loss.detach().clone().item()
                print('teacher loss: output with sampled labels from generator input to model last layer ',
                      teacher_loss.item())
                # this is to further balance oversampled down-sampled synthetic data
                gen_ratio = self.gen_batch_size / self.batch_size
                loss = predictive_loss + gen_ratio * teacher_loss + user_latent_loss
                TEACHER_LOSS += teacher_loss
                LATENT_LOSS += user_latent_loss
            else:
                #### get loss and perform optimization
                loss = predictive_loss
                user_latent_loss = None
                teacher_loss = None

            if user_latent_loss:
                ll_to_log = user_latent_loss.detach().clone().item()
            else:
                ll_to_log = float('nan')
            if teacher_loss:
                tl_to_log = teacher_loss.detach().clone().item()
            else:
                tl_to_log = float('nan')
            wandb.log({'predictive_loss_{}'.format(self.country): predictive_loss.detach().clone().item(),
                       'latent_loss_{}'.format(self.country): ll_to_log,
                       'teacher_loss_{}'.format(self.country): tl_to_log,
                       'training_step': self.steps
                       })
            # self.loss_every_batch.append(loss.item())
            loss.backward()
            self.steps += 1
            self.optimizer.step()  # self.local_model)

        # local-model <=== self.model
        #self.evalute(latent_teacher_loss=latent_teacher_loss_condition)
    """
    wandb.log({f'{self.country}_epoch_predictive_loss': EPOCH_PRED_LOSS[epoch]})
    if not (latent_teacher_loss_condition):
        wandb.log({'{}_epoch_latent_loss'.format(self.country): float('nan')})
        wandb.log({'{}_epoch_teacher_loss'.format(self.country): float('nan')})
    else:
        wandb.log({'{}_epoch_latent_loss'.format(self.country): EPOCH_LATENT_LOSS[epoch]})
        wandb.log({'{}_epoch_teacher_loss'.format(self.country): EPOCH_TEACHER_LOSS[epoch]})
    self.clone_model_paramenter(self.model.parameters(), self.local_model)

    if personalized:
        self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
    self.lr_scheduler.step(glob_iter)
    if regularization and verbose:
        TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.local_epochs * self.K)
        LATENT_LOSS = LATENT_LOSS.detach().numpy() / (self.local_epochs * self.K)
        info = '\nUser Teacher Loss={:.4f}'.format(TEACHER_LOSS)
        info += ', Latent Loss={:.4f}'.format(LATENT_LOSS)
        print(info)
    """
    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        # weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts])  # smaller count --> larger weight
        weights = len(self.available_labels) * weights / np.sum(weights)  # normalized
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights

    def evaluate(self, latent_teacher_loss=False):
        self.model.eval()
        pred_val_loss, latent_val_loss, teacher_val_loss = 0, 0, 0
        #### sample from real dataset (un-weighted)
        for k in range(self.K // 5):
            samples = self.get_next_batch(train=False, count_labels=False)
            X, y = samples['X'], samples['y']
            # bce loss expects targets to be in float. Weird
            y = y.to(torch.float32)
            X.to(self.device)
            y.to(self.device)
            with torch.no_grad():
                model_result = self.model(X, logit=True)
            user_output_sigmoid = model_result['output']
            predictive_loss = self.loss(user_output_sigmoid, y),
            self.pred_val_loss += predictive_loss.detach().clone().item()
            if latent_teacher_loss:
                ### get generator output(latent representation) of the same label
                with torch.no_grad():
                    gen_output = self.generative_model(y, self.device, latent_layer_idx=self.latent_layer_idx)['output']
                target_p = self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['output']
                target_p = target_p.detach().clone()
                user_latent_loss = generative_beta * self.loss(user_output_sigmoid, target_p)
                latent_val_loss += user_latent_loss.detach().clone().item()
                """
                Sampling y for multi label case. At least one label per sample is selected
                """
                sampled_y = sample_multi_label_distribution(len(self.available_labels), self.gen_batch_size)
                sampled_y = torch.tensor(sampled_y)
                with torch.no_grad():
                    gen_result = self.generative_model(sampled_y, self.device, latent_layer_idx=self.latent_layer_idx)
                gen_output = gen_result['output']
                # latent representation when latent = True, x otherwise
                user_output_sigmoid = self.model(gen_output, start_layer_idx=self.latent_layer_idx)['output']

                # implementation's teacher loss uses cross entropy. Replacing it with BCE because of multi label prediction
                teacher_loss = generative_alpha * torch.mean(
                    self.loss(user_output_sigmoid, sampled_y)
                )
                teacher_val_loss += teacher_loss.detach().clone().item()
                gen_ratio = self.gen_batch_size / self.batch_size
                loss = predictive_loss + gen_ratio * teacher_loss + user_latent_loss
                TEACHER_LOSS += teacher_loss
                LATENT_LOSS += user_latent_loss
            else:
                #### get loss and perform optimization
                loss = predictive_loss
        wandb.log({'pred_val_loss_{}'.format(self.country): pred_val_loss,
                   'latent_val_loss_{}'.format(self.country): latent_val_loss,
                   'teacher_val_loss_{}'.format(self.country): teacher_val_loss, 'epoch': epoch})

    def get_next_batch(self, train=True, count_labels=True):
        if train:
            try:
                (X, y) = next(self.iter_trainloader)
            # restart the generator if the previous generator is exhausted.
            except StopIteration:
                self.iter_trainloader = iter(self.train_loader)
        else:
            try:
                (X, y) = next(self.iter_valloader)
            except StopIteration:
                self.iter_valloader = iter(self.val_loader)
                (X, y) = next(self.iter_valloader)
        result = {'X': X, 'y': y}
        if count_labels:
            labels_id = torch.where(y == 1)[1]
            unique_y, counts = torch.unique(labels_id, return_counts=True)
            unique_y = unique_y.detach().numpy()
            counts = counts.detach().numpy()
            result['labels'] = unique_y
            result['counts'] = counts
        return result




