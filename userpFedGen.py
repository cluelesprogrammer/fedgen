import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np
from torchmetrics import Accuracy
from data_utils import sample_multi_label_distribution
import wandb
from tqdm import tqdm

class UserpFedGen():
    def __init__(self,
                 config, id, model, generative_model,
                 train_loader,
                 available_labels, latent_layer_idx, label_info, country,
                 use_adam=False):
        self.config = config
        #user model and general training parameters
        self.model = copy.deepcopy(model)
        self.id = id
        self.model_name = config['backbone']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.local_epochs = config['local_epochs']
        self.beta = config['beta']
        self.lamda = config['lambda']
        self.algorithm = config['algorithm']
        self.K = config['K']
        self.dataset = config['dataset']
        #knowledge distillation and generator parameters
        self.generative_model = generative_model
        self.latent_layer_idx = latent_layer_idx
        self.gen_batch_size = config['gen_batch_size']
        self.generative_alpha = config['generative_alpha']
        self.generative_beta = config['generative_beta']
        #data parameters and info
        self.country = country
        self.train_loader = train_loader
        self.iter_train_loader = iter(self.train_loader)
        self.train_samples = len(self.train_loader) * self.batch_size
        self.unique_labels = config['unique_labels']
        self.available_labels = available_labels
        self.label_info = label_info
        self.prior_distribution = np.tile(np.array([1/len(self.available_labels)]), len(self.available_labels))
        self.label_weights = {k: self.label_info[k]/ sum(self.label_info.values()) for k in self.label_info}

        #loss and optimizer
        self.init_loss_fn()
        self.optimizer=torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.learning_rate, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=1e-2, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.label_counts = {}

        #steps for logging
        self.steps, self.latent_student_loss_step = 0, 0
        self.train_loader = train_loader
        self.config = config
        #to avoid redundant sigmoid calls
        self.sigmoid = nn.Sigmoid()

    def init_loss_fn(self):
        self.loss=nn.BCELoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")#,log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:0 for label in range(self.unique_labels)}

    def train(self, glob_iter, qualified_labels, label_weights, early_stop=True, regularization=True, verbose=False, debug=False):
        self.clean_up_counts()
        self.model.train()
        self.generative_model.eval()
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0
        epoch_wise_loss = {e:0 for e in range(self.local_epochs)}
        EPOCH_PRED_LOSS, EPOCH_LATENT_LOSS, EPOCH_TEACHER_LOSS = epoch_wise_loss, epoch_wise_loss, epoch_wise_loss
        for epoch in tqdm(range(self.local_epochs)):
            print('training on {}. Epoch {} '.format(self.country, epoch+1))
            self.model.train()
            for k in tqdm(range(self.K)):
                self.optimizer.zero_grad()
                #### sample from real dataset (un-weighted)
                samples =self.get_next_batch(count_labels=True)
                X, y = samples['X'], samples['y']
                #bce loss expects targets to be in float. Weird
                y = y.to(torch.float32)

                self.update_label_counts(samples['labels'], samples['counts'])
                X = X.to(self.config['device'])
                y = y.to(self.config['device'])
                model_result=self.model(X, logit=True)
                user_output_sigmoid = model_result['output']
                predictive_loss=self.loss(user_output_sigmoid, y)
                EPOCH_PRED_LOSS[epoch] += predictive_loss.detach().clone().item()
                teacher_loss, latent_loss = None, None
                #### sample y and generate z
                if regularization:
                    generative_alpha=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                    generative_beta=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                    ### get generator output(latent representation) of the same label
                    gen_output=self.generative_model(y, latent_layer_idx=self.latent_layer_idx)['output']
                    target_p=self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['output']
                    target_p = target_p.detach().clone()
                    #kl divergence as ensemble loss
                    user_latent_loss= generative_beta * self.loss(user_output_sigmoid, target_p)
                    #self.latent_loss_every_batch.append(user_latent_loss.detach().clone().item())
                    EPOCH_LATENT_LOSS[epoch] += user_latent_loss.detach().clone().item()
                    """
                    Sampling y for multi label case. At least one label per sample is selected
                    """
                    sampled_y = sample_multi_label_distribution(self.gen_batch_size, self.config['label_p'])
                    sampled_y=torch.tensor(sampled_y).to(torch.float32).to(self.config['device'])
                    gen_result=self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)
                    gen_output=gen_result['output']
                    # latent representation when latent = True, x otherwise
                    user_output_sigmoid =self.model(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    #implementation's teacher loss uses cross entropy. Replacing it with BCE because of multi label prediction
                    teacher_loss =  generative_alpha * torch.mean(
                        self.loss(user_output_sigmoid, sampled_y)
                    )
                    #self.teacher_loss_every_batch.append(teacher_loss.detach().clone().item())
                    EPOCH_TEACHER_LOSS[epoch] += teacher_loss.detach().clone().item()
                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.gen_batch_size / self.batch_size
                    loss=predictive_loss + gen_ratio * teacher_loss + user_latent_loss
                    TEACHER_LOSS+=teacher_loss
                    LATENT_LOSS+=user_latent_loss
                    self.latent_student_loss_step+=1
                    wandb.log({'latent_loss_{}'.format(self.country): user_latent_loss.detach().clone().item(), 
                                'generative_alpha': generative_alpha,
                                'generative_beta': generative_beta,
                                'teacher_loss_{}'.format(self.country): teacher_loss.detach().clone().item(),
                                'latent_teacher_steps': self.latent_student_loss_step
                    })
                else:
                    #### get loss and perform optimization
                    loss=predictive_loss
                wandb.log({'predictive_loss_{}'.format(self.country):  predictive_loss.detach().clone().item(),
                           'training_step': self.steps
                })

                #self.loss_every_batch.append(loss.item())
                loss.backward()
                self.steps += 1
                self.optimizer.step()#self.local_model)
                if (debug and k > 2):
                    break

            # local-model <=== self.model
            wandb.log({'{}_epoch_predictive_loss'.format(self.country): EPOCH_PRED_LOSS[epoch]})
            if (regularization):
                wandb.log({'{}_epoch_latent_loss'.format(self.country): EPOCH_LATENT_LOSS[epoch],
                           '{}_epoch_teacher_loss'.format(self.country): EPOCH_TEACHER_LOSS[epoch],
                           'regularization epoch': epoch})
            self.lr_scheduler.step()
            if regularization and verbose:
                TEACHER_LOSS=TEACHER_LOSS.detach().cpu().numpy() / (self.local_epochs * self.K)
                LATENT_LOSS=LATENT_LOSS.detach().cpu().numpy() / (self.local_epochs * self.K)
                info='\nUser Teacher Loss={:.4f}'.format(TEACHER_LOSS)
                info+=', Latent Loss={:.4f}'.format(LATENT_LOSS)
                print(info)
            if (debug and epoch > 2):
                break

    def get_next_batch(self, count_labels=True):
        try:
            (X, y) = next(self.iter_train_loader)
        except StopIteration:
            self.iter_train_loader = iter(self.train_loader)
            (X, y) = next(self.iter_train_loader)
        result = {'X': X, 'y': y}
        if count_labels:
            labels_id = torch.where(y == 1)[1]
            unique_y, counts = torch.unique(labels_id, return_counts=True)
            unique_y = unique_y.detach().numpy()
            counts = counts.detach().numpy()
            result['labels'] = unique_y
            result['counts'] = counts
        return result

    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        #weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts]) # smaller count --> larger weight
        weights = len(self.available_labels) * weights / np.sum(weights) # normalized
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights





