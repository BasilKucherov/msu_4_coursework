import os
import tqdm
import torch
import json
from models import *
from transforms import *

def training_loop(start_epoch, n_epochs, model, loss_fn, optimizer, dl_train, dl_test, device, save_config):
    train_losses, test_losses = [], []

    experiment_name = save_config['experiment_name']
    experiments_folder = save_config['experiments_folder']

    experiment_folder = os.path.join(experiments_folder, experiment_name)
    checkpoints_folder = os.path.join(experiment_folder, "checkpoints")
    os.makedirs(experiment_folder, exist_ok=True)
    os.makedirs(checkpoints_folder, exist_ok=True)

    for epoch in (pbar := tqdm.tqdm(range(n_epochs), total=n_epochs, position = 0, leave=True)):
        # Переводим сеть в режим обучения
        model.train()

        train_loss = 0
        iter_cnt = 0
        # Итерация обучения сети
        for batch in (pbar2 := tqdm.tqdm(dl_train, total=len(dl_train), position = 1, leave=False)):
            images = batch['input']
            images = torch.unsqueeze(images, 1)

            labels = batch['target']

            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)

            net_out = model(images)
            loss, mean_pos, mean_neg = loss_fn(net_out, labels)
            
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            iter_cnt += 1

            pbar2.set_description(
                'Loss Train / mean pos / mean neg: {0:.5f} / {1:.5f} / {2:.5f}\n'.format(
                    loss.item(), mean_pos , mean_neg
                )
            )
        
        train_losses.append(train_loss / iter_cnt)
        
        # Оцениваем качество модели каждые 3 итерации
        if epoch % 3 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                iter_cnt = 0
                loss_sum = 0

                for batch in (pbar2 := tqdm.tqdm(dl_test, total=len(dl_test), position = 1, leave=False)):
                    images = batch['input']
                    images = torch.unsqueeze(images, 1)

                    labels = batch['target']

                    images = images.to(device)
                    labels = labels.to(device)
                
                    net_out = model(images)
                    loss, _, __ = loss_fn(net_out, labels)

                    loss_sum += loss.item()
                    iter_cnt += 1
                    
                    pbar2.set_description(
                        'Loss Test: {0:.5f}\n'.format(
                            loss.item()
                        )
                    )
                
                test_losses.append(loss_sum / iter_cnt)
    
        checkpoint = {
            'epoch': start_epoch + epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(checkpoints_folder, f"checkpoint_{start_epoch + epoch}"))

        pbar.set_description(
                        'Loss (Train/Test): {0:.5f}/{1:.5f}\n'.format(
                            train_losses[-1], test_losses[-1]
                        )
                    )

    with open(os.path.join(experiment_folder, 'train_test_losses.json'), 'w') as fp:
        json.dump({'train_losses': train_losses,
                   'test_losses': test_losses}, fp)

    return train_losses, test_losses

def create_model(model_description):
        if 'name' not in model_description:
                return '[ERROR]: corrupted model description'

        if model_description['name'] == 'DSCNN':
                n_mels = model_description['n_mels']
                in_shape = (n_mels, 32)
                in_channels = model_description['in_channels']
                ds_cnn_number = model_description['ds_cnn_number']
                ds_cnn_size = model_description['ds_cnn_size']
                is_classifier = model_description['is_classifier']
                classes_number = 0 if not is_classifier else model_description['classes_number']

                return DSCNN(in_channels, in_shape, ds_cnn_number, ds_cnn_size, is_classifier, classes_number)


def do_experiment(experiment_settings):
        experiment_settings['time_start'] = str(datetime.now())

        background_noise_path = 'datasets/speech_commands/_background_noise_'
        train_dataset_path = 'datasets/speech_commands/train'
        valid_dataset_path = 'datasets/speech_commands/validation'

        # prepare device
        device = torch.device('cpu')
        use_gpu = False
        if torch.cuda.is_available():
                use_gpu = True
                device = torch.device('cuda', 0)
        
        print(type(device), device)

        if use_gpu:
                torch.backends.cudnn.benchmark = True

        print(f'Start experiment {experiment_settings["experiment_name"]} -- {str(datetime.now())}')

        save_config = {'experiment_name': experiment_settings['experiment_name'],
                'experiments_folder': experiment_settings['experiments_folder']}
        
        # prepare folder
        experiment_folder = os.path.join(experiment_settings['experiments_folder'], experiment_settings['experiment_name'])
        os.makedirs(experiment_folder, exist_ok=True)

        # create datasets
        n_mels = experiment_settings['model']['n_mels']

        data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])

        bg_dataset = BackgroundNoiseDataset(background_noise_path, data_aug_transform)
        add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)

        train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
        train_dataset = SpeechCommandsDataset(train_dataset_path,
                                        Compose([LoadAudio(),
                                                data_aug_transform,
                                                add_bg_noise,
                                                train_feature_transform]))

        valid_feature_transform = Compose([ToSTFT(), ToMelSpectrogramFromSTFT(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
        valid_dataset = SpeechCommandsDataset(valid_dataset_path,
                                        Compose([LoadAudio(),
                                                FixAudioLength(),
                                                valid_feature_transform]))
    
        
        # Create model
        experiment_settings['model']['classes_number'] = train_dataset.get_classes_number()
        model = create_model(experiment_settings['model'])

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        experiment_settings['model']['total_params'] = pytorch_total_params
        experiment_settings['model']['train_params'] = pytorch_train_params

        print(f'Model total params: {pytorch_total_params}')
        print(f'Model train params: {pytorch_train_params}')

        batch_size = experiment_settings['batch_size']
        n_epoch = experiment_settings['n_epoch']

        if use_gpu:
                model = torch.nn.DataParallel(model).cuda()

        # create dataloaders
        loss_settings = experiment_settings['loss']
        if loss_settings['name'] == 'CrossEntropy':
                dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
                dl_valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

                loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_settings['name'] == 'TripletLoss':
                train_sampler = TripletBatchSampler(train_dataset.get_class_indices(), batch_size, 20)
                valid_sampler = TripletBatchSampler(valid_dataset.get_class_indices(), batch_size, 20)

                dl_train = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=16)
                dl_valid = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=16)
                
                margin = loss_settings['loss_margin'] if 'loss_margin' in loss_settings.keys() else 1
                loss_agr_policy = loss_settings['loss_agr_policy'] if 'loss_agr_policy' in loss_settings.keys() else 'mean'
                
                if loss_settings['triplet_mining_strategy'] == 'batch_random':
                        loss_fn = TripletLossBatchRandom(margin=margin, loss_agr_policy=loss_agr_policy)
                elif loss_settings['triplet_mining_strategy'] == 'batch_hard':
                        loss_fn = TripletLossBatchHard(margin=margin, loss_agr_policy=loss_agr_policy)
        else:
                return

        learning_rate = experiment_settings['learning_rate']
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_losses, test_losses = training_loop(0, n_epoch, model, loss_fn, optimizer, dl_train, dl_valid, device, save_config)

        experiment_settings['time_finish'] = str(datetime.now())
        with open(os.path.join(experiment_folder, 'experiment_settings.json'), 'w') as fp:
                json.dump(experiment_settings, fp)