import models
from dataload import MultiDomainDataset, EpochsDataset

import pickle
from pathlib import Path
from layers import Linear, Sequential, LogSoftmax
from utils import *
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


SAVED_PARAMS_FORMAT = '{}_params_{}.pt'


def reserve_training_from_test(dataset: MultiDomainDataset):
    """Split off any first run for subject-specific training."""
    if isinstance(dataset, ConcatDataset):
        if len(dataset.datasets) == 1:
            training, testing = reserve_training_from_test(dataset.datasets[0])
            if dataset.domains > 1:
                training = MultiDomainDataset([training], force_num_domains=dataset.domains)
        else:
            num_train = len(dataset.datasets) // 4 if len(dataset.datasets) > 4 else 1
            testing = ConcatDataset(dataset.datasets[num_train:])
            # Inflate the representation of the training dataset, but enumerate as a single run
            training = ConcatDataset([Subset(dataset, np.arange(dataset.cummulative_sizes[num_train-1]))] *
                                     (len(dataset.datasets) // num_train))
    elif isinstance(dataset, EpochsDataset):
        training, testing = balanced_split(dataset, dataset.epochs.events[:, -1], frac=1/4)
        training = ConcatDataset([training] * 4)
    elif isinstance(dataset, TensorDataset):
        training, testing = balanced_split(dataset, dataset.tensors[1].numpy(), frac=1/4)
        training = ConcatDataset([training] * 4)
    else:
        raise ValueError('Loaded subjects must be EpochsDataLoader or MultiDomainDataset for args.use_training')
    return training, testing


def multi_split_helper(datasets: list):
    """Helper to split off training from testing and validation dataset lists"""
    split_off = list()
    tqdm.tqdm.write('Adding training data...')
    for i in range(len(datasets)):
        new_train, new_test = reserve_training_from_test(datasets[i])
        datasets[i] = new_test
        split_off.append(new_train)
    return datasets, split_off


def _get_labels_from_nested_epochs_datasets(dataset: Dataset):
    epochsdatasets = list()
    if not isinstance(dataset, EpochsDataset):
        if isinstance(dataset, TensorDataset):
            return dataset.tensors[-1].cpu().numpy()
        elif isinstance(dataset, Subset):
            epochsdatasets.append(_get_labels_from_nested_epochs_datasets(dataset.dataset)[dataset.indices])
        elif isinstance(dataset, ConcatDataset):
            epochsdatasets += [_get_labels_from_nested_epochs_datasets(d) for d in dataset.datasets]
        else:
            raise ValueError('Can only recover labels from nesting instances of Subset and ConcatDataset.')
    else:
        return dataset.epochs.events[:, -1]
    return np.concatenate(epochsdatasets)


def _get_label_balance(dataset):
    labels = _get_labels_from_nested_epochs_datasets(dataset)
    counts = np.bincount(labels)
    train_weights = 1. / torch.tensor(counts, dtype=torch.float)
    sample_weights = train_weights[labels]
    print('Class frequency: ', counts/counts.sum())
    return sample_weights, counts


def balanced_undersampling(dataset, replacement=False):
    sample_weights, counts = _get_label_balance(dataset)
    return WeightedRandomSampler(sample_weights, len(counts) * int(counts.min()), replacement=replacement)


def balanced_oversampling(dataset, replacement=True):
    sample_weights, counts = _get_label_balance(dataset)
    return WeightedRandomSampler(sample_weights, len(counts) * int(counts.max()), replacement=replacement)


def held_out_mdl(args, loaded_subjects: OrderedDict, all_subjects, subject_folds, force_test_run=None):

    base_param_dir = args.load_params

    for test_fold in tqdm.trange(args.xval_folds, unit='Test Folds', desc='Cross-Validation'):
        cv = subject_folds.copy()
        test_subjects = cv.pop(test_fold)

        iterator = tqdm.tqdm(test_subjects, desc='Target thinker', unit='subject')
        for held_out in iterator:
            print('Held out: ', held_out)
            iterator.set_postfix(dict(held_out=held_out))
            test_sets = [loaded_subjects[subject] for subject in test_subjects if subject != held_out]

            # Note that the test and train propotions are swapped from global MDL and we assume Epochs Dataset type
            added_from_test, validating = balanced_split(loaded_subjects[held_out],
                                                         _get_labels_from_nested_epochs_datasets(
                                                             loaded_subjects[held_out]
                                                         ),
                                                         frac=0.5)
            added_from_test = ConcatDataset([added_from_test] * 2)

            testing = ConcatDataset(test_sets)

            cv_validation = cv.copy()
            val_subjects = cv_validation.pop(test_fold - 1)
            # Don't add reserve validation data to focus on only effects of the target subject
            # instead validate that the target accuracy is improving

            if args.fine_tune:
                args.load_params = base_param_dir + SAVED_PARAMS_FORMAT.format(args.model, test_fold)
                training = added_from_test
                args.subjects_used = 1
                print('Fine tuning:')
            else:
                loaded = [loaded_subjects[subject] for subject in chain(*cv_validation)]
                training = MultiDomainDataset(loaded + [added_from_test])
                args.subjects_used = len(training.datasets)
                print('Training Subjects: ', [s for s in all_subjects if s not in test_subjects and s not in val_subjects])

            print('Validation Subjects: ', val_subjects)
            print('Test Subjects: ', [s for s in test_subjects if s != held_out])

            yield training, validating, testing


def lmso_split(args, loaded_subjects: OrderedDict, subject_folds=None, force_test_run=None):
    SUBJECTS = list(loaded_subjects.keys())

    if subject_folds is None:
        subject_folds = [list(x) for x in np.array_split(list(loaded_subjects.keys()), args.xval_folds)]

    if args.mdl_hold or args.fine_tune:
        yield from held_out_mdl(args, loaded_subjects, SUBJECTS, subject_folds, force_test_run=force_test_run)
        return

    for test_fold in tqdm.trange(args.xval_folds, unit='Test Folds', desc='Cross-Validation'):
        cv = subject_folds.copy()
        test_subjects = cv.pop(test_fold)
        test_sets = [loaded_subjects[subject] for subject in test_subjects]

        testing, added_from_test = multi_split_helper(test_sets) if args.use_training else [test_sets, []]
        if force_test_run is not None:
            testing = ConcatDataset([t.datasets[_i] for t in testing for _i in force_test_run])
        else:
            testing = ConcatDataset(testing)

        def xval_iter(val_index):
            cv_validation = cv.copy()
            val_subjects = cv_validation.pop(val_index)
            validating = [loaded_subjects[subject] for subject in val_subjects]
            validating, added_from_validation = multi_split_helper(validating) if args.use_training else [validating, []]
            validating = ConcatDataset(validating)

            loaded = [loaded_subjects[subject] for subject in chain(*cv_validation)]
            training = MultiDomainDataset(loaded + added_from_test + added_from_validation)
            args.subjects_used = len(training.datasets)

            print('Training Subjects: ', [s for s in SUBJECTS if s not in test_subjects and s not in val_subjects])
            print('Validation Subject: ', val_subjects)
            print('Test Subject: ', test_subjects)

            yield training, validating, testing

        if args.full_test:
            for i, val_fold in enumerate(tqdm.tqdm(cv, unit='Train Folds', desc='Validation Fold')):
                yield from xval_iter(i)
        else:
            yield from xval_iter(test_fold - 1)


def loso_split(args, loaded_subjects, force_test_run=None):
    """
    Splits the loaded subjects into training, validation and test according to LOSO procedure.
    Parameters
    ----------
    args
    loaded_subjects: OrderedDict
        Each entry in the ordered dict should be a MultiDomainDataset for compatibility with multiple runs, or just a
        pytorch-style dataset otherwise.
    force_test_run: list
        List of runs to use for the test set, e.g. [1, 2] will force the test data to only consist of runs 1 and 2. When
        used in conjunction with use-training, will use all other data as training.

    Returns
    -------
    training, validation and testing pytorch datasets. Training will be a MultiDomainDataset with subjects enumerated.
    """
    SUBJECTS = list(loaded_subjects.keys())

    def _loop(test_subject, val_subject):
        # Separate test subject
        if args.use_training:
            add_to_train, testing = reserve_training_from_test(loaded_subjects[test_subject])
        elif force_test_run is not None and isinstance(loaded_subjects[test_subject], ConcatDataset):
            testing = ConcatDataset([loaded_subjects[test_subject].datasets[_i] for _i in force_test_run])
        else:
            testing = loaded_subjects[test_subject]

        # A new OrderedDict for training subjects
        training = OrderedDict()
        for s in loaded_subjects:
            if s == test_subject:
                if args.use_training:
                    training[s] = add_to_train
                continue
            training[s] = loaded_subjects[s]

        # Separate validation subject
        val_datasets = training.pop(val_subject)
        if args.use_training:
            add_to_train, validating = reserve_training_from_test(val_datasets)
            training[val_subject] = add_to_train
        else:
            validating = val_datasets
        args.subjects_used = len(training)
        training = MultiDomainDataset(list(training.values()))

        print('Training Subjects: ', [s for s in SUBJECTS if s != test_subject and s != val_subject])
        print('Validation Subject: ', val_subject)
        print('Test Subject: ', test_subject)
        if force_test_run is not None:
            print('Only testing on runs: ', force_test_run)

        yield training, validating, testing

    if args.subject is not None:
        if isinstance(args.subject, str):
            args.subject = SUBJECTS.index(args.subject)
        if args.full_test:
            for _val in tqdm.tqdm(SUBJECTS):
                if _val != args.subject:
                    yield from _loop(SUBJECTS[args.subject], _val)
        else:
            ret_val = yield from _loop(SUBJECTS[args.subject], SUBJECTS[args.subject - 1])
            return ret_val

    for i, subject in enumerate(tqdm.tqdm(SUBJECTS)):
        if args.full_test:
            for j, _val in enumerate(tqdm.tqdm(SUBJECTS)):
                if _val != subject:
                    yield from _loop(subject, _val)
        else:
            yield from _loop(subject, SUBJECTS[i - 1])


def subject_specific(args, loaded_subjects: OrderedDict, force_test_run=None, xval=4):
    """
    Train subject specific models rather than in a LOSO/LMSO fashion.
    Parameters
    ----------
    args
    loaded_subjects
    force_test_run: list
    If specified will set aside a single run as test set, otherwise test set is selected through cross-validation
    xval: int
    Number of cross validation folds, ignored if force_test_runs is specified

    Returns
    -------

    """
    SUBJECTS = list(loaded_subjects.keys())
    args.subjects_used = 1

    def _loop(training, validating, testing):
        print('Training Points: ', len(training))
        print('Validation Points: ', len(validating))
        print('Testing Points: ', len(testing))

        yield training, validating, testing

    for i, subject in enumerate(tqdm.tqdm(SUBJECTS)):
        if args.subject:
            if args.subject != i + 1:
                continue
        print('Model for subject: ', subject)
        ds = loaded_subjects[subject]

        if force_test_run is not None:
            testing = ConcatDataset([ds.datasets[_i] for _i in force_test_run])
            ds = ConcatDataset([ds.datasets[_i] for _i in range(len(ds.datasets)) if _i not in force_test_run])
            print('Only testing on runs: ', force_test_run)
        else:
            testing = None

        for f in tqdm.trange(xval, desc='Cross Validation:'):
            folds = np.array_split(np.arange(len(ds)), xval)
            if testing is None:
                testing = Subset(ds, folds.pop(f))
                val_fold = f - 1 if f > 0 else len(folds) - 1
                tqdm.tqdm.write('Validation Fold: ' + str(val_fold))
            else:
                val_fold = f

            tqdm.tqdm.write('Validation Fold: ' + str(val_fold))
            validating = Subset(ds, folds.pop(val_fold))

            training = Subset(ds, [_i for _f in folds for _i in _f])

            yield from _loop(training, validating, testing)


def full_training(args, loaded_subjects, dataset_name, split_func=loso_split, test_metrics=None, force_test_run=None,
                  evaluation_batch_scaler=None, train_sampler=None, val_sampler=None):
    """
    Performs full training according to args and split function provided.
    Parameters
    ----------
    loaded_subjects : OrderedDict
        All the subject's datasets as an mapping between subject: [Dataset * number of runs]
    training_sample_replacement: Bool
        By default the training is done without replacement, but training can be sped when there are a lot of samples
        by setting replacement to True.

    """
    args.tcrop = args.tcrop if args.tcrop is not None else args.tlen
    print('Training with {} Channels and {} Samples'.format(args.channels, args.sfreq * args.tcrop))
    print('Alignment: ', not args.no_alignment)

    test_metrics = test_metrics if test_metrics is not None else dict()

    if args.results is not None:
        args.results_df = None

    for fold, (training, validating, testing) in enumerate(split_func(args, loaded_subjects,
                                                                      force_test_run=force_test_run)):
        _train_sampler = train_sampler(training) if train_sampler is not None else None

        _val_sampler = val_sampler(validating) if val_sampler is not None else None
        eval_batch_size = int(args.batch_size * evaluation_batch_scaler) if evaluation_batch_scaler is not None else 1

        training = DataLoader(training, sampler=_train_sampler, pin_memory=True, batch_size=args.batch_size,
                              num_workers=10, shuffle=train_sampler is None)
        validating = DataLoader(validating, sampler=_val_sampler, batch_size=eval_batch_size, num_workers=10)
        testing = DataLoader(testing, batch_size=eval_batch_size, shuffle=False, num_workers=10)

        if args.fine_tune:
            y_p, y_t, model = train_fold(args, training, validating, testing, test_metrics)
            for param in model.parameters():
                param.requires_grad = True
            args.lr, args.epochs, args.warmup = args.lr / 10, args.epochs // 2, args.warmup // 2
            y_p, y_t, model = train_fold(args, training, validating, testing, test_metrics, use_model=model)
            # Fix modified values
            args.lr, args.epochs, args.warmup = args.lr * 10, args.epochs * 2, args.warmup * 2
        else:
            y_p, y_t, model = train_fold(args, training, validating, testing, test_metrics)

        if not args.hide_test:
            acc = tensor_acc(y_t.argmax(1), y_p)
            val_stats = {k: test_metrics[k](y_t.argmax(1), y_p) for k in test_metrics}
            if args.save_params:
                state_dict = model.state_dict()
                save_location = Path('./saved_models') / args.dataset
                save_location = save_location / Path(args.results).parent.parent.name /Path(args.results).parent.name /\
                                Path(args.results).stem if args.results is not None else save_location
                save_location.mkdir(parents=True, exist_ok=True)
                save_location = save_location / SAVED_PARAMS_FORMAT.format(args.model, fold)
                torch.save(state_dict, str(save_location))
                tqdm.tqdm.write('Saved model to {}'.format(str(save_location)))
                val_stats['model-params'] = str(save_location)
            if args.results is not None:
                tqdm.tqdm.write('Writing Results to {}'.format(args.results))
                args.results_df = update_pd(
                    args.results_df, make_results_df(split_func.__name__, args.subjects_used, args.targets, acc, fold+1,
                                                     **val_stats))
                update_excel(args.results, args.results_df, '{}-{}-{}'.format(dataset_name, args.dataset, args.model))


def make_trainer_tester(model: models.Module, args, optimizer, loss_fn):

    def train(x, y, subject=None, run=None, progress=0.0):
        model.train()
        optimizer.zero_grad()

        if args.tcrop < args.tlen:
            if bool(torch.rand(1) < args.crop_p):
                offset = torch.IntTensor(1).random_(0, int(args.sfreq * (args.tlen - args.tcrop)))
            else:
                offset = int((args.teval - args.tmin) * args.sfreq)
            x = x[..., offset:int(offset + args.tcrop * args.sfreq)]

        y = one_hot(y, args.targets)

        # Label Smoothing
        if args.label_smoothing > 0:
            y -= args.label_smoothing * (y - 1/(args.targets + 1))

        # Mixup
        if args.mixup > 0:
            lam_mu = np.random.beta(args.mixup, args.mixup)
            mixers = torch.randperm(x.shape[0]).cuda()
            _x = lam_mu * x + (1 - lam_mu) * x[mixers]
            y = lam_mu * y + (1 - lam_mu) * y[mixers]
            if subject is not None:
                subject = lam_mu * subject + (1 - lam_mu) * subject[mixers]
            if run is not None:
                run = lam_mu * run + (1 - lam_mu) * run[mixers]
        else:
            _x = x

        model_out = model(_x)

        if not isinstance(model_out, dict):
            model_out = dict(prediction=model_out)
        assert 'prediction' in model_out.keys()

        loss = list()
        loss += [loss_fn(model_out['prediction'], y)]

        assert len(loss) > 0
        accumulated_loss = loss[0]
        for l in loss[1:]:
            accumulated_loss += l

        if torch.isnan(accumulated_loss):
            raise NaNError()

        accumulated_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        y = y.argmax(-1).squeeze(-1)

        return accumulated_loss.item(), tensor_acc(y, model_out['prediction'])

    def evaluate(val_loader):
        model.eval()

        preds = []
        true = []
        for vals in tqdm.tqdm(val_loader, unit='batch', desc='Evaluating'):
            val_x, val_y = vals[0].to('cuda'), vals[1].to('cuda').squeeze().long()
            val_y = one_hot(val_y, args.targets)
            if len(val_y.shape) < 2:
                val_y = val_y.unsqueeze(0)

            if args.tcrop < args.tlen:
                offset = int((args.teval - args.tmin) * args.sfreq)
                val_x = val_x[..., offset:int(offset + args.tcrop * args.sfreq)]

            model_out = model(val_x)

            if isinstance(model_out, dict):
                model_out = model_out['prediction']
            if len(model_out.shape) == 3:
                if model_out.shape[-1] == 1:
                    model_out = model_out.squeeze(-1)
                else:
                    model_out = torch.Tensor(to_categorical(
                        model_out.argmax(1).to('cpu'), model_out.shape[1]).mean(1)
                                             ).cuda()

            preds.append(model_out.detach().cpu())
            true.append(val_y.cpu())
        return torch.cat(preds).squeeze(), torch.cat(true).squeeze()

    return train, evaluate


def create_model(args):
    param_dict = pickle.load(open(args.model_param_dict, 'rb')) if args.model_param_dict is not None else dict()
    model = models.MODELS[args.model](targets=args.targets, channels=args.channels, do=args.dropout,
                                      samples=int(args.tcrop * args.sfreq), subjects=args.subjects_used,
                                      runs=args.separate_runs, **param_dict)
    if args.load_params is not None:
        params = torch.load(args.load_params)
        if getattr(model, 'restricted_param_loading'):
            model.restricted_param_loading(params, freeze=args.fine_tune)
        else:
            model.load_state_dict(params)
        print('Loaded weights from {}'.format(args.load_params))

    return model


def train_fold(args, training, validating, testing, test_metrics: dict, use_model=None):
    model = use_model if use_model is not None else create_model(args)

    args.ewma_model = None if args.ewma_model == 0 else args.ewma_model
    best_state_dict = model.state_dict() if args.ewma_model is None else ModelQueue(args.ewma_model)

    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    scheduler = CosineDecay(optimizer, args.lr, args.warmup, total_epochs=args.epochs, warm_drop=args.warmup_drop)

    loss_fn = torch.nn.KLDivLoss('batchmean',).cuda()
    train, evaluate = make_trainer_tester(model, args, optimizer, loss_fn)

    pbar = tqdm.trange(args.epochs, unit='epoch')
    best_val, best_acc, best_epoch = 100, 1 / args.targets, -1
    loss_sm, acc_sm = 10, 1 / args.targets
    # See utils.py for these
    print_stats.loss_trend = None
    print_stats.acc_trend = None

    for e in pbar:

        pbar.set_description('Epoch: {}/{}'.format(e, args.epochs))
        pbar.refresh()

        for batch in tqdm.tqdm(training, unit='batch', ):
            batch = [b.to('cuda', non_blocking=True) for b in batch]
            t_loss, t_acc = train(batch[0], batch[1].long(), subject=batch[-1] if len(batch) > 2 else None,
                                  run=batch[2] if len(batch) == 4 else None,
                                  progress=max(0, (e-args.warmup)/(args.epochs-args.warmup)))
            loss_sm = 0.98 * loss_sm + 0.02 * t_loss
            acc_sm = 0.98 * acc_sm + 0.02 * t_acc
            pbar.set_postfix(loss=t_loss, acc=t_acc, running_loss=loss_sm, running_acc=acc_sm,
                             lr=scheduler.get_lr(), best_acc=best_acc)

        y_p, y_t = evaluate(validating)

        # Done on cpu to avoid GPU memory filling
        val_loss = loss_fn.cpu()(y_p, y_t)
        val_acc = tensor_acc(y_t.argmax(1), y_p)
        val_stats = {k: test_metrics[k](y_t.argmax(1), y_p) for k in test_metrics}

        l, a = print_stats(e, val_loss, val_acc, trend_factor=0.4)
        tqdm.tqdm.write('Running Training Loss: {} Accuracy: {}'.format(loss_sm, acc_sm))
        if len(val_stats) > 0:
            tqdm.tqdm.write(str(val_stats))
        if a > best_acc or (a == best_acc and val_loss < best_val):
            best_val, best_acc, best_epoch = l, a, e + 1
            if args.ewma_model is None:
                best_state_dict = model.state_dict(best_state_dict)
            if args.val_out:
                np.savez('best_val', y_p=y_p, y_t=y_t)
        if args.ewma_model is not None:
            best_state_dict.push(model.state_dict())

        scheduler.step()

    tqdm.tqdm.write('Best Validation Accuracy: {:.2f}%, Smoothed loss: {} at Epoch {}'.format(
        best_acc * 100, best_val, best_epoch)
    )

    if not args.hide_test:
        if args.ewma_model is not None:
            best_state_dict = best_state_dict.ewma()
        model.load_state_dict(best_state_dict)
        y_p, y_t = evaluate(testing)

        tqdm.tqdm.write(
            'Final Test Accuracy: {:.2f}%, Loss: {}'.format(100 * tensor_acc(y_t.argmax(1), y_p), loss_fn.cpu()(y_p, y_t))
        )
        val_stats = {k: test_metrics[k](y_t.argmax(1), y_p) for k in test_metrics}
        if len(val_stats) > 0:
            tqdm.tqdm.write('Metrics: ' + str(val_stats))
        # tqdm.tqdm.write("Saving progress...")
        # np.savez('results/' + args.model + '/{}_test_predictions_{}'.format(args.dataset, args.subject),
        #          y_p=y_p, y_t=y_t)

    return y_p, y_t, model
