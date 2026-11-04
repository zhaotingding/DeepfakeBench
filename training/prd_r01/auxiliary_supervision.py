"""联合主头实验共用的训练损失及可训练参数身份。"""
import hashlib

from torch.nn import functional as F


def training_loss(output, labels, auxiliary_input='none', auxiliary_lambda=1.0):
    main_ce = F.cross_entropy(output['logits'], labels)
    if auxiliary_input == 'none':
        if 'auxiliary_logits' in output:
            raise ValueError('无辅助协议意外产生了辅助预测')
        return main_ce, main_ce, None
    if auxiliary_input not in {'original_aux', 'residual_aux'} or auxiliary_lambda != 1.0:
        raise ValueError('辅助监督输入或固定系数不符合本批协议')
    auxiliary_ce = F.cross_entropy(output['auxiliary_logits'], labels)
    return main_ce + auxiliary_lambda * auxiliary_ce, main_ce, auxiliary_ce


def trainable_identity(model):
    all_parameters = hashlib.sha256()
    main_parameters = hashlib.sha256()
    auxiliary_parameters = hashlib.sha256()
    main_count = 0
    auxiliary_count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('auxiliary_head.'):
            partition = auxiliary_parameters
            auxiliary_count += parameter.numel()
        else:
            partition = main_parameters
            main_count += parameter.numel()
        value = parameter.detach().cpu().numpy().tobytes()
        for digest in (all_parameters, partition):
            digest.update(name.encode())  # 沿用原引擎参数名、顺序和原始张量字节。
            digest.update(value)
    return dict(trainable_initial_sha256=all_parameters.hexdigest(),
                trainable_parameters=main_count+auxiliary_count,
                main_trainable_initial_sha256=main_parameters.hexdigest(),
                main_trainable_parameters=main_count,
                auxiliary_initial_sha256=auxiliary_parameters.hexdigest() if auxiliary_count else None,
                auxiliary_parameters=auxiliary_count)
