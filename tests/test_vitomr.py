import torch
from torch import nn
from acai_omr.models.models import OMREncoder, FineTuneOMREncoder, OMRDecoder, TeacherForcedViTOMR, NUM_CHANNELS, OMRLoss, ScheduledSamplingViTOMR, GRPOViTOMR
from acai_omr.train.pre_train import PE_MAX_HEIGHT, PE_MAX_WIDTH
from acai_omr.train.datasets import OlimpicDataset
from acai_omr.train.omr_teacher_force_train import MAX_LMX_SEQ_LEN, set_up_omr_teacher_force_train
from acai_omr.config import DEBUG_PRETRAINED_MAE_PATH, OLIMPIC_SYNTHETIC_ROOT_DIR, DEBUG_TEACHER_FORCED_PATH
from acai_omr.utils.utils import show_vitomr_prediction
import pytest
import copy


debug_kwargs = {"num_layers": 2, "num_heads": 1, "hidden_dim": 10, "mlp_dim": 1}
# the encoder structure used in the pre_train loop test
pretrained_debug_encoder = FineTuneOMREncoder(16, PE_MAX_HEIGHT, PE_MAX_WIDTH, 1, **debug_kwargs)
debug_mae_state_dict = torch.load(DEBUG_PRETRAINED_MAE_PATH)
debug_decoder = OMRDecoder(MAX_LMX_SEQ_LEN, "lmx_vocab.txt", **debug_kwargs)
debug_teacher_forced_vitomr = TeacherForcedViTOMR(pretrained_debug_encoder, debug_mae_state_dict, debug_decoder)

debug_teacher_forced_state_dict = torch.load(DEBUG_TEACHER_FORCED_PATH)
debug_grpo_vitomr = GRPOViTOMR(pretrained_debug_encoder, debug_teacher_forced_vitomr.transition_head, debug_decoder, debug_teacher_forced_state_dict)

_, _, base_img_transform, base_lmx_transform = set_up_omr_teacher_force_train()

vocab_len = debug_teacher_forced_vitomr.decoder.vocab_embedding.num_embeddings
 
def test_encoder_batchify():
    patch_size = 2
    hidden_dim = NUM_CHANNELS * patch_size ** 2
    encoder = OMREncoder(patch_size, PE_MAX_HEIGHT, PE_MAX_WIDTH, hidden_dim=hidden_dim, num_heads=1)
    encoder.projection = nn.Identity()
    encoder.pos_embedding = nn.Parameter(torch.ones(50, 50, hidden_dim))

    x = [torch.ones(NUM_CHANNELS, 4, 4), torch.ones(NUM_CHANNELS, 4, 8)]
    embeddings, mask = encoder.batchify(x)
    print(f"Output:\nEmbeddings:\n{embeddings}\nMask:\n{mask}")
    first_embedding_seq = torch.cat([torch.ones(NUM_CHANNELS, 4, hidden_dim) + 1, torch.zeros(NUM_CHANNELS, 4, hidden_dim)], dim=1)
    second_embedding_seq = torch.ones(NUM_CHANNELS, 8, hidden_dim) + 1
    embeddings_target = torch.cat([first_embedding_seq, second_embedding_seq])
    assert torch.equal(embeddings_target, embeddings)
    first_ex_mask = (torch.arange(8) >= 4).unsqueeze(0)
    second_ex_mask = (torch.arange(8) >= 8).unsqueeze(0)
    assert torch.equal(mask, torch.cat((first_ex_mask, second_ex_mask)))

    # test larger inputs where pe needs to be interpolated
    x = [torch.ones(NUM_CHANNELS, 10, 600)]
    embeddings, mask = encoder.batchify(x) 
    # since PE grid is all ones, should be interpolated to a grid of all ones matching input image dims
    assert torch.equal(embeddings, torch.ones(NUM_CHANNELS, 1500, hidden_dim) + 1)

def test_encoder_forward():
    encoder = pretrained_debug_encoder
    x = [torch.rand(NUM_CHANNELS, 32, 32), torch.rand(NUM_CHANNELS, 32, 64)]
    x, mask = encoder(x)
    print(f"Encodings\n{x}\nAttention mask\n{mask}")
    assert x.shape == torch.Size([2, 8, debug_kwargs["hidden_dim"]])

    # test larger inputs where pe needs to be interpolated
    x = [torch.rand(NUM_CHANNELS, 256, 4000)]
    x, mask = encoder(x)
    print(f"Encodings\n{x}\nAttention mask\n{mask}")
    assert x.shape == torch.Size([NUM_CHANNELS, 4000, debug_kwargs["hidden_dim"]])

def test_decoder():
    hidden_dim = 10
    decoder = OMRDecoder(MAX_LMX_SEQ_LEN, "lmx_vocab.txt", hidden_dim=hidden_dim, num_heads=1, num_layers=1, mlp_dim=1)

    # test vocabulary creation
    assert decoder.vocab_embedding.weight.shape == torch.Size([vocab_len, hidden_dim])

    # test forward 
    decoder.pos_embedding = nn.Parameter((torch.ones(MAX_LMX_SEQ_LEN, dtype=torch.float) + 500).unsqueeze(-1).repeat(1, hidden_dim))
    decoder.vocab_embedding.weight = nn.Parameter(torch.arange(vocab_len, dtype=torch.float).unsqueeze(-1).repeat(1, hidden_dim))
    print(f"Positional embedding grid\n{decoder.pos_embedding}\nEmbedding weights{decoder.vocab_embedding.weight}")

    input_seqs = [torch.tensor([0, 2, 3]), torch.tensor([0, 2, 2, 3, 4])]
    input_seqs = torch.nested.as_nested_tensor(input_seqs)
    input_seqs = input_seqs.to_padded_tensor(padding=decoder.padding_idx)
    lmx_attention_mask = torch.arange(5).unsqueeze(0).repeat(2, 1) >= torch.tensor([3, 5]).reshape(2, 1)

    img_latent = torch.ones(2, 10, hidden_dim)
    latent_attention_mask = torch.cat([
        torch.cat([torch.tensor([False]).repeat(7), torch.tensor([True]).repeat(3)]).unsqueeze(0),
        torch.tensor([False]).repeat(10).unsqueeze(0)
    ], dim=0)
    print(f"Input lmx sequences\n{input_seqs}\nImage latent encoding\n{img_latent}\nLmx mask\n{lmx_attention_mask}\nLatent mask\n{latent_attention_mask}")

    pred = decoder(input_seqs, img_latent, lmx_attention_mask, latent_attention_mask)
    print(f"Decoder output prediction: {pred}")
    assert pred.shape == torch.Size([2, 5, vocab_len])

def test_decoder_gradient_flow():
    hidden_dim = 10
    decoder = OMRDecoder(MAX_LMX_SEQ_LEN, "lmx_vocab.txt", hidden_dim=hidden_dim, num_heads=1, num_layers=1, mlp_dim=1)
    loss_fn = OMRLoss(decoder.padding_idx)

    lmx_seqs = [torch.tensor([0, 2, 3, 226]), torch.tensor([0, 2, 2, 3, 4, 226])]
    lmx_seqs = torch.nested.as_nested_tensor(lmx_seqs)
    lmx_seqs = lmx_seqs.to_padded_tensor(padding=decoder.padding_idx)
    input_seqs = lmx_seqs[:, :-1]
    target_seqs = lmx_seqs[:, 1:]
    lmx_attention_mask = torch.ones_like(input_seqs).bool()

    img_latent = torch.ones(2, 10, hidden_dim)
    latent_attention_mask = torch.cat([
        torch.cat([torch.tensor([False]).repeat(7), torch.tensor([True]).repeat(3)]).unsqueeze(0),
        torch.tensor([False]).repeat(10).unsqueeze(0)
    ], dim=0)
    print(f"Input lmx sequences\n{input_seqs}\nImage latent encoding\n{img_latent}\nLatent mask\n{latent_attention_mask}")
    before = decoder.pos_embedding.clone().detach()
 
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01)
    pred = decoder(input_seqs, img_latent, lmx_attention_mask, latent_attention_mask)
    print(f"Target lmx sequences\n{target_seqs}")
    loss = loss_fn(pred, target_seqs)
    loss.backward()
    grad = decoder.pos_embedding.grad.clone().detach()
    optimizer.step()
    optimizer.zero_grad()
    after = decoder.pos_embedding

    assert not torch.equal(before, after)
    assert torch.sum(grad[5:, :]) == 0
    assert torch.equal(before[5:, :], after[5:, :])
 
    # one example with just padding indices at end, ensure no update
    lmx_seqs = [torch.tensor([0, 226, 1, 1])]
    lmx_seqs = torch.nested.as_nested_tensor(lmx_seqs)
    lmx_seqs = lmx_seqs.to_padded_tensor(padding=decoder.padding_idx)
    input_seqs = lmx_seqs[:, :-1]
    target_seqs = lmx_seqs[:, 1:]
    lmx_attention_mask = torch.ones_like(input_seqs).bool()

    img_latent = torch.ones(1, 4, hidden_dim)
    latent_attention_mask = torch.tensor([True]).repeat(4).unsqueeze(0)
    print(f"Input lmx sequences\n{input_seqs}\nImage latent encoding\n{img_latent}\nLatent mask\n{latent_attention_mask}")
    before = decoder.pos_embedding.clone().detach()

    pred = decoder(input_seqs, img_latent, lmx_attention_mask, latent_attention_mask)
    print(f"Target lmx sequences\n{target_seqs}")
    loss = loss_fn(pred, target_seqs)
    loss.backward()
    grad = decoder.pos_embedding.grad.clone().detach()
    optimizer.step()
    optimizer.zero_grad()
    after = decoder.pos_embedding

    assert torch.sum(grad[1:, :]) == 0
    assert torch.equal(before[1:, :], after[1:, :])

def test_batchify_and_split_lmx_seqs():
    vitomr = debug_teacher_forced_vitomr

    lmx_seqs = [torch.tensor([0, 2, 3, 226]), torch.tensor([0, 2, 2, 3, 4, 226])]
    input_seqs, target_seqs, mask = vitomr.batchify_and_split_lmx_seqs(lmx_seqs, "cpu")
    print(f"Input sequences\n{input_seqs}\nTarget sequences\n{target_seqs}\nAttention mask\n{mask}")

    print(torch.cat([lmx_seqs[0][:-1], torch.tensor(vitomr.decoder.padding_idx).repeat(2)]))
    input_seqs_expected = torch.cat([
        torch.cat([lmx_seqs[0], torch.tensor([vitomr.decoder.padding_idx])]).unsqueeze(0),
        lmx_seqs[1][:-1].unsqueeze(0)
    ], dim=0)

    target_seqs_expected = torch.cat([
        torch.cat([lmx_seqs[0][1:], torch.tensor([vitomr.decoder.padding_idx]).repeat(2)]).unsqueeze(0),
        lmx_seqs[1][1:].unsqueeze(0)
    ], dim=0)

    assert torch.equal(input_seqs, input_seqs_expected)
    assert torch.equal(target_seqs, target_seqs_expected)
    # it's fine for the non-max length input sequences to have their <eos> tokens leftover since the causal mask
    # will prevent earlier tokens attending to it and the position will be padded out in the loss calculation
    assert torch.equal(mask[0, :], torch.tensor([False, False, False, False, False, True]))
    assert torch.equal(mask[1, :], torch.tensor([False, False, False, False, False, False]))

def test_vitomr():
    vitomr = debug_teacher_forced_vitomr
    optimizer = torch.optim.SGD(vitomr.parameters(), lr=0.1)
    loss_fn = OMRLoss(vitomr.decoder.padding_idx)

    encoder_before = {name: param.clone().detach() for name, param in vitomr.encoder.named_parameters()}
    x = [(torch.rand(NUM_CHANNELS, 64, 128), torch.randint(high=vocab_len, size=(8,))),
         (torch.rand(NUM_CHANNELS, 32, 32), torch.randint(high=vocab_len, size=(6,)))]
    pred, target_seqs = debug_teacher_forced_vitomr(x)
    loss = loss_fn(pred, target_seqs)
    loss.backward()
    optimizer.step()
    print(f"Prediction\n{pred}\nTarget sequences\n{target_seqs}")
    assert pred.shape == torch.Size([2, 7, vocab_len])
    encoder_after = {name: param.clone().detach() for name, param in vitomr.encoder.named_parameters()}
    
    # ensure encoder is frozen
    for name, param in encoder_before.items():
        assert torch.equal(param, encoder_after[name])

def test_show_vitomr_prediction():
    vitomr = debug_teacher_forced_vitomr

    debug_dataset = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform)
    show_vitomr_prediction(vitomr, debug_dataset[0], "vitomr_prediction_test")

def test_partial_fine_tune():
    encoder = FineTuneOMREncoder(16, PE_MAX_HEIGHT, PE_MAX_WIDTH, 1, **debug_kwargs)
    vitomr = TeacherForcedViTOMR(encoder, debug_mae_state_dict, debug_decoder)
    optimizer = torch.optim.SGD(vitomr.parameters(), lr=10)
    loss_fn = OMRLoss(vitomr.decoder.padding_idx)

    encoder_before = {name: param.clone() for name, param in vitomr.encoder.named_parameters()}
    x = [(torch.rand(NUM_CHANNELS, 64, 128), torch.randint(high=vocab_len, size=(8,))),
         (torch.rand(NUM_CHANNELS, 32, 32), torch.randint(high=vocab_len, size=(6,)))]
    pred, target_seqs = vitomr(x)
    loss = loss_fn(pred, target_seqs)
    loss.backward()
    optimizer.step()
    print(f"Prediction\n{pred}\nTarget sequences\n{target_seqs}")
    assert pred.shape == torch.Size([2, 7, vocab_len])
    encoder_after = {name: param for name, param in vitomr.encoder.named_parameters()}

    # ensure encoder finetuning is working properly
    for name, param in encoder_before.items():
        if "frozen" in name:
            assert not param.requires_grad
            assert torch.equal(param, encoder_after[name])
        elif "pos_embedding" in name or "projection" in name:
            assert not param.requires_grad
            assert torch.equal(param, encoder_after[name])
        else:
            assert param.requires_grad
            assert not torch.equal(param, encoder_after[name])

def test_create_param_groups():
    # partial fine tune
    encoder = FineTuneOMREncoder(16, PE_MAX_HEIGHT, PE_MAX_WIDTH, 1, **debug_kwargs)
    vitomr = TeacherForcedViTOMR(encoder, debug_mae_state_dict, debug_decoder)
    
    base_lr = 100.0
    base_fine_tune_lr = 50.0
    lr_decay = 0.5
    param_groups, _ = vitomr.create_fine_tune_param_groups(base_lr, base_fine_tune_lr, lr_decay)
    
    # map all parameters in all the created groups to their group lr
    param_to_lr = {}
    for group in param_groups:
        for param in group["params"]:
            param_to_lr[param] = group["lr"]

    print("PARTIAL FINE TUNE")
    print(f"{'Parameter Name':<70} {'LR':<10}\n{'-' * 80}")
    for name, param in vitomr.named_parameters():
        if param.requires_grad:
            # check parameter was assigned to a group/lr
            lr = param_to_lr.get(param, None)
            assert lr
            print(f"{name:<70} {lr:<10}")
            # check lrs look correct for each group
            if "decoder" in name or "transition" in name:
                assert lr == base_lr
            else:
                assert lr <= base_fine_tune_lr

    # full fine tune
    encoder = FineTuneOMREncoder(16, PE_MAX_HEIGHT, PE_MAX_WIDTH, 2, **debug_kwargs)
    vitomr = TeacherForcedViTOMR(encoder, debug_mae_state_dict, debug_decoder)
    
    base_lr = 100.0
    base_fine_tune_lr = 50.0
    lr_decay = 0.5
    param_groups, _ = vitomr.create_fine_tune_param_groups(base_lr, base_fine_tune_lr, lr_decay)
    
    # map all parameters in all the created groups to their group lr
    param_to_lr = {}
    for group in param_groups:
        for param in group["params"]:
            param_to_lr[param] = group["lr"]

    print("FULL FINE TUNE")
    print(f"{'Parameter Name':<70} {'LR':<10}\n{'-' * 80}")
    for name, param in vitomr.named_parameters():
        if param.requires_grad:
            # check parameter was assigned to a group/lr
            lr = param_to_lr.get(param, None)
            assert lr
            print(f"{name:<70} {lr:<10}")
            # check lrs look correct for each group
            if "decoder" in name or "transition" in name:
                assert lr == base_lr
            else:
                assert lr <= base_fine_tune_lr

def test_fine_tune_with_llrd():
    # partial fine-tune
    encoder = FineTuneOMREncoder(16, PE_MAX_HEIGHT, PE_MAX_WIDTH, 1, **debug_kwargs)
    print(encoder)
    vitomr = TeacherForcedViTOMR(encoder, debug_mae_state_dict, debug_decoder)
    loss_fn = OMRLoss(vitomr.decoder.padding_idx)
    param_groups, _ = vitomr.create_fine_tune_param_groups(100.0, 50.0, 0.99)
    optimizer = torch.optim.SGD(param_groups)

    vitomr_before = {name: param.clone() for name, param in vitomr.named_parameters()}
    x = [(torch.rand(NUM_CHANNELS, 64, 128), torch.randint(high=vocab_len, size=(8,))),
         (torch.rand(NUM_CHANNELS, 32, 32), torch.randint(high=vocab_len, size=(6,)))]
    pred, target_seqs = vitomr(x)
    loss = loss_fn(pred, target_seqs)
    loss.backward()
    optimizer.step()

    print("PARTIAL FINE-TUNE")
    for name, param in vitomr.named_parameters():
        param_before = vitomr_before[name]
        if any(x in name for x in ["decoder", "fine_tune", "transition_head"]):
            print(f"Trainable: {name}")
            assert not torch.equal(param, param_before)
        else:
            print(f"Frozen: {name}")
            assert torch.equal(param, param_before)

    # full fine-tune
    encoder = FineTuneOMREncoder(16, PE_MAX_HEIGHT, PE_MAX_WIDTH, 2, **debug_kwargs)
    print(encoder)
    vitomr = TeacherForcedViTOMR(encoder, debug_mae_state_dict, debug_decoder)
    loss_fn = OMRLoss(vitomr.decoder.padding_idx)
    param_groups, _ = vitomr.create_fine_tune_param_groups(100.0, 50.0, 0.99)
    optimizer = torch.optim.SGD(param_groups)

    vitomr_before = {name: param.clone() for name, param in vitomr.named_parameters()}
    x = [(torch.rand(NUM_CHANNELS, 64, 128), torch.randint(high=vocab_len, size=(8,))),
         (torch.rand(NUM_CHANNELS, 32, 32), torch.randint(high=vocab_len, size=(6,)))]
    pred, target_seqs = vitomr(x)
    loss = loss_fn(pred, target_seqs)
    loss.backward()
    optimizer.step()

    print("FULL FINE-TUNE")
    for name, param in vitomr.named_parameters():
        param_before = vitomr_before[name]
        print(f"Trainable: {name}")
        assert not torch.equal(param, param_before)

def test_sample_and_mix_seqs():
    vitomr = ScheduledSamplingViTOMR(pretrained_debug_encoder, debug_mae_state_dict, debug_decoder)

    teacher_forcing_prob = 0.8
    sample_tau = 0.1
    hard = False
    tf_input_seqs = torch.full([1, 5], dtype=torch.long, fill_value=10)
    tf_input_seqs[:, 0] = vitomr.decoder.bos_idx
    tf_pred_logits = torch.full([1, 5, vocab_len], fill_value=5.0)
    tf_pred_logits[:, :, 2] = 100.0 # set token at index 2 in vocab to have highest probability

    print(f"Input token indices:\n{tf_input_seqs}\nFirst pass predicted vocab distributions:\n{tf_pred_logits}")
    mixed_seqs = vitomr.sample_and_mix_seqs(teacher_forcing_prob, tf_input_seqs, tf_pred_logits, sample_tau, hard, "cpu")
    print(f"Result from sampling and mixing:\n{mixed_seqs}")
    assert mixed_seqs.shape == torch.Size([1, 5, vitomr.encoder.hidden_dim])
    assert torch.equal(mixed_seqs[:, 0, :], vitomr.decoder.vocab_embedding(torch.tensor([vitomr.decoder.bos_idx])))

    teacher_forcing_prob = 0
    hard = True
    mixed_seqs = vitomr.sample_and_mix_seqs(teacher_forcing_prob, tf_input_seqs, tf_pred_logits, sample_tau, hard, "cpu")
    print(f"Result from hard-sampling all positions:\n{mixed_seqs}")
    # <bos> token embedding should still be there and different from all others, but everything else should be the same un-mixed embedding since essentially
    # just indexing the vocab embedding matrix
    assert torch.equal(mixed_seqs[:, 0, :], vitomr.decoder.vocab_embedding(torch.tensor([vitomr.decoder.bos_idx])))
    assert torch.equal(mixed_seqs[:, 1:, :], vitomr.decoder.vocab_embedding(torch.tensor([2])).expand(4, -1).unsqueeze(0))

def test_scheduled_sampling_vitomr():
    encoder = FineTuneOMREncoder(16, PE_MAX_HEIGHT, PE_MAX_WIDTH, 2, **debug_kwargs)
    vitomr = ScheduledSamplingViTOMR(encoder, debug_mae_state_dict, debug_decoder)

    loss_fn = OMRLoss(vitomr.decoder.padding_idx)
    param_groups, _ = vitomr.create_fine_tune_param_groups(100.0, 50.0, 0.99)
    optimizer = torch.optim.SGD(param_groups)

    x = [(torch.rand(NUM_CHANNELS, 64, 128), torch.randint(high=vocab_len, size=(8,))),
         (torch.rand(NUM_CHANNELS, 32, 32), torch.randint(high=vocab_len, size=(6,)))]
    pred, target_seqs = vitomr.forward_train(x, 0.7, 0.5, False)
    loss = loss_fn(pred, target_seqs)
    loss.backward()
    optimizer.step()

def test_grpo_conversion():
    # test weights were transferred successfully
    print(debug_grpo_vitomr)
    assert torch.equal(debug_teacher_forced_state_dict["encoder.frozen_blocks.layers.0.self_attn.in_proj_weight"], debug_grpo_vitomr.encoder.encoder_blocks.layers[0].self_attn.in_proj_weight)
    assert torch.equal(debug_teacher_forced_state_dict["encoder.fine_tune_blocks.layers.0.self_attn.in_proj_weight"], debug_grpo_vitomr.encoder.encoder_blocks.layers[1].self_attn.in_proj_weight)
    for param in debug_grpo_vitomr.encoder.parameters():
        assert not param.requires_grad
    for param in debug_grpo_vitomr.transition_head.parameters():
        assert not param.requires_grad

def test_expand_img_tensors():
    vitomr = GRPOViTOMR(pretrained_debug_encoder, debug_mae_state_dict, debug_decoder)
    seq_len = 3
    encoder_hidden_dim = vitomr.encoder.hidden_dim

    first_latent = torch.zeros(seq_len, encoder_hidden_dim) 
    second_latent = torch.ones(seq_len, encoder_hidden_dim)
    img_latent = torch.stack([first_latent, second_latent])

    first_mask = torch.tensor([False] * 3)
    second_mask = torch.tensor([False, False, True])
    mask = torch.stack([first_mask, second_mask])
    print(f"Image latent and mask before expansion:\n{img_latent}\n{mask}")

    num_rollouts = 2
    img_latent, mask = vitomr.expand_img_tensors_for_rollout(img_latent, mask, num_rollouts)

    print(f"After expansion:\n{img_latent}\n{mask}")
    img_latent_expected = torch.cat([first_latent.repeat(num_rollouts, 1, 1), second_latent.repeat(num_rollouts, 1, 1)])
    mask_expected = torch.cat([first_mask.repeat(num_rollouts, 1), second_mask.repeat(num_rollouts, 1)])

    assert torch.equal(img_latent, img_latent_expected)
    assert torch.equal(mask, mask_expected)

test_args = [
    # test both sequences ended at the same time
    (torch.tensor([
        [debug_grpo_vitomr.decoder.bos_idx, 10, 10, debug_grpo_vitomr.decoder.eos_idx],
        [debug_grpo_vitomr.decoder.bos_idx, 20, 20, debug_grpo_vitomr.decoder.eos_idx],
    ]), torch.full([2, 4], fill_value=True)),
    # test one sequence ended earlier with a junk eos after
    (torch.tensor([
        [debug_grpo_vitomr.decoder.bos_idx, debug_grpo_vitomr.decoder.eos_idx, 10, debug_grpo_vitomr.decoder.eos_idx],
        [debug_grpo_vitomr.decoder.bos_idx, 20, 20, debug_grpo_vitomr.decoder.eos_idx],
    ]), torch.tensor([[True, True, False, False], [True] * 4])),
    # test neither sequence has an eos
    (torch.tensor([
        [debug_grpo_vitomr.decoder.bos_idx, 10, 10, 10],
        [debug_grpo_vitomr.decoder.bos_idx, 20, 20, 20],
    ]), torch.full([2, 4], fill_value=True)),
    # test one sequence ended early, one never did
    (torch.tensor([
        [debug_grpo_vitomr.decoder.bos_idx, 20, 20, 20],
        [debug_grpo_vitomr.decoder.bos_idx, debug_grpo_vitomr.decoder.eos_idx, 10, 10],
    ]), torch.tensor([[True] * 4, [True, True, False, False]])),
]
@pytest.mark.parametrize("rollouts, expected_mask", test_args)
def test_rollout_masking(rollouts, expected_mask):
    vitomr = GRPOViTOMR(pretrained_debug_encoder, debug_mae_state_dict, debug_decoder)

    mask = vitomr.create_rollout_mask(rollouts)
    assert torch.equal(mask, expected_mask)

test_args = [
    # test both sequences ended at the same time
    (torch.tensor([
        [debug_grpo_vitomr.decoder.bos_idx, 10, 10, debug_grpo_vitomr.decoder.eos_idx],
        [debug_grpo_vitomr.decoder.bos_idx, 20, 20, debug_grpo_vitomr.decoder.eos_idx],
    ]), torch.tensor([4, 4])),
    # test one sequence ended earlier with a junk eos after
    (torch.tensor([
        [debug_grpo_vitomr.decoder.bos_idx, debug_grpo_vitomr.decoder.eos_idx, 10, debug_grpo_vitomr.decoder.eos_idx],
        [debug_grpo_vitomr.decoder.bos_idx, 20, 20, debug_grpo_vitomr.decoder.eos_idx],
    ]), torch.tensor([2, 4])),
    # test neither sequence has an eos
    (torch.tensor([
        [debug_grpo_vitomr.decoder.bos_idx, 10, 10, 10],
        [debug_grpo_vitomr.decoder.bos_idx, 20, 20, 20],
    ]), torch.tensor([4, 4])),
    # test one sequence ended early, one never did
    (torch.tensor([
        [debug_grpo_vitomr.decoder.bos_idx, 20, 20, 20],
        [debug_grpo_vitomr.decoder.bos_idx, debug_grpo_vitomr.decoder.eos_idx, 10, 10],
    ]), torch.tensor([4, 2])),
]
@pytest.mark.parametrize("rollouts, expected_score", test_args)
def test_score_rollouts(rollouts, expected_score):
    vitomr = GRPOViTOMR(pretrained_debug_encoder, debug_mae_state_dict, debug_decoder)

    log_probs = torch.ones([2, 4], dtype=torch.float)

    scores, _ = vitomr.score_rollouts(rollouts, log_probs)
    assert torch.equal(scores, expected_score)

def test_rollout_policy():
    vitomr = GRPOViTOMR(pretrained_debug_encoder, debug_mae_state_dict, debug_decoder)
    batch_size = 2
    seq_len = 4
    encoder_hidden_dim = vitomr.encoder.hidden_dim

    img_latent = torch.rand([batch_size, seq_len, encoder_hidden_dim])
    latent_mask = torch.full([img_latent.shape[0], img_latent.shape[1]], fill_value=False)

    num_rollouts = 3 
    rollouts, rollout_scores, rollout_entropies = vitomr.forward_rollout_policy(img_latent, latent_mask, num_rollouts, max_actions=10)
    print(f"Rollouts:\n{rollouts}\nScores:\n{rollout_scores}\nEntropies:\n{rollout_entropies}")

def test_prepare_rollouts():
    vitomr = GRPOViTOMR(pretrained_debug_encoder, debug_mae_state_dict, debug_decoder)
    
    # basic case: non-ragged, terminated sequences
    rollouts = torch.tensor([[vitomr.decoder.bos_idx, 10, 10, vitomr.decoder.eos_idx], [vitomr.decoder.bos_idx, 10, 10, vitomr.decoder.eos_idx]])
    rollout_mask = vitomr.create_rollout_mask(rollouts)

    expected_rollouts = copy.deepcopy(rollouts[:, :-1])
    expected_mask = torch.tensor([[False] * 3, [False] * 3])

    print(f"Before:\nRollouts:\n{rollouts}\nMask:\n{rollout_mask}")
    rollouts, rollout_mask = vitomr.prepare_rollouts_for_policy_theta(rollouts, rollout_mask)
    print(f"After:\nRollouts:\n{rollouts}\nMask:\n{rollout_mask}")
    assert torch.equal(rollouts, expected_rollouts)
    assert torch.equal(rollout_mask, expected_mask)

    # ragged <eos>'s
    rollouts = torch.tensor([[vitomr.decoder.bos_idx, 10, 10, vitomr.decoder.eos_idx], [vitomr.decoder.bos_idx, vitomr.decoder.eos_idx, 30, 30]])
    rollout_mask = vitomr.create_rollout_mask(rollouts)

    expected_rollouts = copy.deepcopy(rollouts[:, :-1])
    expected_rollouts[1, 1:] = vitomr.decoder.pad_idx # one pad token created
    expected_mask = torch.tensor([[False] * 3, [False, True, True]])

    print(f"Before:\nRollouts:\n{rollouts}\nMask:\n{rollout_mask}")
    rollouts, rollout_mask = vitomr.prepare_rollouts_for_policy_theta(rollouts, rollout_mask)
    print(f"After:\nRollouts:\n{rollouts}\nMask:\n{rollout_mask}")
    assert torch.equal(rollouts, expected_rollouts)
    assert torch.equal(rollout_mask, expected_mask)

    # one sequence with no <eos>
    rollouts = torch.tensor([[vitomr.decoder.bos_idx, 10, vitomr.decoder.eos_idx, 30], [vitomr.decoder.bos_idx, 10, 10, 10]])
    rollout_mask = vitomr.create_rollout_mask(rollouts)

    expected_rollouts = copy.deepcopy(rollouts[:, :-1])
    print(f"EXPECTED: {expected_rollouts}")
    expected_rollouts[0, 2] = vitomr.decoder.pad_idx
    expected_mask = torch.tensor([[False, False, True], [False] * 3])

    print(f"Before:\nRollouts:\n{rollouts}\nMask:\n{rollout_mask}")
    rollouts, rollout_mask = vitomr.prepare_rollouts_for_policy_theta(rollouts, rollout_mask)
    print(f"After:\nRollouts:\n{rollouts}\nMask:\n{rollout_mask}")
    print(f"EXPECTED: {expected_rollouts}")
    assert torch.equal(rollouts, expected_rollouts)
    assert torch.equal(rollout_mask, expected_mask)

    # no <eos>'s
    rollouts = torch.tensor([[vitomr.decoder.bos_idx, 10, 10, 10], [vitomr.decoder.bos_idx, 10, 10, 10]])
    rollout_mask = vitomr.create_rollout_mask(rollouts)

    expected_rollouts = copy.deepcopy(rollouts[:, :-1])
    expected_mask = torch.tensor([[False] * 3, [False] * 3])

    print(f"Before:\nRollouts:\n{rollouts}\nMask:\n{rollout_mask}")
    rollouts, rollout_mask = vitomr.prepare_rollouts_for_policy_theta(rollouts, rollout_mask)
    print(f"After:\nRollouts:\n{rollouts}\nMask:\n{rollout_mask}")
    assert torch.equal(rollouts, expected_rollouts)
    assert torch.equal(rollout_mask, expected_mask)

if __name__ == "__main__":
    test_grpo_conversion()
