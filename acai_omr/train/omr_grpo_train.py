import torch
from acai_omr.train.omr_teacher_force_train import set_up_omr_teacher_force_train
from acai_omr.models.models import GRPOViTOMR

TEACHER_FORCED_STATE_DICT_PATH = ""

# policy_theta is the policy to be used in scoring rollouts from the old_policy and updated
def train_loop(old_policy: GRPOViTOMR, policy_theta: GRPOViTOMR):
    num_rollouts_per_img = 3
    grpo_epochs = 4

    # rollout using old policy
    with torch.no_grad():
        img_latent, latent_attention_mask = old_policy.expand_img_tensors_for_rollout(img_latent, latent_attention_mask, num_rollouts_per_img)
        rollouts, old_policy_log_probs, rollout_mask = old_policy.rollout_policy(img_latent, latent_attention_mask)

    # reward rollouts and calculate group-normalized advantages

    # calculate cumulative log-probs for rollouts to calculate ratios against old policy
    rollouts, rollout_attention_mask = old_policy.prepare_rollouts_for_policy_theta(rollouts, rollout_mask)
    for i in range(grpo_epochs):
        # generate next token logits at each time step through treating rollouts like a teacher forcing step
        logits = policy_theta.decoder(rollouts, img_latent, rollout_attention_mask, latent_attention_mask)

TEACHER_FORCED_STATE_DICT_PATH = "debug_omr_train/debug_vitomr.pth"

teacher_forced_vitomr, base_img_transform, base_lmx_transform, device = set_up_omr_teacher_force_train()
encoder = teacher_forced_vitomr.encoder
transition_head = teacher_forced_vitomr.transition_head
decoder = teacher_forced_vitomr.decoder

teacher_forced_state_dict = torch.load(TEACHER_FORCED_STATE_DICT_PATH)

vitomr = GRPOViTOMR(encoder, transition_head, decoder, teacher_forced_state_dict)
print(vitomr)
