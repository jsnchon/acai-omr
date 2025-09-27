# vitomr

setup instructions:
setup script at least assumes ubuntu 24.04 (for installing the right python version)
simplest is to add add export PATH="/root/.local/bin:$PATH" to shell config file, pjust run setup script, transfer model weights and/or dataset, install musescore3 and imagemagick through cli (not part of setup script since only needed for inference/requires sudo). Can run flask app if developing locally with poetry run flask --app acai_omr run
sudo apt install musescore3
sudo apt install imagemagick

also need to make sure git clone with submodules flag or init them after pulling/cloning

training stuff will be helped along by new machine setup util script, need to also download data

can run as flask for developing using poetry run flask --app acai_omr run or gunicorn for production using poetry run gunicorn "acai_omr.wsgi:app" --bind 0.0.0.0:8000

digital ocean server sets up gunicorn as a systemd service bound to a unix domain socket file, nginx reverse proxy, certbot to set up DNS

important papers in this project:
mae are scalable vision learners
attention is all you need
(maybes:)
scheduled sampling for transformers 
Differentiable Scheduled Sampling for Credit Assignment
CATEGORICAL REPARAMETERIZATION
WITH GUMBEL-SOFTMAX
DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models (GRPO)

rl phase notes:
Reward function
R = lambda_t * TEDn_score + lambda_w * well_formedness + lambda_f * token level f1 score - lambda_r * repeat_penalty - lambda_l * len_penalty
All components are squished to [0, 1] to allow easy weighting of each with the lambda/beta terms
TEDn_score = exp(-alpha_t * TEDn), alpha_t contorls steepness (< 1 since TEDn can be high)
well-formedness = -gamma if generated sequence is unparseable, otherwise exp(-alpha_w * num_errs) where num_errs is the number of minor syntactic but parsable errors
f1 score is the standard f1 token similarity metric. Adds some token level smoothness to the otherwise sparse reward
repeat_penalty = 1/(4 * (|y| - 1)) * sum of consecutive 2, 3, and 4-gram repeats in the generated sequence. |y - 1| term gives a value between [0, 1] that acts as an average number of repeats per possible repeat opportunity
len_penalty = 0 if generated sequence length is within a threshold delta of the target sequence length. Otherwise, define tau = the length difference at/beyond which we clip penalty to the max of 1. The penalty = clip(exp((ln2/tau) * x-1), [0, 1]). Basically, higher errors are punished exponentially up to a point tau, at which we clip to 1 to prevent exploding penalties
Curriculum
Start with a high temperature and top_k in the policy rollouts, a decently high beta, a medium max_actions, and a low lambda_l. The thought process is to allow the model in the early stages to focus on learning good structure over medium-length sequences. Truncated sequences will have lower reward but that can help the model learn not to truncate and it'll focus on doing the best it can with its budget. Keep these values for a brief phase
Then start annealing temperature, top_k, and beta to move from exploration -> exploitation. Also slowly increase max_actions and lambda_l to force the model to start learning how to generate long sequences

Starting values:
λ_t = 0.35 (TED score — encourages global correctness)

λ_f = 0.25 (token F1 — local correctness)

λ_w = 0.10 (wellformedness)

λ_r = 0.15 (repeat penalty)

λ_l start = 0.02 (len penalty low initially) → anneal up to 0.35 over curriculum

β (entropy) start = 0.20 → anneal down to 0.02

Define L_reward using this stuff, then add L_CE (teacher-forced CE loss using policy being updated logits) to it with a weight lambda_CE (low, and also anneal it over time, eventually leaving it out entirely. Just want to keep the model grounded esp in beginning and esp since using no KL divergence since pre-trained state sucks)

alpha_t = 0.01, reward is 0.5 at around 70
alpha_w = 0.2, reward is 0.5 at around 3
gamma = 3, catastrophic mal-formedness is 3x worse than perfect well-formedness
delta = 10, tau = 100: allow some leeway in length, length diffs past 100 are all equally (really) bad

GRPO objective
entropy_bonus = 1/(T * lnV) * sum of policy entropy at each time step during generation. The average entropy squished to [0, 1]
